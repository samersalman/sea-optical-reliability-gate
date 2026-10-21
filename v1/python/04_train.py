"""Train the class-agnostic instrument segmenters (U-Net ResNet34, SegFormer-B0) on FOV-square 384 px crops.

One run = one model x one fold (0-4: trained on the other Center 1 patients; "final": all Center 1). BCE + soft
Dice, AdamW with a per-step cosine schedule, geometric augmentation only (flip, rotation, scale, translation,
one affine grid shared by image and mask; no colour, brightness, contrast, blur or noise change of any kind).
A checkpoint is written after every epoch, so a killed run resumes where it stopped.

  04_train.py --model unet --fold 0 --epochs 20 --pilot     fold-0 pilot, inner validation split per epoch
  04_train.py --model unet --fold 0 --epochs 20             cross-fitting model for fold 0
  04_train.py --model segformer --fold final --epochs 20    model on all Center 1
  04_train.py --all [--epochs N]                            whole queue, skipping finished models
  (once config/analysis_lock.yaml freezes epochs, a non-pilot --epochs that differs from it is an error)
  04_train.py --bench                                       precision / throughput benchmark -> work/logs/train_bench.json
  04_train.py --smoke                                       64 real frames, a few steps per model

Outputs in work/models: {model}_fold{k}.pt or {model}_final.pt (fp32 state_dict), train_log_{model}_fold{k}.csv,
and while running {model}_fold{k}_ckpt.pt. Pilot runs write the same files with a `_pilot` suffix
({model}_fold0_pilot.pt ...), so --all never mistakes a pilot model (inner-validation patients left out) for the
real fold-0 model.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # before torch is imported

import argparse
import json
import math
import random
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import fov  # noqa: E402
from common import ROOT, load_config, set_seed  # noqa: E402
from models import MODEL_NAMES, build_model  # noqa: E402

CFG = load_config()
SEG = CFG["segmenters"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
LOGS_DIR = ROOT / CFG["paths"]["logs"]
LOCK_PATH = ROOT / "v1" / "config" / "analysis_lock.yaml"
BATCH = int(SEG["batch_size"])
LR = float(SEG["optimizer"]["lr"])
SEED = int(CFG["seeds"]["train"])
FOLDS = [str(k) for k in range(int(SEG["cv_folds"]))] + ["final"]

# Not in analysis_plan.yaml: weight decay is from the task spec, augmentation ranges and the precision
# are choices made here (precision from work/logs/train_bench.json).
WEIGHT_DECAY = 1e-4
AUG = dict(hflip=0.5, vflip=0.5, rot_deg=30.0, scale=(0.85, 1.15), shift=0.08)
PRECISION = "fp32"
FAIL_IOU = float(CFG["endpoints"]["failure_iou_threshold"])
EVAL_BATCH = 32
AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------- augmentation, loss, metrics

def sample_theta(n, gen, aug=AUG):
    """n random 2 x 3 affine matrices in affine_grid convention (output coordinates -> input coordinates).

    Composition per sample: random flips, zoom by s (the grid samples a 1/s sized window), rotation, shift.
    """
    u = torch.rand(n, 6, generator=gen)
    ang = (u[:, 0] * 2 - 1) * math.radians(aug["rot_deg"])
    s = aug["scale"][0] + u[:, 1] * (aug["scale"][1] - aug["scale"][0])
    fx = torch.where(u[:, 2] < aug["hflip"], -1.0, 1.0)
    fy = torch.where(u[:, 3] < aug["vflip"], -1.0, 1.0)
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([
        torch.stack([cos * fx, -sin * fy, (u[:, 4] * 2 - 1) * aug["shift"]], dim=1),
        torch.stack([sin * fx, cos * fy, (u[:, 5] * 2 - 1) * aug["shift"]], dim=1),
    ], dim=1)
    return theta


def apply_affine(x, m, theta):
    """Warp image x (B, 3, H, W, bilinear) and mask m (B, 1, H, W, nearest) with the same grid; zeros outside."""
    grid = F.affine_grid(theta.to(x.dtype), list(x.shape), align_corners=False)
    xw = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mw = F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return xw, mw


def seg_loss(logits, target):
    """BCE + soft Dice (per-frame Dice, averaged over the batch), computed in fp32."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1.0) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1.0)
    return bce + dice.mean()


def batch_metrics(logits, target, disk):
    """Per-frame IoU and Dice of the thresholded logits inside the FOV disk. An empty prediction scores 0."""
    pred = (logits > 0) & disk
    gt = (target > 0.5) & disk
    inter = (pred & gt).sum((1, 2, 3)).float()
    union = (pred | gt).sum((1, 2, 3)).float().clamp(min=1)
    total = (pred.sum((1, 2, 3)) + gt.sum((1, 2, 3))).float().clamp(min=1)
    return inter / union, 2 * inter / total


def to_batch(images, masks, idx, device):
    """uint8 numpy rows -> contiguous float image (B, 3, H, W) in [0, 1] and mask (B, 1, H, W) on the device."""
    x = torch.from_numpy(images[idx]).permute(0, 3, 1, 2).float().div_(255).contiguous()
    y = torch.from_numpy(masks[idx]).unsqueeze(1).float()
    return x.to(device), y.to(device)


def augmented_batch(images, masks, idx, gen, device):
    """Batch warped on the host. grid_sample on MPS was seen returning dimmed, low-contrast images for real crops
    (layout and batch-size dependent) that the net then trained on while being scored on clean ones, so the warp
    runs on the CPU (13 ms per batch) and only the finished batch goes to the device."""
    x, y = to_batch(images, masks, idx, "cpu")
    x, y = apply_affine(x, y, sample_theta(len(x), gen))
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, images, masks, idx, device):
    """Mean/median IoU, mean Dice and share of frames below the failure IoU on the crops (inside the FOV disk)."""
    model.eval()
    disk = torch.from_numpy(fov.fov_disk_mask(images.shape[1], fov.ERODE_FRAC)).to(device)
    ious, dices = [], []
    for i in range(0, len(idx), EVAL_BATCH):
        x, y = to_batch(images, masks, idx[i:i + EVAL_BATCH], device)
        iou, dice = batch_metrics(model(x), y, disk)
        ious.append(iou.cpu())
        dices.append(dice.cpu())
    iou, dice = torch.cat(ious), torch.cat(dices)
    return dict(val_iou=float(iou.mean()), val_dice=float(dice.mean()), val_iou_median=float(iou.median()),
                val_fail_frac=float((iou < FAIL_IOU).float().mean()))


# ---------------------------------------------------------------- training

@dataclass
class Run:
    name: str
    tag: str                  # fold0, final, fold0_pilot
    epochs: int
    seed: int
    device: str
    precision: str
    ckpt_path: Path
    log_path: Path


def train_step(model, opt, sched, x, y, precision="fp32", scaler=None):
    """One optimiser step; returns the detached loss (no host sync)."""
    amp = precision != "fp32"
    with torch.autocast(device_type=x.device.type, dtype=AMP_DTYPE.get(precision), enabled=amp):
        logits = model(x)
    loss = seg_loss(logits, y)
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    if sched is not None:
        sched.step()
    return loss.detach()


def train_epoch(model, opt, sched, images, masks, idx, epoch, run):
    """One pass over idx in a seeded random order with a seeded augmentation stream (so a resume repeats it)."""
    model.train()
    order = np.random.default_rng([run.seed, epoch]).permutation(idx)
    gen = torch.Generator().manual_seed(run.seed * 1000 + epoch)
    losses = []
    for s in range(len(order) // BATCH):
        x, y = augmented_batch(images, masks, order[s * BATCH:(s + 1) * BATCH], gen, run.device)
        losses.append(train_step(model, opt, sched, x, y, run.precision))
    loss = float(torch.stack(losses).mean())
    if not math.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss in epoch {epoch}")
    return loss


def make_training(name, epochs, steps_per_epoch, device, seed):
    set_seed(seed)
    model = build_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    return model, opt, sched


def rng_state(device):
    state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if device == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(run, model, opt, sched, images, masks, tr_idx, va_idx, log=print, stop_after=None):
    """Train from the last checkpoint (if any) to run.epochs. stop_after ends early as if the process died (tests)."""
    log_rows, start = [], 0
    if run.ckpt_path.exists():
        ck = torch.load(run.ckpt_path, map_location="cpu", weights_only=False)
        if ck["epochs"] != run.epochs or ck["precision"] != run.precision:
            raise ValueError(f"{run.ckpt_path.name} was started with epochs={ck['epochs']}, precision={ck['precision']}; "
                             "delete it to start over")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        set_rng_state(ck["rng"])
        log_rows, start = ck["log"], ck["epoch"]
        log(f"[{run.name} {run.tag}] resuming after epoch {start}/{run.epochs}")
    for epoch in range(start, run.epochs):
        t0 = time.time()
        loss = train_epoch(model, opt, sched, images, masks, tr_idx, epoch, run)
        row = dict(epoch=epoch + 1, train_loss=loss, lr=sched.get_last_lr()[0])
        if len(va_idx):
            row.update(evaluate(model, images, masks, va_idx, run.device))
        row["seconds"] = time.time() - t0
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run.log_path, index=False)
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch + 1,
                         epochs=run.epochs, precision=run.precision, rng=rng_state(run.device), log=log_rows), run.ckpt_path)
        val = f"  val IoU {row['val_iou']:.3f} Dice {row['val_dice']:.3f}" if "val_iou" in row else ""
        log(f"[{run.name} {run.tag}] epoch {epoch + 1}/{run.epochs}  loss {loss:.4f}{val}  {row['seconds']:.0f} s")
        if stop_after is not None and epoch + 1 >= stop_after:
            return log_rows
    return log_rows


def run_paths(name, fold, pilot):
    tag = ("final" if fold == "final" else f"fold{fold}") + ("_pilot" if pilot else "")
    return tag, MODELS_DIR / f"{name}_{tag}.pt", MODELS_DIR / f"{name}_{tag}_ckpt.pt", MODELS_DIR / f"train_log_{name}_{tag}.csv"


def train_run(name, fold, epochs, pilot, table, images, masks, device, precision=PRECISION, log=print):
    """Train one model x fold and write its weights. Skips when the weights file already exists."""
    tag, final_path, ckpt_path, log_path = run_paths(name, fold, pilot)
    if final_path.exists():
        log(f"[{name} {tag}] {final_path.name} exists, skipping")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pats = dataset.patients_for(fold, pilot)
    tr_idx = np.flatnonzero(table.patient_id.isin(pats["train"]).to_numpy())
    va_idx = np.flatnonzero(table.patient_id.isin(pats["val"]).to_numpy())
    log(f"[{name} {tag}] {len(pats['train'])} train patients / {len(tr_idx)} frames, {len(pats['val'])} inner-val patients / "
        f"{len(va_idx)} frames, {len(pats['heldout'])} held-out patients (not loaded into training)")
    seed = SEED + (len(FOLDS) - 1 if fold == "final" else int(fold))
    run = Run(name, tag, epochs, seed, device, precision, ckpt_path, log_path)
    model, opt, sched = make_training(name, epochs, len(tr_idx) // BATCH, device, seed)
    fit(run, model, opt, sched, images, masks, tr_idx, va_idx, log)
    save_atomic({k: v.detach().cpu() for k, v in model.state_dict().items()}, final_path)
    ckpt_path.unlink(missing_ok=True)
    log(f"[{name} {tag}] done, wrote {final_path.name}")


def make_logger(path=None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        if path is not None:
            with open(path, "a") as fh:
                fh.write(line + "\n")
    return log


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lock_epochs():
    """Epochs frozen in config/analysis_lock.yaml as {model: int}; models without an entry are left out."""
    if not LOCK_PATH.exists():
        return {}
    e = (yaml.safe_load(open(LOCK_PATH)) or {}).get("epochs")
    return {m: e[m] for m in MODEL_NAMES if isinstance(e, dict) and isinstance(e.get(m), int)}


def check_epochs_against_lock(arg_epochs, names):
    """Exit when --epochs differs from the frozen epochs of any of `names` (no lock entry means no check)."""
    locked = lock_epochs()
    bad = {m: locked[m] for m in names if m in locked and locked[m] != arg_epochs}
    if bad:
        sys.exit(f"--epochs {arg_epochs} contradicts the frozen epochs in {LOCK_PATH.name} {bad}; drop --epochs to use "
                 "the frozen value (pilot runs, --bench and --smoke are exempt)")


def queue_epochs(arg_epochs):
    if arg_epochs:
        check_epochs_against_lock(arg_epochs, MODEL_NAMES)
        return {m: arg_epochs for m in MODEL_NAMES}
    locked = lock_epochs()
    if len(locked) == len(MODEL_NAMES):
        return locked
    sys.exit("--all needs --epochs N or an `epochs: {unet: N, segformer: N}` entry in config/analysis_lock.yaml")


def run_all(args):
    log = make_logger(LOGS_DIR / "train_all.log")
    epochs = queue_epochs(args.epochs)
    queue = [(m, str(f)) for m in MODEL_NAMES for f in range(int(SEG["cv_folds"]))] + [(m, "final") for m in MODEL_NAMES]
    todo = [(m, f) for m, f in queue if not run_paths(m, f, False)[1].exists()]
    log(f"queue: {len(todo)}/{len(queue)} runs to do, epochs {epochs}, precision {args.precision}")
    if not todo:
        return
    device = pick_device()
    table, images, masks = dataset.load_center1(args.workers, log)
    failed = []
    for m, f in todo:
        try:
            train_run(m, f, epochs[m], False, table, images, masks, device, args.precision, log)
        except Exception:
            log(f"[{m} {f}] FAILED\n{traceback.format_exc()}")
            failed.append((m, f))
    log("queue finished" + (f"; failed (rerun --all to resume them): {failed}" if failed else ", no failures"))
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------- benchmark and smoke test

def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def sample_frames(table, n, seed, workers):
    rows = np.random.default_rng(seed).choice(len(table), size=min(n, len(table)), replace=False)
    sub = table.iloc[np.sort(rows)].reset_index(drop=True)
    images, masks = dataset.build_crops(sub, workers=workers, log=lambda m: None)
    return sub, images, masks


def bench_data_ms(images, masks, device, n=60):
    """Host-side cost of one batch: gather uint8 rows, copy to the device, build the float tensors, augment."""
    gen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    sync(device)
    t = time.time()
    for _ in range(n):
        augmented_batch(images, masks, rng.choice(len(images), BATCH, replace=False), gen, device)
    sync(device)
    return 1000 * (time.time() - t) / n


def bench_one(name, precision, images, masks, device, steps, warmup, seed):
    """Constant-lr training on one batch and augmentation stream: steady-state throughput and the loss curve."""
    set_seed(seed)
    model = build_model(name).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device) if precision == "fp16" else None
    order = np.random.default_rng(seed).permutation(np.tile(np.arange(len(images)), steps * BATCH // len(images) + 1))
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for s in range(steps):
        if s == warmup:
            sync(device)
            t_start = time.time()
        x, y = augmented_batch(images, masks, order[s * BATCH:(s + 1) * BATCH], gen, device)
        losses.append(train_step(model, opt, None, x, y, precision, scaler))
    sync(device)
    wall = time.time() - t_start
    curve = torch.stack(losses).float().cpu().numpy()
    tail = curve[-50:]
    return dict(model=name, precision=precision, steps=steps, img_per_s=(steps - warmup) * BATCH / wall,
                finite=bool(np.isfinite(curve).all()), loss_first10=float(np.nanmean(curve[:10])), loss_last50=float(np.nanmean(tail)),
                loss_last50_sd=float(np.nanstd(tail)), loss_curve_every10=[round(float(v), 4) for v in curve[::10]])


def bench(table, args):
    device = pick_device()
    log = make_logger()
    sub, images, masks = sample_frames(table, args.bench_frames, SEED, args.workers)
    load = os.getloadavg()[0]
    data_ms = bench_data_ms(images, masks, device)
    log(f"benchmark on {len(sub)} real crops, device {device}, 1-min load average {load:.1f} (throughput is pessimistic if "
        f"other jobs are running), host data cost {data_ms:.1f} ms/batch")
    rows = []
    for name in MODEL_NAMES:
        for precision in ("fp32", "bf16", "fp16"):
            r = bench_one(name, precision, images, masks, device, args.bench_steps, 20, SEED)
            rows.append(r)
            log(f"  {name:9s} {precision}: {r['img_per_s']:.1f} img/s, loss {r['loss_first10']:.3f} -> {r['loss_last50']:.3f} "
                f"(sd {r['loss_last50_sd']:.3f}), finite {r['finite']}")

    def row(name, precision):
        return next(r for r in rows if r["model"] == name and r["precision"] == precision)

    # An autocast mode replaces fp32 only if, for BOTH models, it stays finite, ends within 5% of the fp32 loss and is
    # at least 10% faster; among such modes the faster one (mean over models) wins.
    ok = {p: all(row(m, p)["finite"] and row(m, p)["loss_last50"] <= row(m, "fp32")["loss_last50"] * 1.05
                 and row(m, p)["img_per_s"] >= row(m, "fp32")["img_per_s"] * 1.10 for m in MODEL_NAMES) for p in ("bf16", "fp16")}
    speed = {p: np.mean([row(m, p)["img_per_s"] / row(m, "fp32")["img_per_s"] for m in MODEL_NAMES]) for p in ok}
    precision = max((p for p in ok if ok[p]), key=speed.get, default="fp32")

    cv = pd.read_csv(dataset.DERIVED / "folds.csv").query("role == 'cv'")
    n_final = float(cv.n_frames.sum())
    n_fold = float((n_final - cv.groupby("fold").n_frames.sum()).mean())
    epochs_grid = (10, 15, 20, 25, 30)
    est = {}
    for name in MODEL_NAMES:
        ips = row(name, precision)["img_per_s"]
        m_fold, m_final = n_fold / ips / 60, n_final / ips / 60
        est[name] = dict(img_per_s=ips, min_per_epoch_cv_fold=m_fold, min_per_epoch_final=m_final,
                         queue_hours_at_epochs={str(e): (5 * m_fold + m_final) * e / 60 for e in epochs_grid})
    est["both_models_queue_hours_at_epochs"] = {str(e): sum(est[m]["queue_hours_at_epochs"][str(e)] for m in MODEL_NAMES)
                                                for e in epochs_grid}
    out = dict(device=device, batch_size=BATCH, benchmark_frames=len(sub), steps_per_config=args.bench_steps, warmup_steps=20,
               load_average_1min_at_start=load, note="throughput measured while other CPU jobs may have been running",
               torch=torch.__version__, host_data_ms_per_batch=data_ms, runs=rows, autocast_qualifies=ok, speedup_vs_fp32=speed,
               chosen_precision=precision, estimates=est, train_frames_per_cv_fold_mean=n_fold, train_frames_final=n_final,
               ram_gb_center1_crops=round(len(table) * 384 * 384 * 4 / 1e9, 2))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "train_bench.json", "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"chosen precision {precision}; wrote {LOGS_DIR / 'train_bench.json'}")


def smoke(table, args):
    """Train both models for a few steps on 64 real frames: shapes, finite and falling loss, validation metrics."""
    device = pick_device()
    log = make_logger()
    sub, images, masks = sample_frames(table, 64, SEED, args.workers)
    log(f"smoke test: {len(sub)} real crops {images.shape[1:]} uint8, mask values {sorted(np.unique(masks).tolist())}, "
        f"foreground fraction mean {masks.mean():.3f}")
    tr, va = np.arange(48), np.arange(48, 64)
    for name in MODEL_NAMES:
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(name, "smoke", 10, SEED, device, PRECISION, Path(tmp) / "ck.pt", Path(tmp) / "log.csv")
            model, opt, sched = make_training(name, run.epochs, len(tr) // BATCH, device, SEED)
            with torch.no_grad():
                out_shape = tuple(model.eval()(to_batch(images, masks, va[:2], device)[0]).shape)
            rows = fit(run, model, opt, sched, images, masks, tr, va, log)
        log(f"{name}: logits {out_shape}, loss {rows[0]['train_loss']:.3f} -> {rows[-1]['train_loss']:.3f}, "
            f"val IoU {rows[0]['val_iou']:.3f} -> {rows[-1]['val_iou']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES)
    p.add_argument("--fold", choices=FOLDS)
    p.add_argument("--epochs", type=int)
    p.add_argument("--pilot", action="store_true", help="hold out an inner validation split from the training patients and log its IoU per epoch")
    p.add_argument("--all", action="store_true", help="run the whole queue, skipping models whose weights exist")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--precision", choices=["fp32", "bf16"], default=PRECISION)
    p.add_argument("--workers", type=int, default=dataset.WORKERS)
    p.add_argument("--out-dir", type=Path, help="write weights, checkpoints and logs here instead of work/models (trial runs)")
    p.add_argument("--bench-frames", type=int, default=640)
    p.add_argument("--bench-steps", type=int, default=200)
    args = p.parse_args()
    if args.out_dir:
        global MODELS_DIR
        MODELS_DIR = args.out_dir
    if args.all:
        return run_all(args)
    if args.bench or args.smoke:
        table = dataset.frame_table("center1_labeled")
        dataset.log_exclusions(table, make_logger())
        return bench(table, args) if args.bench else smoke(table, args)
    if not (args.model and args.fold and args.epochs):
        p.error("give --model, --fold and --epochs (or --all, --bench, --smoke)")
    if args.pilot and args.fold == "final":
        p.error("--pilot needs a held-out fold (0-4)")
    if not args.pilot:
        check_epochs_against_lock(args.epochs, [args.model])
    log = make_logger()
    table, images, masks = dataset.load_center1(args.workers, log)
    train_run(args.model, args.fold, args.epochs, args.pilot, table, images, masks, pick_device(), args.precision, log)


if __name__ == "__main__":
    main()
