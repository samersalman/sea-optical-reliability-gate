"""Inference for the trained segmenters: probability maps, per-frame metrics, optics on the predicted background.

Shared by 05_predict.py (labeled frames), 07_stream.py (unlabeled stream) and 06_perturb.py (perturbed canvases).

Pipeline per frame: image -> FOV-square crop -> 384 x 384 (area interpolation, identical to the training crops
in dataset.py) -> sigmoid probability map. Worker processes (cv2 single-threaded) decode and crop; the main process
runs the forward pass and the metrics. Only the forward pass touches the GPU: every resize, crop, threshold and
metric stays on the CPU, because MPS resampling ops gave wrong pixels once (see 04_train.augmented_batch).

Metrics (CONTRACTS.md predictions.parquet), all inside the eroded FOV disk (fov.ERODE_FRAC):
- iou, dice, pred_area_frac, gt_area_frac at NATIVE resolution: the ground truth is the FOV-square crop at
  crop_side px, the 384 probability map is upsampled bilinearly to crop_side and cut at 0.5. Prediction empty
  and ground truth not -> IoU 0 and Dice 0. Both empty -> IoU 1 and Dice 1 (not expected for labeled frames; counted in the run summary).
- entropy_fov: mean binary entropy (bits) of the 384 map over the 384 disk. entropy_roi: the same over the
  predicted mask (0.5 cut on the 384 grid) dilated by ROI_DILATE_PX with an elliptical kernel; an empty
  prediction falls back to the entropy_fov value and is counted (the count goes to the run summary).
- ens_dis = 1 - Dice(U-Net mask, SegFormer mask) on the 384 grid inside the disk; both empty -> 0.
- unet_bg_*: the 11 optical metrics on the 512 canvas over the disk minus the U-Net predicted mask (the native
  mask resampled to 512 with nearest neighbour). The ground truth never enters. Fewer than optics_lib.MIN_PIXELS
  background pixels give NaN for all 11; those frames are counted.

Model selection: Center 1 labeled frames use the out-of-fold model of their patient's fold, Center 2 and the
stream use the *_final.pt models. check_selection() asserts, from folds.csv, that no frame is scored by a model
whose training set contains its patient. Pilot weights (fold0_pilot) are for smoke tests only.

Long runs are chunked: each chunk (one model tag, up to CHUNK frames) is written to its own small parquet part in
parts_dir by an atomic rename, with a plan fingerprint (frames, tags, SHA-256 of the weights), so a killed run continues
with the missing chunks; finished chunks of a different frame set, retrained weights, or code / FOV geometry
that changed since (SHA-256 of infer.py, optics_lib.py, fov.py, fov.parquet) are never reused.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # before torch is imported

import hashlib
import itertools
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import fov
import optics_lib
import perturb_ops
from common import ROOT, abspath, load_config, read_image, read_mask
from models import MODEL_NAMES, build_model

CFG = load_config()
DERIVED = ROOT / CFG["paths"]["derived"]
MODELS_DIR = ROOT / CFG["paths"]["models"]
LOGS_DIR = ROOT / CFG["paths"]["logs"]
SIZE = int(CFG["fov"]["segmenter_input_px"])
CANON = int(CFG["fov"]["canonical_diameter_px"])
METRICS = tuple(CFG["optical_metrics"])
SEED_PERTURB = int(CFG["seeds"]["perturb"])

THRESHOLD = float(CFG["segmenters"]["mask_threshold"])
ROI_DILATE_PX = int(CFG["entropy"]["roi_dilate_px"])
HERE = Path(__file__).resolve().parent
PARTS_ROOT = ROOT / "v1" / "work" / "parts"     # resumable chunks of the real runs
# Files whose content decides the numbers of a run: a change makes finished chunks stale (see code_fingerprint).
CODE_FILES = (HERE / "infer.py", HERE / "optics_lib.py", HERE / "fov.py", DERIVED / "fov.parquet")
CHUNK = 500              # frames per resumable chunk (predictions, stream)
BATCH = 32               # images per forward pass
POST_THREADS = 4         # threads for the per-frame metrics in the main process
PERTURB_CHUNK = 10       # frames per resumable chunk of the perturbation run
BG_COLS = [f"unet_bg_{m}" for m in METRICS]
PRED_COLS = (["frame_id", "patient_id", "model", "source", "partition", "fold", "iou", "dice", "pred_area_frac",
              "gt_area_frac", "entropy_fov", "entropy_roi", "ens_dis"] + BG_COLS)
STREAM_COLS = [c for c in PRED_COLS if c not in ("iou", "dice", "gt_area_frac")] + ["video_patient", "t_index"]
PERTURB_COLS = (["frame_id", "patient_id", "domain", "scenario", "instrument", "model", "family", "severity", "param",
                 "iou", "dice", "pred_area_frac", "entropy_fov"] + list(METRICS))


# ---------------------------------------------------------------- model selection

_TAG = re.compile(r"^(?:fold(?P<k>[0-4])(?P<pilot>_pilot)?|final)$")


def trained_on(tag, folds):
    """Patient ids whose frames a model with this tag saw in training (inner validation included)."""
    m = _TAG.match(tag)
    if m is None:
        raise ValueError(f"unknown model tag {tag!r}")
    cv = folds[folds.role == "cv"]
    if tag == "final":
        return set(cv.patient_id)
    return set(cv[cv.fold != int(m["k"])].patient_id)


def assign_models(table, pilot=False):
    """Add `tag` (weights suffix) and `source` (oof | final) columns.

    Center 1 labeled frames: the model of their patient's fold. Center 2 and stream: the final model. In pilot
    mode every frame takes fold0_pilot (smoke tests; the caller must drop the frames it trained on).
    """
    t = table.copy()
    c1 = t.partition == "center1_labeled"
    t["source"] = np.where(c1, "oof", "final")
    if pilot:
        t["tag"] = "fold0_pilot"
    else:
        t["tag"] = np.where(c1, "fold" + t.fold.fillna(-1).astype(int).astype(str), "final")
    return t


def check_selection(table, folds):
    """Raise if any frame would be scored by a model that trained on its patient (needs `tag`, `patient_id`)."""
    bad = []
    for tag, g in table.groupby("tag"):
        seen = trained_on(tag, folds) & set(g.patient_id)
        if seen:
            bad.append(f"{tag}: {sorted(seen)[:5]}{' ...' if len(seen) > 5 else ''}")
    if bad:
        raise AssertionError("frames scored by a model trained on their patient: " + "; ".join(bad))
    fold_of = folds.set_index("patient_id").fold
    oof = table[table.source == "oof"]
    if (oof.fold.astype(int).to_numpy() != fold_of.loc[oof.patient_id].to_numpy()).any():
        raise AssertionError("table fold column disagrees with folds.csv")
    real = table[~table.tag.str.endswith("_pilot")]
    want = np.where(real.source == "oof", "fold" + real.fold.astype(int).astype(str), "final")
    if (real.tag.to_numpy() != want).any():
        raise AssertionError("oof frames must use their fold's model and Center 2 / stream frames the final model")


def weights_path(name, tag, models_dir=MODELS_DIR):
    return Path(models_dir) / f"{name}_{tag}.pt"


def require_weights(tags, models_dir=MODELS_DIR):
    """Fail before any work starts if a weights file a run needs is missing."""
    missing = [weights_path(n, t, models_dir).name for t in sorted(set(tags)) for n in MODEL_NAMES
               if not weights_path(n, t, models_dir).exists()]
    if missing:
        raise FileNotFoundError(f"missing weights in {models_dir}: {', '.join(missing)}")


def weights_fingerprint(tags, models_dir=MODELS_DIR):
    """SHA-256 prefix of every weights file the tags need, so finished chunks of retrained models are never reused."""
    out = {}
    for t in sorted(set(tags)):
        for n in MODEL_NAMES:
            path = weights_path(n, t, models_dir)
            out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def code_fingerprint(extra=()):
    """SHA-256 prefix of the code and geometry a run depends on (CODE_FILES plus `extra`), keyed by file name."""
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] for p in (*CODE_FILES, *extra)}


def load_models(tag, device, models_dir=MODELS_DIR):
    """{name: eval-mode model on device} for one tag (fp32 state_dicts written by 04_train.py)."""
    out = {}
    for name in MODEL_NAMES:
        path = weights_path(name, tag, models_dir)
        if not path.exists():
            raise FileNotFoundError(f"missing weights {path}")
        net = build_model(name, pretrained=False)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        out[name] = net.eval().to(device)
    return out


@torch.inference_mode()
def forward_probs(model, images):
    """Sigmoid probability maps (N, S, S) float32 from images (N, S, S, 3), uint8 or float in [0, 1].

    Same batch semantics as 04_train.to_batch: contiguous NCHW float32, /255 for uint8; the sigmoid runs on the CPU.
    """
    x = torch.from_numpy(np.ascontiguousarray(images))
    x = x.permute(0, 3, 1, 2).float()
    if images.dtype == np.uint8:
        x = x.div_(255)
    logits = model(x.contiguous().to(next(model.parameters()).device))
    return torch.sigmoid(logits.float().cpu())[:, 0].numpy()


# ---------------------------------------------------------------- metrics (numpy, CPU)

@lru_cache(maxsize=24)
def disk(size):
    d = fov.fov_disk_mask(size, fov.ERODE_FRAC)
    d.flags.writeable = False
    return d


def binary_entropy_bits(p):
    p = np.asarray(p, dtype=np.float32)
    a, b = np.clip(p, 1e-12, 1.0), np.clip(1.0 - p, 1e-12, 1.0)
    return -(a * np.log2(a) + b * np.log2(b))


def overlap(pred, gt, region):
    """(IoU, Dice) of two bool masks inside region. Both empty -> (1, 1); one empty -> (0, 0)."""
    p, g = pred & region, gt & region
    inter = int((p & g).sum())
    sp, sg = int(p.sum()), int(g.sum())
    if sp + sg == 0:
        return 1.0, 1.0
    return inter / (sp + sg - inter), 2 * inter / (sp + sg)


def ens_dis(unet_mask, segformer_mask, region):
    """1 - Dice between the two models' masks inside region (both empty -> 0)."""
    return 1.0 - overlap(unet_mask, segformer_mask, region)[1]


def roi_entropy(ent, pred, region, px=ROI_DILATE_PX):
    """Mean of ent over the prediction (inside region) dilated by px; None when the prediction is empty."""
    seed = pred & region
    if not seed.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    roi = cv2.dilate(seed.astype(np.uint8), kernel).astype(bool) & region
    return float(ent[roi].mean())


def seg_stats(prob, side, gt=None):
    """Metrics of one probability map.

    prob: (S, S) float32 map on the 384 grid; side: native crop_side; gt: bool (side, side) or None (stream).
    Returns (row, pred384, pred_native, roi_fallback): row has pred_area_frac, entropy_fov, entropy_roi and, with
    gt, iou, dice, gt_area_frac; the masks are bool at the 384 grid and the native grid.
    """
    d384, dn = disk(prob.shape[0]), disk(side)
    pred384 = prob > THRESHOLD
    pred_native = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR) > THRESHOLD
    ent = binary_entropy_bits(prob)
    h_fov = float(ent[d384].mean())
    h_roi = roi_entropy(ent, pred384, d384)
    row = dict(pred_area_frac=float((pred_native & dn).sum() / dn.sum()), entropy_fov=h_fov,
               entropy_roi=h_fov if h_roi is None else h_roi)
    if gt is not None:
        row["iou"], row["dice"] = overlap(pred_native, gt, dn)
        row["gt_area_frac"] = float((gt & dn).sum() / dn.sum())
    return row, pred384, pred_native, h_roi is None


def grid_stats(prob, gt):
    """iou, dice, pred_area_frac, entropy_fov on the grid of prob itself (perturbation run, 384 vs the 384 GT crop)."""
    d = disk(prob.shape[0])
    pred = prob > THRESHOLD
    iou, dice = overlap(pred, gt, d)
    return dict(iou=iou, dice=dice, pred_area_frac=float((pred & d).sum() / d.sum()),
                entropy_fov=float(binary_entropy_bits(prob)[d].mean()))


def bg_metrics(canvas, pred_native):
    """The 11 optical metrics on the 512 canvas over the disk minus the predicted mask, as unet_bg_<name>."""
    bg = ~cv2.resize(pred_native.astype(np.uint8), (CANON, CANON), interpolation=cv2.INTER_NEAREST).astype(bool)
    m = optics_lib.compute_metrics(canvas, disk(CANON), bg)
    return {f"unet_bg_{k}": v for k, v in m.items()}


# ---------------------------------------------------------------- loading workers

def _init_worker():
    cv2.setNumThreads(1)


def load_job(job):
    """One frame -> (crop384 uint8, canvas512 float32, native GT bool or None, seconds spent).

    job = (image_path, mask_path or None, cx, cy, r, crop_side). The crop is built exactly as dataset._build_one
    does for training; the canvas is the optics canvas of 02_optics.py.
    """
    t0 = time.perf_counter()
    image_path, mask_path, cx, cy, r, side = job
    img = read_image(abspath(image_path))
    crop = fov.crop_square(img, cx, cy, r, SIZE, "auto")
    canvas, _ = fov.canonical_canvas(img, dict(fov_cx=cx, fov_cy=cy, fov_r=r))
    gt = None
    if mask_path is not None:
        gt = fov.crop_square(read_mask(abspath(mask_path)), cx, cy, r, side, "nearest")
    return crop, canvas, gt, time.perf_counter() - t0


def family_rng(frame_id, family):
    """Generator for one (frame, family): rebuilt for every severity so glare blobs and noise fields nest."""
    h = int.from_bytes(hashlib.sha256(frame_id.encode()).digest()[:8], "big")
    return np.random.default_rng([SEED_PERTURB, h, perturb_ops.FAMILIES.index(family)])


# condition 0 is the unperturbed canvas; then every (family, severity 1-5) in perturb_ops order
COND = [(None, 0)] + [(f, s) for f in perturb_ops.FAMILIES for s in perturb_ops.SEVERITIES if s > 0]


def perturb_job(job):
    """One frame -> (images (len(COND), 384, 384, 3) float32, 11 optical metrics per condition, GT384 bool, seconds).

    job = (frame_id, image_path, mask_path, cx, cy, r). Operators get the FULL uneroded FOV disk (so no rim ring
    survives); metrics use the eroded disk. Everything goes native -> 512 canvas -> perturb -> metrics -> 384 (area).
    """
    t0 = time.perf_counter()
    frame_id, image_path, mask_path, cx, cy, r = job
    img = read_image(abspath(image_path))
    canvas, eroded = fov.canonical_canvas(img, dict(fov_cx=cx, fov_cy=cy, fov_r=r))
    full = fov.fov_disk_mask(CANON, 0.0)
    gt384 = fov.crop_square(read_mask(abspath(mask_path)), cx, cy, r, SIZE, "nearest")
    images = np.empty((len(COND), SIZE, SIZE, 3), np.float32)
    metrics = []
    for k, (fam, sev) in enumerate(COND):
        pert = canvas if fam is None else perturb_ops.apply(canvas, full, fam, sev, family_rng(frame_id, fam))
        metrics.append(optics_lib.compute_metrics(pert, eroded))
        images[k] = cv2.resize(pert, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return images, metrics, gt384, time.perf_counter() - t0


def bounded_imap(pool, fn, jobs, window):
    """Ordered pool.imap that keeps at most `window` results in flight (Pool.imap would queue them all in RAM).

    pool=None runs in-process, which is what the unit tests use.
    """
    if pool is None:
        yield from map(fn, jobs)
        return
    jobs = iter(jobs)
    pending = deque(pool.apply_async(fn, (j,)) for j in itertools.islice(jobs, window))
    while pending:
        res = pending.popleft().get()
        for j in itertools.islice(jobs, 1):
            pending.append(pool.apply_async(fn, (j,)))
        yield res


class Engine:
    """Worker pool plus the model pair of the tag currently in use."""

    def __init__(self, device, workers, models_dir=MODELS_DIR, batch=BATCH, threads=POST_THREADS):
        self.device, self.workers, self.models_dir, self.batch, self.threads = device, workers, models_dir, batch, threads
        self.pool, self._tag, self._models, self._threads = None, None, None, None

    def __enter__(self):
        if self.workers > 0:
            self.pool = get_context("spawn").Pool(self.workers, initializer=_init_worker)
        if self.threads > 1:
            self._threads = ThreadPoolExecutor(self.threads)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        if self._threads is not None:
            self._threads.shutdown()

    def map(self, fn, items):
        """In-order map over items, in threads when configured (numpy / cv2 release the GIL)."""
        return self._threads.map(fn, items) if self._threads is not None else map(fn, items)

    def models(self, tag):
        if tag != self._tag:
            self._models = None            # free the old pair before the new one is built
            self._models, self._tag = load_models(tag, self.device, self.models_dir), tag
        return self._models

    def imap(self, fn, jobs, window=None):
        """In-order results with at most `window` in flight. The default (8 per worker) matters on a cold disk: file
        latency has a heavy tail and a short window lets one slow file idle the other workers (1.6x slower at 2 per worker)."""
        return bounded_imap(self.pool, fn, jobs, window or max(8 * self.workers, 2))

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()


def pick_device(name="auto"):
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- resumable chunks

def _write_atomic(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def fingerprint(*parts):
    return hashlib.sha256("\n".join(map(str, parts)).encode()).hexdigest()


def run_chunks(chunks, work, parts_dir, plan, restart=False, max_new=None, log=print):
    """Run work(payload) -> (DataFrame, stats dict) for every (chunk_id, payload) without a finished part.

    A part is parts_dir/<chunk_id>.parquet plus <chunk_id>.json (stats), both renamed into place only when
    complete. plan (JSON-able) is stored on first use; a different plan in a non-empty parts_dir is an error
    unless restart=True (which clears the parts). max_new stops after that many new chunks (used to simulate a
    killed run in the tests). Returns the number of chunks computed.
    """
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    plan_path = parts_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        if not restart:
            raise RuntimeError(f"{parts_dir} holds parts of a different run; pass --restart to discard them")
        for p in parts_dir.glob("*"):
            p.unlink()
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    new = 0
    for i, (cid, payload) in enumerate(chunks):
        if (parts_dir / f"{cid}.parquet").exists():
            continue
        if max_new is not None and new >= max_new:
            break
        t0 = time.perf_counter()
        df, stats = work(payload)
        stats["wall_s"] = time.perf_counter() - t0
        (parts_dir / f"{cid}.json.tmp").write_text(json.dumps(stats))
        os.replace(parts_dir / f"{cid}.json.tmp", parts_dir / f"{cid}.json")
        _write_atomic(df, parts_dir / f"{cid}.parquet")
        new += 1
        log(f"chunk {cid} ({i + 1}/{len(chunks)}): {stats.get('n_frames', len(df))} frames in {stats['wall_s']:.1f} s")
    return new


def read_parts(parts_dir, chunk_ids):
    """(concatenated parts in chunk order, summed stats). Raises if a chunk is missing."""
    parts_dir = Path(parts_dir)
    missing = [c for c in chunk_ids if not (parts_dir / f"{c}.parquet").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} chunks not finished (first: {missing[0]}); rerun to resume")
    df = pd.concat([pd.read_parquet(parts_dir / f"{c}.parquet") for c in chunk_ids], ignore_index=True)
    tot = {}
    for c in chunk_ids:
        for k, v in json.loads((parts_dir / f"{c}.json").read_text()).items():
            tot[k] = tot.get(k, 0) + v
    return df, tot


def chunk_table(table, size):
    """[(chunk_id, frames)] : frames grouped by model tag, ordered by frame_id, `size` per chunk."""
    out = []
    for tag, g in table.sort_values(["tag", "frame_id"]).groupby("tag", sort=True):
        for i in range(0, len(g), size):
            out.append((f"{tag}_{i // size:04d}", g.iloc[i:i + size]))
    return out


# ---------------------------------------------------------------- predictions and stream

def _post_one(prob_u, prob_s, canvas, gt, side, ident):
    """Rows and counters of one frame; pure numpy/cv2 so frames can be processed in threads."""
    t = time.perf_counter()
    res = {"unet": seg_stats(prob_u, side, gt), "segformer": seg_stats(prob_s, side, gt)}
    ens = ens_dis(res["unet"][1], res["segformer"][1], disk(SIZE))
    t_post = time.perf_counter() - t
    t = time.perf_counter()
    bg = bg_metrics(canvas, res["unet"][2])
    cnt = dict(post_s=t_post, optics_s=time.perf_counter() - t, bg_nan=int(np.isnan(bg["unet_bg_lum_mean"])),
               gt_empty=int(gt is not None and not (gt & disk(side)).any()),
               roi_fallback_unet=int(res["unet"][3]), roi_fallback_segformer=int(res["segformer"][3]))
    return [{**ident, "model": n, **res[n][0], "ens_dis": ens, **bg} for n in MODEL_NAMES], cnt


def _predict_chunk(eng, frames, labeled):
    models = eng.models(frames.tag.iloc[0])
    jobs = [(r.image_path, r.mask_path if labeled else None, r.fov_cx, r.fov_cy, r.fov_r, int(r.crop_side))
            for r in frames.itertuples()]
    ident = frames[["frame_id", "patient_id", "source", "partition", "fold"]].to_dict("records")
    st = dict(n_frames=len(frames), load_cpu_s=0.0, forward_s=0.0, post_s=0.0, optics_s=0.0,
              roi_fallback_unet=0, roi_fallback_segformer=0, bg_nan=0, gt_empty=0)
    rows = []

    def flush(batch, start):
        t = time.perf_counter()
        crops = np.stack([b[0] for b in batch])
        probs = {n: forward_probs(models[n], crops) for n in MODEL_NAMES}
        st["forward_s"] += time.perf_counter() - t
        todo = [(probs["unet"][j], probs["segformer"][j], b[1], b[2], jobs[start + j][5], ident[start + j])
                for j, b in enumerate(batch)]
        for frame_rows, cnt in eng.map(lambda a: _post_one(*a), todo):
            rows.extend(frame_rows)
            for k, v in cnt.items():
                st[k] += v

    batch, start = [], 0
    for res in eng.imap(load_job, jobs):
        st["load_cpu_s"] += res[3]
        batch.append(res)
        if len(batch) == eng.batch:
            flush(batch, start)
            start, batch = start + len(batch), []
    if batch:
        flush(batch, start)
    eng.sync()
    return pd.DataFrame(rows), st


def predict_table(table, labeled, eng, parts_dir, chunk=CHUNK, restart=False, max_new=None, log=print):
    """Both segmenters on every frame of `table` (frame_table columns plus tag/source from assign_models).

    Returns (DataFrame with PRED_COLS, or STREAM_COLS without video_patient / t_index when labeled is false; summed
    stats; chunks computed in this call). The frame is None while chunks are still missing (max_new reached).
    """
    folds = pd.read_csv(DERIVED / "folds.csv")
    check_selection(table, folds)
    require_weights(table.tag, eng.models_dir)
    chunks = chunk_table(table, chunk)
    plan = dict(kind="predict", labeled=labeled, chunk=chunk, weights=weights_fingerprint(table.tag, eng.models_dir),
                code=code_fingerprint(),
                frames=fingerprint(*(f"{r.frame_id}:{r.tag}" for r in table.sort_values("frame_id").itertuples())))
    new = run_chunks(chunks, lambda f: _predict_chunk(eng, f, labeled), parts_dir, plan, restart, max_new, log)
    ids = [c for c, _ in chunks]
    if any(not (Path(parts_dir) / f"{c}.parquet").exists() for c in ids):
        return None, None, new
    df, stats = read_parts(parts_dir, ids)
    order = {f: i for i, f in enumerate(table.frame_id)}
    rank = {n: i for i, n in enumerate(MODEL_NAMES)}
    df = df.assign(_o=df.frame_id.map(order), _m=df.model.map(rank)).sort_values(["_o", "_m"]).drop(columns=["_o", "_m"])
    df = df.reset_index(drop=True)
    assert len(df) == 2 * len(table) and not df.duplicated(["frame_id", "model"]).any(), "rows do not cover frames x models"
    df["fold"] = df.fold.astype("int32")
    return df[PRED_COLS if labeled else STREAM_COLS[:-2]], stats, new


# ---------------------------------------------------------------- perturbation run

_PARAM = {(c["family"], c["severity"]): c["param"] for c in perturb_ops.CONDITIONS}


def _perturb_chunk(eng, frames):
    models = eng.models(frames.tag.iloc[0])
    jobs = [(r.frame_id, r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r) for r in frames.itertuples()]
    meta = frames[["frame_id", "patient_id", "domain", "scenario", "instrument"]].to_dict("records")
    st = dict(n_frames=len(frames), load_cpu_s=0.0, forward_s=0.0, post_s=0.0)
    rows = []
    for m, (images, metrics, gt, secs) in zip(meta, eng.imap(perturb_job, jobs, window=eng.workers + 2)):
        st["load_cpu_s"] += secs
        stats = {}
        for n in MODEL_NAMES:
            t = time.perf_counter()
            probs = np.concatenate([forward_probs(models[n], images[i:i + eng.batch]) for i in range(0, len(images), eng.batch)])
            st["forward_s"] += time.perf_counter() - t
            t = time.perf_counter()
            stats[n] = list(eng.map(lambda p: grid_stats(p, gt), probs))
            st["post_s"] += time.perf_counter() - t
        for fam in perturb_ops.FAMILIES:
            for sev in perturb_ops.SEVERITIES:
                k = 0 if sev == 0 else COND.index((fam, sev))
                for n in MODEL_NAMES:
                    rows.append({**m, "model": n, "family": fam, "severity": sev, "param": _PARAM[(fam, sev)],
                                 **stats[n][k], **metrics[k]})
    eng.sync()
    return pd.DataFrame(rows), st


def perturb_table(frames, eng, parts_dir, chunk=PERTURB_CHUNK, restart=False, max_new=None, log=print):
    """Perturbation grid for the selected frames (columns: frame_id, patient_id, domain, scenario, instrument, image_path,
    mask_path, fov_*, tag, source, fold). Returns (DataFrame with PERTURB_COLS or None, stats, chunks computed)."""
    folds = pd.read_csv(DERIVED / "folds.csv")
    check_selection(frames, folds)
    require_weights(frames.tag, eng.models_dir)
    chunks = chunk_table(frames, chunk)
    plan = dict(kind="perturb", chunk=chunk, weights=weights_fingerprint(frames.tag, eng.models_dir),
                code=code_fingerprint(extra=(HERE / "perturb_ops.py",)),
                frames=fingerprint(*(f"{r.frame_id}:{r.tag}" for r in frames.sort_values("frame_id").itertuples())))
    new = run_chunks(chunks, lambda f: _perturb_chunk(eng, f), parts_dir, plan, restart, max_new, log)
    ids = [c for c, _ in chunks]
    if any(not (Path(parts_dir) / f"{c}.parquet").exists() for c in ids):
        return None, None, new
    df, stats = read_parts(parts_dir, ids)
    order = {f: i for i, f in enumerate(frames.frame_id)}
    fam = {f: i for i, f in enumerate(perturb_ops.FAMILIES)}
    rank = {n: i for i, n in enumerate(MODEL_NAMES)}
    df = df.assign(_o=df.frame_id.map(order), _m=df.model.map(rank), _f=df.family.map(fam))
    df = df.sort_values(["_o", "_m", "_f", "severity"]).drop(columns=["_o", "_m", "_f"]).reset_index(drop=True)
    df["severity"] = df.severity.astype("int32")
    return df[PERTURB_COLS], stats, new


# ---------------------------------------------------------------- script helpers

def is_smoke(pilot, limit):
    return bool(pilot or limit)


def guard_output(out, default, pilot, limit):
    """Refuse to put smoke output (pilot weights or a frame limit) at the real derived path or in the logs directory."""
    out = Path(out).resolve()
    if is_smoke(pilot, limit) and (out == Path(default).resolve() or out.parent in (DERIVED.resolve(), LOGS_DIR.resolve())):
        raise SystemExit(f"--pilot / --limit runs are smoke tests: pass --out somewhere other than {DERIVED} or {LOGS_DIR}")
    return out


def guard_side_path(path, pilot, limit):
    """A smoke run may not use an explicit --parts / --alloc inside the real-run directories (work/parts, work/logs)."""
    p = Path(path).resolve()
    if is_smoke(pilot, limit) and any(d.resolve() in (p, *p.parents) for d in (PARTS_ROOT, LOGS_DIR)):
        raise SystemExit(f"--pilot / --limit runs are smoke tests: {p} lies in {PARTS_ROOT} or {LOGS_DIR}, which hold real-run files")
    return p


def default_parts(out, smoke=False):
    """Chunk directory: work/parts/<stem> for a real run, <stem>_parts next to --out for a smoke run."""
    out = Path(out)
    return out.parent / f"{out.stem}_parts" if smoke else PARTS_ROOT / out.stem


def write_final(df, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(df, out)


def write_summary(out, smoke=False, **fields):
    """Run summary JSON: work/logs/<stem>_summary.json, or next to --out for a smoke run."""
    path = (Path(out).parent if smoke else LOGS_DIR) / f"{Path(out).stem}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=1, default=float))
    return path


def timing_summary(stats, n_frames):
    """Seconds per stage (load_cpu_s is summed worker time; the rest is main-process wall time) and frames/s."""
    keys = ("load_cpu_s", "forward_s", "post_s", "optics_s", "wall_s")
    out = {k: round(stats[k], 1) for k in keys if k in stats}
    out["frames_per_s"] = round(n_frames / stats["wall_s"], 2) if stats.get("wall_s") else None
    return out
