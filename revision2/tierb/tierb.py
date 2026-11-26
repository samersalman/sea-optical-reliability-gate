"""Revision 2, Tier B pixel analyses (post hoc, exploratory). Reuses the locked v1 code read-only.

Subcommands (run with the README environment and PYTHONDONTWRITEBYTECODE=1 so nothing is written into v1;
set OPTICS_V1 to the v1 folder if it is not next to this folder's parent):
  boundary   all labeled frames, original model assignment (Center 1 out-of-fold, Center 2 final models):
             recompute the masks, check IoU against derived/predictions.parquet, add Boundary IoU and HD95
  harmonize  Center 1 frames resampled so the field-of-view radius equals the Center 2 median, then the 11 optical
             metrics, out-of-fold segmentation, IoU, entropy and disagreement recomputed at that resolution
  perturb2   the 400 perturbation frames: localized (one feathered disk, 25% of the FOV area) and combined
             (two families in sequence) degradations at severities 3 and 5
  latency    per-frame wall time of the optical metrics, one segmenter with entropy, and the two-model ensemble

Outputs: <this folder>/../out/tierb_<name>.parquet and tierb_latency.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_PY = Path(os.environ.get("OPTICS_V1", HERE.parents[1] / "v1")) / "python"   # repository layout by default
OUT = HERE.parent / "out"
sys.path.insert(0, str(V1_PY))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import dataset  # noqa: E402
import fov  # noqa: E402
import infer  # noqa: E402
import optics_lib  # noqa: E402
import perturb_ops  # noqa: E402
import common  # noqa: E402
from common import abspath  # noqa: E402
from models import MODEL_NAMES  # noqa: E402


def _retry(fn, tries=6):
    """iCloud may evict a file under storage pressure; the first read then fails while it downloads again."""
    def wrapped(path):
        for k in range(tries):
            try:
                return fn(path)
            except OSError:
                if k == tries - 1:
                    raise
                time.sleep(5 * (k + 1))
    return wrapped


read_image, read_mask = _retry(common.read_image), _retry(common.read_mask)
infer.read_image, infer.read_mask = read_image, read_mask      # also used by infer.load_job in the workers
PARTS = OUT / "tierb_parts"

SIZE, CANON, THR = infer.SIZE, infer.CANON, infer.THRESHOLD
METRICS = list(optics_lib.METRICS)
BIOU_FRAC = 0.02          # Boundary IoU band width: 2% of the image diagonal (Cheng et al., CVPR 2021)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- boundary metrics

def _band(mask, d):
    """Pixels of mask within d px of its contour (mask minus its erosion by a (2d+1) disk)."""
    if not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    er = cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return mask & ~er


def boundary_iou(pred, gt, region):
    """Boundary IoU inside region; both empty -> 1, one empty -> 0."""
    p, g = pred & region, gt & region
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    d = max(1, int(round(BIOU_FRAC * np.hypot(*p.shape))))
    pb, gb = _band(p, d), _band(g, d)
    inter, union = int((pb & gb).sum()), int((pb | gb).sum())
    return inter / union if union else 1.0


def hd95(pred, gt, region):
    """Symmetric 95th-percentile Hausdorff distance between the two contours, in px; NaN when either mask is empty."""
    p, g = pred & region, gt & region
    if not p.any() or not g.any():
        return float("nan")
    pb, gb = _band(p, 1), _band(g, 1)
    dt_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 5)
    return float(max(np.percentile(dt_g[pb], 95), np.percentile(dt_p[gb], 95)))


def _labeled_table(partitions=("center1_labeled", "center2_labeled")):
    t = pd.concat([dataset.frame_table(p) for p in partitions], ignore_index=True)
    t = infer.assign_models(t)
    infer.check_selection(t, pd.read_csv(infer.DERIVED / "folds.csv"))
    return t


def _frames_by_tag(table, fn_job, fn_row, eng, label):
    """Run both segmenters over table grouped by model tag; fn_job(row) -> job for fn_load, fn_row -> output rows."""
    rows, done, t0 = [], 0, time.time()
    for tag, g in table.sort_values(["tag", "frame_id"]).groupby("tag", sort=True):
        part = PARTS / label / f"{tag}_{len(g)}.parquet"
        if part.exists():
            rows.extend(pd.read_parquet(part).to_dict("records"))
            done += len(g)
            log(f"{label}: {tag} reused from {part.name}")
            continue
        n0 = len(rows)
        models = eng.models(tag)
        recs = g.to_dict("records")
        batch = []

        def flush(batch):
            crops = np.stack([b[1][0] for b in batch])
            probs = {n: infer.forward_probs(models[n], crops) for n in MODEL_NAMES}
            for j, (rec, res) in enumerate(batch):
                rows.extend(fn_row(rec, res, {n: probs[n][j] for n in MODEL_NAMES}))

        for rec, res in zip(recs, eng.imap(fn_job[0], [fn_job[1](r) for r in recs])):
            batch.append((rec, res))
            if len(batch) == eng.batch:
                flush(batch)
                done += len(batch)
                batch = []
        if batch:
            flush(batch)
            done += len(batch)
        part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows[n0:]).to_parquet(part, index=False)
        log(f"{label}: {tag} finished, {done}/{len(table)} frames, {time.time() - t0:.0f} s")
    eng.sync()
    return pd.DataFrame(rows)


def _seg_rows(rec, probs, side, gt, extra=None):
    dn, d384 = infer.disk(side), infer.disk(SIZE)
    out, masks384 = [], {}
    for n in MODEL_NAMES:
        row, p384, pnat, _ = infer.seg_stats(probs[n], side, gt)
        masks384[n] = p384
        row.update(boundary_iou=boundary_iou(pnat, gt, dn), hd95_px=hd95(pnat, gt, dn), side_px=side)
        out.append((n, row))
    ens = infer.ens_dis(masks384["unet"], masks384["segformer"], d384)
    ident = dict(frame_id=rec["frame_id"], patient_id=rec["patient_id"], partition=rec["partition"],
                 source=rec["source"], fold=rec["fold"])
    return [{**ident, "model": n, **row, "ens_dis": ens, **(extra or {})} for n, row in out]


def cmd_boundary(eng, limit=None):
    table = _labeled_table()
    if limit:
        table = table.groupby("partition", group_keys=False).apply(lambda g: g.sample(min(limit, len(g)), random_state=1))
    job = (infer.load_job, lambda r: (r["image_path"], r["mask_path"], r["fov_cx"], r["fov_cy"], r["fov_r"], int(r["crop_side"])))
    df = _frames_by_tag(table, job, lambda rec, res, probs: _seg_rows(rec, probs, int(rec["crop_side"]), res[2]), eng, "boundary")
    stored = pd.read_parquet(infer.DERIVED / "predictions.parquet", columns=["frame_id", "model", "iou", "entropy_fov", "ens_dis"])
    chk = df.merge(stored, on=["frame_id", "model"], suffixes=("", "_stored"))
    diff = {k: float(np.abs(chk[k] - chk[f"{k}_stored"]).max()) for k in ("iou", "entropy_fov", "ens_dis")}
    log(f"max abs difference vs stored predictions: {diff}")
    return df, dict(n_rows=len(df), max_abs_diff_vs_stored=diff, biou_band_frac_of_diagonal=BIOU_FRAC)


# ---------------------------------------------------------------- resolution harmonization

def _target_radius():
    f = pd.read_parquet(infer.DERIVED / "fov.parquet")
    m = pd.read_parquet(infer.DERIVED / "manifest.parquet", columns=["frame_id", "center", "labeled"])
    x = f.merge(m, on="frame_id")
    return float(x[(x.center == 2) & x.labeled].fov_r.median())


R_TARGET = None


def harm_job(job):
    """Native frame downsampled (area) by s = R_TARGET / r when s < 1, then the standard crops, canvas and metrics."""
    t0 = time.perf_counter()
    image_path, mask_path, cx, cy, r, r_target = job
    img, msk = read_image(abspath(image_path)), read_mask(abspath(mask_path))
    s = min(1.0, r_target / r)
    if s < 1.0:
        h, w = img.shape[:2]
        size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        cx, cy, r = cx * s, cy * s, r * s
    side = fov.crop_box(cx, cy, r)[2]
    crop = fov.crop_square(img, cx, cy, r, SIZE, "auto")
    canvas, disk = fov.canonical_canvas(img, dict(fov_cx=cx, fov_cy=cy, fov_r=r))
    met = optics_lib.compute_metrics(canvas, disk)
    gt = fov.crop_square(msk, cx, cy, r, side, "nearest")
    return crop, met, gt, side, s, time.perf_counter() - t0


def cmd_harmonize(eng, limit=None):
    rt = _target_radius()
    table = _labeled_table(("center1_labeled",))
    if limit:
        table = table.sample(limit, random_state=1)
    job = (harm_job, lambda r: (r["image_path"], r["mask_path"], r["fov_cx"], r["fov_cy"], r["fov_r"], rt))

    def row_fn(rec, res, probs):
        _, met, gt, side, s, _ = res
        return _seg_rows(rec, probs, side, gt, extra={**met, "scale": s})
    df = _frames_by_tag(table, job, row_fn, eng, "harmonize")
    return df, dict(n_rows=len(df), target_fov_radius_px=rt, rule="native frame area-downsampled so fov_r equals the Center 2 median; never upsampled")


# ---------------------------------------------------------------- localized and combined perturbations

LOCAL_FAMS = ("defocus", "haze", "exposure_under", "exposure_over", "blood_absorption", "noise")
PAIRS = (("defocus", "haze"), ("haze", "exposure_under"), ("blood_absorption", "exposure_under"),
         ("exposure_over", "glare"), ("defocus", "noise"), ("glare", "noise"))
SEVS = (3, 5)
LOCAL_AREA = 0.25
CONDS = ([("none", "baseline", 0)] + [("local", f, s) for f in LOCAL_FAMS for s in SEVS]
         + [("combined", f"{a}+{b}", s) for a, b in PAIRS for s in SEVS])


def _local_alpha(frame_id, full):
    """Feathered disk of LOCAL_AREA of the FOV area, centre drawn inside half the FOV radius (fixed per frame)."""
    h = int.from_bytes(hashlib.sha256(f"{frame_id}:local".encode()).digest()[:8], "big")
    rng = np.random.default_rng([infer.SEED_PERTURB, h])
    R = CANON / 2.0
    rl = np.sqrt(LOCAL_AREA) * R
    rho, th = 0.5 * R * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    c = (CANON - 1) / 2.0
    yy, xx = np.mgrid[0:CANON, 0:CANON]
    hard = ((xx - (c + rho * np.cos(th))) ** 2 + (yy - (c + rho * np.sin(th))) ** 2 <= rl ** 2).astype(np.float32)
    a = cv2.GaussianBlur(hard, (0, 0), sigmaX=0.1 * rl)
    return (a * full)[..., None]


def perturb2_job(job):
    t0 = time.perf_counter()
    frame_id, image_path, mask_path, cx, cy, r = job
    img = read_image(abspath(image_path))
    canvas, eroded = fov.canonical_canvas(img, dict(fov_cx=cx, fov_cy=cy, fov_r=r))
    full = fov.fov_disk_mask(CANON, 0.0)
    gt384 = fov.crop_square(read_mask(abspath(mask_path)), cx, cy, r, SIZE, "nearest")
    alpha = _local_alpha(frame_id, full)
    images = np.empty((len(CONDS), SIZE, SIZE, 3), np.float32)
    metrics = []
    for k, (kind, name, sev) in enumerate(CONDS):
        if kind == "none":
            pert = canvas
        elif kind == "local":
            whole = perturb_ops.apply(canvas, full, name, sev, infer.family_rng(frame_id, name))
            pert = (alpha * whole + (1 - alpha) * canvas).astype(np.float32)
        else:
            pert = canvas
            for fam in name.split("+"):
                pert = perturb_ops.apply(pert, full, fam, sev, infer.family_rng(frame_id, fam))
        metrics.append(optics_lib.compute_metrics(pert, eroded))
        images[k] = cv2.resize(pert, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return images, metrics, gt384, time.perf_counter() - t0


def cmd_perturb2(eng, limit=None):
    ids = pd.read_parquet(infer.DERIVED / "perturb.parquet", columns=["frame_id", "domain"]).drop_duplicates("frame_id")
    table = _labeled_table(("center1_labeled",))
    table = table[table.frame_id.isin(ids.frame_id)].merge(ids, on="frame_id", how="left", suffixes=("", "_p"))
    assert len(table) == len(ids), "perturbation frames missing from the labeled table"
    if limit:
        table = table.sample(limit, random_state=1)
    rows, t0 = [], time.time()
    for tag, g in table.sort_values(["tag", "frame_id"]).groupby("tag", sort=True):
        part = PARTS / "perturb2" / f"{tag}_{len(g)}.parquet"
        if part.exists():
            rows.extend(pd.read_parquet(part).to_dict("records"))
            continue
        n0 = len(rows)
        models = eng.models(tag)
        recs = g.to_dict("records")
        jobs = [(r["frame_id"], r["image_path"], r["mask_path"], r["fov_cx"], r["fov_cy"], r["fov_r"]) for r in recs]
        for rec, (images, metrics, gt, _) in zip(recs, eng.imap(perturb2_job, jobs, window=eng.workers + 2)):
            stats = {}
            for n in MODEL_NAMES:
                probs = np.concatenate([infer.forward_probs(models[n], images[i:i + eng.batch]) for i in range(0, len(images), eng.batch)])
                stats[n] = [infer.grid_stats(p, gt) for p in probs]
            for k, (kind, name, sev) in enumerate(CONDS):
                for n in MODEL_NAMES:
                    rows.append(dict(frame_id=rec["frame_id"], patient_id=rec["patient_id"], domain=rec["domain"], model=n,
                                     kind=kind, condition=name, severity=sev, **stats[n][k], **metrics[k]))
        part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows[n0:]).to_parquet(part, index=False)
        log(f"perturb2: {tag} finished, {len(rows) // (2 * len(CONDS))}/{len(table)} frames, {time.time() - t0:.0f} s")
    eng.sync()
    return pd.DataFrame(rows), dict(n_rows=len(rows), local_area_frac=LOCAL_AREA, severities=list(SEVS),
                                    local_families=list(LOCAL_FAMS), pairs=["+".join(p) for p in PAIRS])


# ---------------------------------------------------------------- latency

def _ms(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(1000 * (time.perf_counter() - t))
    return out


def _summ(x):
    x = np.asarray(x)
    return dict(median_ms=round(float(np.median(x)), 2), p10_ms=round(float(np.percentile(x, 10)), 2),
                p90_ms=round(float(np.percentile(x, 90)), 2), n=int(x.size))


def cmd_latency(n_frames=100, reps=3):
    cv2.setNumThreads(1)
    table = _labeled_table(("center2_labeled",)).sample(n_frames, random_state=1)
    imgs = [read_image(abspath(p)) for p in table.image_path]
    res = {"frames": n_frames, "reps_per_frame": reps, "torch": torch.__version__, "machine": os.uname().machine,
           "note": "decode excluded; single CPU thread for metrics; batch size 1"}
    t_fov, t_opt = [], []
    for img in imgs:
        t_fov += _ms(lambda: fov.detect_fov(img), reps)
        f = fov.detect_fov(img)
        t_opt += _ms(lambda: optics_lib.compute_metrics(*fov.canonical_canvas(img, f)), reps)
    res["fov_detection"] = _summ(t_fov)
    res["optical_metrics_11"] = _summ(t_opt)
    crops = [fov.crop_square(img, r.fov_cx, r.fov_cy, r.fov_r, SIZE, "auto") for img, r in zip(imgs, table.itertuples())]
    for dev in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
        models = infer.load_models("final", dev)
        d384 = infer.disk(SIZE)

        def one(n, c):
            p = infer.forward_probs(models[n], c[None])[0]
            if dev == "mps":
                torch.mps.synchronize()
            return p

        for n in MODEL_NAMES:                      # warm-up
            for c in crops[:5]:
                one(n, c)
        for n in MODEL_NAMES:
            res[f"{dev}_{n}_forward"] = _summ([t for c in crops for t in _ms(lambda: one(n, c), reps)])
            res[f"{dev}_{n}_forward_plus_entropy"] = _summ([t for c in crops for t in _ms(
                lambda: float(infer.binary_entropy_bits(one(n, c))[d384].mean()), reps)])

        def ens(c):
            pu, ps = one("unet", c), one("segformer", c)
            return infer.ens_dis(pu > THR, ps > THR, d384)
        res[f"{dev}_ensemble_disagreement"] = _summ([t for c in crops for t in _ms(lambda: ens(c), reps)])
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["boundary", "harmonize", "perturb2", "latency"])
    ap.add_argument("--limit", type=int, help="smoke test: N frames (per partition for boundary)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.limit else ""
    if args.cmd == "latency":
        res = cmd_latency(args.limit or 100)
        (OUT / f"tierb_latency{tag}.json").write_text(json.dumps(res, indent=1))
        log(json.dumps(res, indent=1))
        return
    t0 = time.time()
    with infer.Engine(infer.pick_device("auto"), args.workers) as eng:
        df, meta = {"boundary": cmd_boundary, "harmonize": cmd_harmonize, "perturb2": cmd_perturb2}[args.cmd](eng, args.limit)
    df.to_parquet(OUT / f"tierb_{args.cmd}{tag}.parquet", index=False)
    meta.update(wall_s=round(time.time() - t0, 1), device=eng.device)
    (OUT / f"tierb_{args.cmd}{tag}.json").write_text(json.dumps(meta, indent=1, default=float))
    log(f"wrote tierb_{args.cmd}{tag}.parquet ({len(df)} rows) {meta}")


if __name__ == "__main__":
    main()
