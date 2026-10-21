"""Training data: FOV-square 384 x 384 uint8 crops of the labeled Center 1 frames, held in RAM.

Each image is cropped to the square around its detected field of view (fov.parquet), zero-padded where the
circle leaves the frame, and resized with area interpolation; the mask (union of segments, > 0) takes the same
crop with nearest-neighbour resizing. All 9,593 frames take about 5.7 GB as two contiguous arrays, so there is
no disk cache: every process rebuilds them (about a minute with 12 workers).

patients_for() cuts the patient lists: training patients are the cv patients outside the held-out fold; in
pilot mode a patient-grouped inner validation split is carved from them so the held-out fold stays untouched.
"""
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import fov
from common import ROOT, abspath, load_config, read_image, read_mask

CFG = load_config()
DERIVED = ROOT / CFG["paths"]["derived"]
SIZE = int(CFG["fov"]["segmenter_input_px"])
WORKERS = 12
N_INNER = 10               # inner-validation patients in pilot mode
MIN_INNER_PER_DOMAIN = 2
INNER_DRAWS = 300          # random candidate splits; the one closest to the target frame share wins


def patients_for(fold, pilot=False, n_inner=N_INNER, folds=None):
    """Patient lists for one run: dict(train, val, heldout), each a sorted list of patient ids.

    fold is 0-4 or "final" (all cv patients, nothing held out). heldout = the fold's patients. val is empty
    unless pilot: then n_inner patients are drawn from the non-held-out ones, domains in proportion to their
    patient share, at random but deterministically (seeds.train, fold), picking the draw whose frame share
    is closest to its patient share so that a few giant videos do not dominate the validation set.
    """
    folds = pd.read_csv(DERIVED / "folds.csv") if folds is None else folds
    cv = folds[folds.role == "cv"]
    if fold == "final":
        if pilot:
            raise ValueError("pilot mode needs a held-out fold; use fold 0-4")
        return dict(train=sorted(cv.patient_id), val=[], heldout=[])
    fold = int(fold)
    held = cv[cv.fold == fold]
    if held.empty:
        raise ValueError(f"no cv patients in fold {fold}")
    rest = cv[cv.fold != fold]
    out = dict(train=sorted(rest.patient_id), val=[], heldout=sorted(held.patient_id))
    if pilot:
        val = _inner_split(rest, n_inner, [CFG["seeds"]["train"], fold])
        out["val"] = sorted(val)
        out["train"] = sorted(set(rest.patient_id) - set(val))
    return out


def _inner_split(rest, n_inner, seed):
    share = rest.domain.value_counts(normalize=True)
    want = (share * n_inner).clip(lower=MIN_INNER_PER_DOMAIN)
    n_dom = want.astype(int)
    for d in (want - n_dom).sort_values(ascending=False).index:   # largest remainders fill up to n_inner
        if n_dom.sum() >= n_inner:
            break
        n_dom[d] += 1
    while n_dom.sum() > n_inner:
        n_dom[n_dom.idxmax()] -= 1
    rng = np.random.default_rng(seed)
    by_dom = {d: g for d, g in rest.groupby("domain")}
    target = n_inner / len(rest)
    best, best_gap = None, np.inf
    for _ in range(INNER_DRAWS):
        pick = [by_dom[d].iloc[rng.choice(len(by_dom[d]), n_dom[d], replace=False)] for d in n_dom.index]
        pick = pd.concat(pick)
        gap = abs(pick.n_frames.sum() / rest.n_frames.sum() - target)
        if gap < best_gap:
            best, best_gap = pick, gap
    return list(best.patient_id)


def frame_table(partition="center1_labeled"):
    """Manifest rows of one partition joined with FOV geometry and (cv patients) fold, in manifest order.

    Frames without a detected field of view are dropped and counted in table.attrs["dropped_no_fov"]; the
    fallback geometry is not a valid crop, so fov.exclude_not_found must be true in analysis_plan.yaml.
    """
    if not CFG["fov"]["exclude_not_found"]:
        raise ValueError("fov.exclude_not_found is false: frames without a detected FOV would be cropped with fallback "
                         "geometry, which is not a valid crop; set it to true (the training data never keeps them)")
    m = pd.read_parquet(DERIVED / "manifest.parquet")
    m = m[m.partition == partition]
    g = pd.read_parquet(DERIVED / "fov.parquet")
    t = m.merge(g, on="frame_id", how="left", validate="one_to_one")
    if t.fov_found.isna().any():
        raise ValueError(f"{int(t.fov_found.isna().sum())} {partition} frames missing from fov.parquet")
    folds = pd.read_csv(DERIVED / "folds.csv")[["patient_id", "fold"]]
    t = t.merge(folds, on="patient_id", how="left", validate="many_to_one")
    n0 = len(t)
    t = t[t.fov_found.astype(bool)].reset_index(drop=True)
    t.attrs["dropped_no_fov"] = n0 - len(t)
    return t


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def _build_one(job):
    image_path, mask_path, cx, cy, r, size = job
    img = fov.crop_square(read_image(abspath(image_path)), cx, cy, r, size, "auto")
    msk = fov.crop_square(read_mask(abspath(mask_path)), cx, cy, r, size, "nearest")
    return img, msk.astype(np.uint8)


def build_crops(table, size=SIZE, workers=WORKERS, log=print):
    """(images uint8 N x S x S x 3, masks uint8 N x S x S in {0, 1}) aligned with the rows of table."""
    n = len(table)
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    masks = np.empty((n, size, size), dtype=np.uint8)
    jobs = [(r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, size) for r in table.itertuples()]
    t0 = time.time()
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, (img, msk) in enumerate(pool.imap(_build_one, jobs, chunksize=8)):
            images[i], masks[i] = img, msk
            if (i + 1) % 2000 == 0 or i + 1 == n:
                log(f"crops {i + 1}/{n} ({time.time() - t0:.0f} s)")
    return images, masks


def log_exclusions(table, log=print):
    log(f"Center 1 labeled frames: {len(table)} used, {table.attrs['dropped_no_fov']} excluded (no detected FOV)")


def load_center1(workers=WORKERS, log=print):
    """All Center 1 labeled frames: (table with fold column, images, masks)."""
    table = frame_table("center1_labeled")
    log_exclusions(table, log)
    images, masks = build_crops(table, workers=workers, log=log)
    return table, images, masks
