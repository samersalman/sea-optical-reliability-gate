"""Perturbation experiment: IoU and entropy of both segmenters under seven physical degradations -> derived/perturb.parquet.

Frames. 400 out-of-fold Center 1 frames with U-Net IoU >= 0.90 in predictions.parquet, stratified by
domain x scenario x instrument (the non-empty cells). Allocation, seeded (seeds.perturb):
  1. every cell asks for floor(400 / n_cells) frames (equal allocation, so rare instruments are not drowned by
     forceps) and takes min(asked, available);
  2. the shortfall of a short cell is filled from the NEAREST cell with spare frames: same domain and scenario
     first, then same domain, then any, then the cell with the fewest frames so far (random tie-break);
  3. the remainder of 400 / n_cells goes one frame at a time to the cell with the fewest frames and spare capacity.
  Inside a cell, frames are drawn one per patient per pass (patients in random order), which spreads the sample
  over as many patients as possible. If fewer than 400 frames qualify, all are taken and the shortage is reported.
The allocation table (available, asked, taken, short_by, extra = taken beyond the equal share) is written to --alloc and to the summary.

Per frame, every condition (baseline + 7 families x severities 1-5) runs native -> 512 canvas -> perturb_ops.apply with
the FULL uneroded FOV disk -> 11 optical metrics on the perturbed canvas with the ERODED disk -> resize to 384 (area)
-> OOF model of the frame's fold (both architectures). One Generator is built per (frame, family) and rebuilt for
every severity, so glare blobs and noise fields nest across severity.

Scoring grid. iou, dice, pred_area_frac and entropy_fov are computed on the 384 grid inside the eroded 384 disk against
the nearest-neighbour 384 GT crop. This differs from predictions.parquet, whose IoU is at native resolution; severity 0
here is recomputed on the same canvas path rather than copied from the OOF row, so it will differ slightly (the run
summary reports the difference).

Rows: frame x model x family x severity (0-5). Severity 0 is one image per frame, repeated under every family label so
each family has a full 0-5 curve; its param is the family's identity value. The 11 optical metrics belong to the image,
so the unet and segformer rows of the same (frame, family, severity) carry identical values.

  06_perturb.py                                            the real run (T15; needs predictions.parquet)
  06_perturb.py --pred work/smoke/predictions.parquet --limit 3 --pilot --out work/smoke/perturb.parquet
                                                           smoke test (chunks, summary and allocation land next to --out)

A real run (no --pilot / --limit) refuses to start unless --pred is derived/predictions.parquet with all 21,318 rows
and exactly 400 frames can be selected.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import infer  # noqa: E402
import perturb_ops  # noqa: E402
from common import load_config  # noqa: E402

CFG = load_config()
DEFAULT_OUT = infer.DERIVED / "perturb.parquet"
DEFAULT_PRED = infer.DERIVED / "predictions.parquet"
MIN_IOU = float(CFG["perturbation"]["min_oof_iou"])
_EXP = CFG["dataset_counts_expected"]
PRED_ROWS = 2 * (_EXP["center1_frames"] + _EXP["center2_frames"]
                 - CFG["stream"]["expected_not_found_center1_labeled"] - CFG["stream"]["expected_not_found_center2_labeled"])
STRATA = ["domain", "scenario", "instrument"]


def _closeness(a, b):
    """Sort key of cell b as a donor for cell a: smaller is nearer (differ in domain, then scenario, then instrument)."""
    return tuple(int(x != y) for x, y in zip(a, b))


def allocate(avail, total, rng):
    """Frames per cell, {cell: n}. avail: {cell: n eligible}. Takes everything when fewer than `total` qualify."""
    cells = sorted(avail)
    if sum(avail.values()) <= total:
        return dict(avail)
    ask = total // len(cells)
    taken = {c: min(avail[c], ask) for c in cells}

    def lowest(cands):
        low = min(taken[d] for d in cands)
        tied = sorted(d for d in cands if taken[d] == low)
        return tied[rng.integers(len(tied))]

    for c in cells:                                      # shortfall of a short cell: nearest cells with spare frames
        for _ in range(ask - avail[c]):
            donors = [d for d in cells if avail[d] > taken[d]]
            best = min(_closeness(c, d) for d in donors)
            taken[lowest([d for d in donors if _closeness(c, d) == best])] += 1
    while sum(taken.values()) < total:                   # remainder of total / n_cells
        taken[lowest([d for d in cells if avail[d] > taken[d]])] += 1
    return taken


def pick_frames(frames, k, rng):
    """k frame ids from one cell, one per patient per pass over the patients (random order)."""
    by_patient = {p: list(rng.permutation(sorted(g.frame_id))) for p, g in frames.groupby("patient_id")}
    order = list(rng.permutation(sorted(by_patient)))
    out = []
    while len(out) < k:
        for p in order:
            if by_patient[p] and len(out) < k:
                out.append(by_patient[p].pop())
    return out


def select_frames(eligible, n_frames, seed):
    """eligible: frames (STRATA columns plus patient_id, frame_id). Returns (chosen frame ids, allocation table)."""
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in eligible.groupby(STRATA)}
    taken = allocate({c: len(g) for c, g in groups.items()}, n_frames, rng)
    ids = []
    for c in sorted(groups):
        ids += pick_frames(groups[c], taken[c], rng)
    ask = n_frames // len(groups)
    table = pd.DataFrame([dict(zip(STRATA, c), available=len(groups[c]), asked=ask, taken=taken[c],
                               short_by=max(ask - len(groups[c]), 0), extra=taken[c] - min(len(groups[c]), ask))
                          for c in sorted(groups)])
    return ids, table


def check_real_pred(path, pred):
    """A real run selects from the real predictions.parquet (all rows), not a smoke or partial file."""
    if Path(path).resolve() != DEFAULT_PRED.resolve():
        raise SystemExit(f"a real run selects from {DEFAULT_PRED}; got --pred {path} (smoke inputs need --pilot or --limit)")
    if len(pred) != PRED_ROWS:
        raise SystemExit(f"{path} has {len(pred)} rows, expected {PRED_ROWS}: rerun 05_predict.py before the perturbation run")


def check_real_frames(n_selected, n_frames):
    if n_selected != n_frames:
        raise SystemExit(f"selected {n_selected} frames, the real run needs exactly {n_frames}: see the allocation table")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=str(DEFAULT_PRED), help="predictions.parquet to select from")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--parts", help="chunk directory (default v1/work/parts/<out stem>; smoke runs: <out stem>_parts next to --out)")
    ap.add_argument("--alloc", help="allocation CSV (default work/logs/<out stem>_allocation.csv; smoke runs: next to --out)")
    ap.add_argument("--limit", type=int, help="number of frames instead of the planned 400 (smoke test)")
    ap.add_argument("--pilot", action="store_true", help="use the fold-0 pilot weights; fold-0 frames only (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--models-dir", default=str(infer.MODELS_DIR), help="weights directory (read only)")
    ap.add_argument("--chunk", type=int, default=infer.PERTURB_CHUNK)
    ap.add_argument("--restart", action="store_true", help="discard finished chunks of a different run")
    args = ap.parse_args()

    out = infer.guard_output(args.out, DEFAULT_OUT, args.pilot, args.limit)
    smoke = infer.is_smoke(args.pilot, args.limit)
    parts = infer.guard_side_path(args.parts, args.pilot, args.limit) if args.parts else infer.default_parts(out, smoke)
    n_frames = args.limit or int(CFG["perturbation"]["n_frames"])
    pred = pd.read_parquet(args.pred)
    if not smoke:
        check_real_pred(args.pred, pred)
    oof = pred[(pred.model == "unet") & (pred.source == "oof") & (pred.iou >= MIN_IOU)]
    frames = dataset.frame_table("center1_labeled")
    frames = frames[frames.frame_id.isin(oof.frame_id)]
    if args.pilot:
        folds = pd.read_csv(infer.DERIVED / "folds.csv")
        frames = frames[~frames.patient_id.isin(infer.trained_on("fold0_pilot", folds))]
    if frames.empty:
        raise SystemExit(f"no OOF frame with U-Net IoU >= {MIN_IOU} in {args.pred}")
    ids, alloc = select_frames(frames, n_frames, CFG["seeds"]["perturb"])
    frames = infer.assign_models(frames[frames.frame_id.isin(ids)].reset_index(drop=True), args.pilot)
    short = n_frames - len(frames)
    if args.alloc:
        alloc_path = infer.guard_side_path(args.alloc, args.pilot, args.limit)
    else:
        alloc_path = (out.parent if smoke else infer.LOGS_DIR) / f"{out.stem}_allocation.csv"
    alloc_path.parent.mkdir(parents=True, exist_ok=True)
    alloc.to_csv(alloc_path, index=False)
    print(f"{len(oof)} eligible OOF frames (U-Net IoU >= {MIN_IOU}), selected {len(frames)} of {n_frames} requested "
          f"from {len(alloc)} strata, {frames.patient_id.nunique()} patients; allocation in {alloc_path}", flush=True)
    if short > 0:
        print(f"SHORT by {short} frames: fewer than {n_frames} frames qualify", flush=True)
    if not smoke:
        check_real_frames(len(frames), n_frames)

    t0 = time.time()
    with infer.Engine(infer.pick_device(args.device), args.workers, args.models_dir) as eng:
        df, stats, new = infer.perturb_table(frames, eng, parts, args.chunk, args.restart)
    check_grid(df, frames)
    base = df[(df.severity == 0) & (df.family == "defocus") & (df.model == "unet")].set_index("frame_id").iou
    ref = oof.set_index("frame_id").iou.reindex(base.index)
    infer.write_final(df, out)
    path = infer.write_summary(out, smoke, output=str(out), frames=len(frames), requested=n_frames, short_by=max(short, 0),
                               rows=len(df), patients=int(frames.patient_id.nunique()), pilot=args.pilot,
                               chunks_computed_this_call=new, wall_total_s=round(time.time() - t0, 1),
                               timing=infer.timing_summary(stats, len(frames)),
                               allocation=alloc.to_dict("records"),
                               by_domain=frames.domain.value_counts().to_dict(), by_scenario=frames.scenario.value_counts().to_dict(),
                               by_instrument=frames.instrument.value_counts().to_dict(),
                               sev0_iou384_minus_oof_native_iou=dict(mean=float((base - ref).mean()), median=float((base - ref).median()),
                                                                     abs_median=float((base - ref).abs().median()),
                                                                     abs_max=float((base - ref).abs().max())))
    print(f"wrote {out} ({len(df)} rows); summary {path}", flush=True)


def check_grid(df, frames):
    """Completeness, baseline repetition and image-level metric identity across the two model rows."""
    fams, sevs = perturb_ops.FAMILIES, perturb_ops.SEVERITIES
    assert len(df) == len(frames) * len(infer.MODEL_NAMES) * len(fams) * len(sevs), "grid incomplete"
    assert not df.duplicated(["frame_id", "model", "family", "severity"]).any()
    key = ["frame_id", "family", "severity"]
    m = list(infer.METRICS)
    a, b = (df[df.model == n].set_index(key)[m] for n in infer.MODEL_NAMES)
    pd.testing.assert_frame_equal(a, b.loc[a.index], check_exact=True)
    s0 = df[df.severity == 0].drop(columns=["family", "param"]).drop_duplicates()
    assert len(s0) == len(frames) * len(infer.MODEL_NAMES), "severity 0 differs between families"


if __name__ == "__main__":
    main()
