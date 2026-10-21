"""Predictions of both segmenters on every labeled frame -> derived/predictions.parquet.

10,659 frames x 2 models = 21,318 rows (the 3 Center 1 frames without a detected FOV are excluded and counted).
Center 1 frames take the out-of-fold model of their patient's fold, Center 2 the final models; see infer.py for the
metric definitions. Resumable: finished chunks live in --parts and are skipped on a rerun.

  05_predict.py                              the real run (T15)
  05_predict.py --limit 40 --pilot --out work/smoke/predictions.parquet   smoke test (chunks and summary land next to --out)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import infer  # noqa: E402
from common import load_config  # noqa: E402

CFG = load_config()
DEFAULT_OUT = infer.DERIVED / "predictions.parquet"
PARTITIONS = ("center1_labeled", "center2_labeled")


def build_table(pilot=False, limit=None):
    """Labeled frames with FOV geometry, fold and model assignment, in manifest order."""
    exp = CFG["dataset_counts_expected"]
    tables = [dataset.frame_table(p) for p in PARTITIONS]
    dropped = [t.attrs["dropped_no_fov"] for t in tables]
    want = [CFG["stream"]["expected_not_found_center1_labeled"], CFG["stream"]["expected_not_found_center2_labeled"]]
    assert dropped == want, f"frames without FOV {dropped}, expected {want}"
    table = pd.concat(tables, ignore_index=True)
    assert [len(t) for t in tables] == [exp["center1_frames"] - want[0], exp["center2_frames"] - want[1]]
    if pilot:                                    # smoke only: keep the frames the pilot model never trained on
        folds = pd.read_csv(infer.DERIVED / "folds.csv")
        table = table[~table.patient_id.isin(infer.trained_on("fold0_pilot", folds))]
    if limit:
        table = table.sample(n=min(limit, len(table)), random_state=CFG["seeds"]["global"]).sort_index()
    return infer.assign_models(table.reset_index(drop=True), pilot), dict(zip(PARTITIONS, dropped))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--parts", help="chunk directory (default v1/work/parts/<out stem>; smoke runs: <out stem>_parts next to --out)")
    ap.add_argument("--limit", type=int, help="random sample of N frames (smoke test)")
    ap.add_argument("--pilot", action="store_true", help="use the fold-0 pilot weights for every frame (smoke test)")
    ap.add_argument("--workers", type=int, default=12, help="loader processes; reads are latency bound on a cold disk")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--models-dir", default=str(infer.MODELS_DIR), help="weights directory (read only)")
    ap.add_argument("--chunk", type=int, default=infer.CHUNK)
    ap.add_argument("--restart", action="store_true", help="discard finished chunks of a different run")
    args = ap.parse_args()

    out = infer.guard_output(args.out, DEFAULT_OUT, args.pilot, args.limit)
    smoke = infer.is_smoke(args.pilot, args.limit)
    parts = infer.guard_side_path(args.parts, args.pilot, args.limit) if args.parts else infer.default_parts(out, smoke)
    table, dropped = build_table(args.pilot, args.limit)
    print(f"{len(table)} frames ({table.tag.value_counts().to_dict()}), excluded for no FOV: {dropped}", flush=True)
    t0 = time.time()
    with infer.Engine(infer.pick_device(args.device), args.workers, args.models_dir) as eng:
        df, stats, new = infer.predict_table(table, True, eng, parts, args.chunk, args.restart)
    if not smoke:
        exp = CFG["dataset_counts_expected"]
        n_rows = 2 * (exp["center1_frames"] + exp["center2_frames"] - sum(dropped.values()))
        assert len(df) == n_rows == 21318, f"{len(df)} rows, expected {n_rows}"
    infer.write_final(df, out)
    n = len(table)
    path = infer.write_summary(out, smoke, output=str(out), frames=n, rows=len(df), excluded_no_fov=dropped, pilot=args.pilot,
                               chunks_computed_this_call=new, wall_total_s=round(time.time() - t0, 1),
                               timing=infer.timing_summary(stats, n),
                               bg_frames_below_min_pixels=stats["bg_nan"], gt_empty_in_disk=stats["gt_empty"],
                               roi_empty_fallback={m: stats[f"roi_fallback_{m}"] for m in infer.MODEL_NAMES})
    print(f"wrote {out} ({len(df)} rows); summary {path}", flush=True)


if __name__ == "__main__":
    main()
