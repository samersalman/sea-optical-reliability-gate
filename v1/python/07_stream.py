"""Predictions of both final segmenters on the unlabeled stream -> derived/stream.parquet.

37,823 frames x 2 models = 75,646 rows (25 frames without a detected FOV are excluded and counted). Same columns as
predictions.parquet without iou, dice and gt_area_frac, plus video_patient (each stream patient is one video) and
t_index: 0-based rank of the frame inside its video among the ANALYSED frames, ordered by frame index. It has no
gaps; the time axis (1 frame/s) is the manifest frame_idx, which does skip frames that were excluded or missing.

  07_stream.py                                                   the real run (T15)
  07_stream.py --limit 40 --pilot --out work/smoke/stream.parquet   smoke test (chunks and summary land next to --out)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import infer  # noqa: E402
from common import load_config  # noqa: E402

CFG = load_config()
DEFAULT_OUT = infer.DERIVED / "stream.parquet"


def build_table(pilot=False, limit=None):
    t = dataset.frame_table("stream")
    dropped = t.attrs["dropped_no_fov"]
    assert dropped == CFG["stream"]["expected_not_found"], f"{dropped} stream frames without FOV"
    assert len(t) == CFG["stream"]["expected_frames"] - dropped
    assert t.groupby("patient_id").video_idx.nunique().eq(1).all(), "a stream patient has more than one video"
    if limit:                                    # whole leading stretch of a few videos keeps t_index meaningful
        t = t.sample(n=min(limit, len(t)), random_state=CFG["seeds"]["global"])
    t = t.sort_values(["patient_id", "frame_idx"]).reset_index(drop=True)
    t["video_patient"] = t.patient_id
    t["t_index"] = t.groupby("patient_id").cumcount().astype("int32")
    return infer.assign_models(t, pilot), dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--parts", help="chunk directory (default v1/work/parts/<out stem>; smoke runs: <out stem>_parts next to --out)")
    ap.add_argument("--limit", type=int, help="random sample of N frames (smoke test)")
    ap.add_argument("--pilot", action="store_true", help="use the fold-0 pilot weights (smoke test)")
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
    print(f"{len(table)} stream frames, excluded for no FOV: {dropped}", flush=True)
    t0 = time.time()
    with infer.Engine(infer.pick_device(args.device), args.workers, args.models_dir) as eng:
        df, stats, new = infer.predict_table(table, False, eng, parts, args.chunk, args.restart)
    df = df.merge(table[["frame_id", "video_patient", "t_index"]], on="frame_id", how="left", validate="many_to_one")
    assert df[["video_patient", "t_index"]].notna().all(axis=None)
    if not smoke:
        n_rows = 2 * (CFG["stream"]["expected_frames"] - dropped)
        assert len(df) == n_rows == 75646, f"{len(df)} rows, expected {n_rows}"
    infer.write_final(df[infer.STREAM_COLS], out)
    n = len(table)
    path = infer.write_summary(out, smoke, output=str(out), frames=n, rows=len(df), excluded_no_fov=dropped, pilot=args.pilot,
                               chunks_computed_this_call=new, wall_total_s=round(time.time() - t0, 1),
                               timing=infer.timing_summary(stats, n), bg_frames_below_min_pixels=stats["bg_nan"],
                               roi_empty_fallback={m: stats[f"roi_fallback_{m}"] for m in infer.MODEL_NAMES})
    print(f"wrote {out} ({len(df)} rows); summary {path}", flush=True)


if __name__ == "__main__":
    main()
