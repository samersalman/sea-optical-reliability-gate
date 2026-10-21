"""Stage 02: FOV geometry and the 11 optical metrics for every frame of the manifest.

Subcommands
  fov      derived/fov.parquet     (fov.detect_fov per frame)
  metrics  derived/optics.parquet  (fov.canonical_canvas + optics_lib.compute_metrics; needs fov.parquet)
  g1       Gate G1 evidence: work/logs/g1_report.json and the overlay montages in work/qc/g1
  all      fov, metrics, g1

Frames are cut into chunks of the manifest's row order and a fork pool computes one chunk per task. Each finished
chunk is written to work/logs/optics_parts/<stage>/ at once, so a killed run resumes with the chunks that are
missing; a stamp (code and input fingerprint) discards parts that no longer match. The merged outputs follow the
manifest's row order. Frames without a plausible FOV keep their fallback geometry, get NaN metrics and are counted.
The ground-truth mask is read only by the g1 overlays, never by a metric.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fov as fovlib  # noqa: E402
import optics_lib  # noqa: E402
from common import ROOT, abspath, load_config, read_image, read_mask  # noqa: E402

FOV_COLUMNS = ["frame_id", "fov_found", "fov_cx", "fov_cy", "fov_r", "fov_method", "fov_resid",
               "crop_x0", "crop_y0", "crop_side"]
OPTICS_COLUMNS = ["frame_id", *optics_lib.METRICS]
QUANTILES = {"min": 0.0, "p1": 0.01, "p50": 0.5, "p99": 0.99, "max": 1.0}
TILE_W, TILE_H, LABEL_H, MONTAGE_COLS = 560, 360, 44, 4
YELLOW, GREEN, WHITE, RED = (255, 215, 0), (0, 255, 80), (255, 255, 255), (255, 40, 40)


# ---------------------------------------------------------------- chunked pool

def _make_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log


def _init_worker():
    cv2.setNumThreads(1)


def _fov_chunk(job):
    i, tasks = job
    rows = []
    for frame_id, rel in tasks:
        f = fovlib.detect_fov(read_image(abspath(rel)))
        x0, y0, side = fovlib.crop_box(f["cx"], f["cy"], f["r"])
        rows.append(dict(frame_id=frame_id, fov_found=bool(f["found"]), fov_cx=f["cx"], fov_cy=f["cy"], fov_r=f["r"],
                         fov_method=f["method"], fov_resid=f["resid"], crop_x0=x0, crop_y0=y0, crop_side=side))
    return i, rows


def _metrics_chunk(job):
    i, tasks = job
    rows = []
    for frame_id, rel, found, cx, cy, r in tasks:
        if not found:
            rows.append(dict(frame_id=frame_id, **dict.fromkeys(optics_lib.METRICS, float("nan"))))
            continue
        canvas, disk = fovlib.canonical_canvas(read_image(abspath(rel)), dict(cx=cx, cy=cy, r=r))
        rows.append(dict(frame_id=frame_id, **optics_lib.compute_metrics(canvas, disk)))
    return i, rows


def _sha(*blobs):
    h = hashlib.sha256()
    for b in blobs:
        h.update(b if isinstance(b, bytes) else str(b).encode())
    return h.hexdigest()


def _code_sha(*modules):
    return _sha(*[Path(m.__file__).read_bytes() for m in modules])


def _part(parts_dir, i):
    return parts_dir / f"part_{i:05d}.parquet"


def _part_ok(path, frame_ids):
    try:
        return path.exists() and pd.read_parquet(path, columns=["frame_id"]).frame_id.tolist() == list(frame_ids)
    except Exception:
        return False


def _run_chunks(stage, frame_ids, make_tasks, worker, parts_root, fingerprint, workers, chunk, log):
    """Compute the chunks whose part file is missing or stale. make_tasks(lo, hi) -> payload for rows lo:hi.
    Returns dict(done, skipped, wall_s) for this invocation; the parts dir holds the results."""
    parts_dir = Path(parts_root) / stage
    parts_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = parts_dir / "_stamp.json"
    stamp = dict(chunk=chunk, fingerprint=fingerprint, runs=[])
    if stamp_path.exists():
        old = json.loads(stamp_path.read_text())
        if old.get("chunk") == chunk and old.get("fingerprint") == fingerprint:
            stamp = old
        else:
            log(f"{stage}: inputs or code changed since the parts were written, discarding them")
            for p in parts_dir.glob("part_*.parquet"):
                p.unlink()
    n = len(frame_ids)
    bounds = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    for stale in parts_dir.glob("part_*.parquet"):
        if int(stale.stem.split("_")[1]) >= len(bounds):
            stale.unlink()
    todo = [i for i, (lo, hi) in enumerate(bounds) if not _part_ok(_part(parts_dir, i), frame_ids[lo:hi])]
    skipped = len(bounds) - len(todo)
    log(f"{stage}: {n} frames, {len(bounds)} chunks of {chunk}, {skipped} already done, {len(todo)} to run on {workers} workers")
    t0 = time.time()
    frames_done = 0
    total = sum(bounds[i][1] - bounds[i][0] for i in todo)
    if todo:
        with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
            jobs = ((i, make_tasks(*bounds[i])) for i in todo)
            for k, (i, rows) in enumerate(pool.imap_unordered(worker, jobs), 1):
                tmp = parts_dir / f"tmp_{i:05d}.parquet"
                pd.DataFrame(rows).to_parquet(tmp, index=False)
                os.replace(tmp, _part(parts_dir, i))
                frames_done += len(rows)
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t0
                    rate = frames_done / el
                    eta = (total - frames_done) / rate
                    log(f"{stage}: chunk {k}/{len(todo)}, {frames_done} frames, {rate:.0f} frames/s, "
                        f"elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min")
    wall = time.time() - t0
    stamp["runs"].append(dict(frames=frames_done, wall_s=round(wall, 2), workers=workers, at=datetime.now().isoformat(timespec="seconds")))
    stamp_path.write_text(json.dumps(stamp, indent=1))
    return dict(done=len(todo), skipped=skipped, wall_s=wall)


def _merge(stage, parts_root, n_chunks, frame_ids, columns, out_path):
    parts_dir = Path(parts_root) / stage
    df = pd.concat([pd.read_parquet(_part(parts_dir, i)) for i in range(n_chunks)], ignore_index=True)[columns]
    assert df.frame_id.tolist() == list(frame_ids), f"{stage}: merged frame_id order differs from the manifest"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return df


# ---------------------------------------------------------------- stages

def run_fov(manifest_path, out_path, parts_root, workers=12, chunk=250, log=print):
    """Detect the FOV of every manifest frame and write out_path (contract: derived/fov.parquet)."""
    m = pd.read_parquet(manifest_path, columns=["frame_id", "image_path"])
    ids, paths = m.frame_id.tolist(), m.image_path.tolist()
    fingerprint = _sha(_code_sha(fovlib), "\n".join(f"{a}|{b}" for a, b in zip(ids, paths)))
    res = _run_chunks("fov", ids, lambda lo, hi: list(zip(ids[lo:hi], paths[lo:hi])), _fov_chunk, parts_root,
                      fingerprint, workers, chunk, log)
    df = _merge("fov", parts_root, math.ceil(len(ids) / chunk), ids, FOV_COLUMNS, out_path)
    log(f"fov: wrote {out_path} ({len(df)} rows), found {int(df.fov_found.sum())}, not found {int((~df.fov_found).sum())}")
    return res


def run_metrics(manifest_path, fov_path, out_path, parts_root, workers=12, chunk=250, log=print):
    """Compute the 11 metrics for every frame (NaN where no FOV was found) and write out_path."""
    m = pd.read_parquet(manifest_path, columns=["frame_id", "image_path"])
    f = pd.read_parquet(fov_path)
    ids, paths = m.frame_id.tolist(), m.image_path.tolist()
    assert f.frame_id.tolist() == ids, "fov.parquet does not follow the manifest row order; rerun the fov stage"
    geom = list(zip(f.fov_found.tolist(), f.fov_cx.tolist(), f.fov_cy.tolist(), f.fov_r.tolist()))
    fingerprint = _sha(_code_sha(fovlib, optics_lib), "\n".join(f"{a}|{b}" for a, b in zip(ids, paths)),
                       pd.util.hash_pandas_object(f, index=False).values.tobytes())
    res = _run_chunks("metrics", ids,
                      lambda lo, hi: [(ids[k], paths[k], *geom[k]) for k in range(lo, hi)],
                      _metrics_chunk, parts_root, fingerprint, workers, chunk, log)
    df = _merge("metrics", parts_root, math.ceil(len(ids) / chunk), ids, OPTICS_COLUMNS, out_path)
    n_nan = int(df[list(optics_lib.METRICS)].isna().any(axis=1).sum())
    log(f"metrics: wrote {out_path} ({len(df)} rows), {n_nan} rows with NaN metrics "
        f"({int((~f.fov_found).sum())} frames without a FOV)")
    return res


# ---------------------------------------------------------------- G1 report

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _quant(s):
    s = pd.Series(s, dtype="float64")
    s = s[np.isfinite(s)]
    out = {"n": int(len(s))}
    out.update({k: (float(s.quantile(q)) if len(s) else None) for k, q in QUANTILES.items()})
    return out


def worst_residual(manifest, fov, k=20):
    """The k found frames with the largest circle-fit residual, with the columns needed to open them."""
    d = manifest.merge(fov, on="frame_id")
    d = d[d.fov_found].sort_values("fov_resid", ascending=False, kind="stable").head(k)
    return d


def _stage_runtime(parts_root, stage):
    p = Path(parts_root) / stage / "_stamp.json"
    if not p.exists():
        return None
    runs = json.loads(p.read_text())["runs"]
    return dict(wall_s=sum(r["wall_s"] for r in runs), frames_computed=sum(r["frames"] for r in runs),
                invocations=len(runs), workers=runs[-1]["workers"] if runs else None)


def g1_report(manifest_path, fov_path, optics_path, parts_root, out_json, min_found_rate, log=print):
    m = pd.read_parquet(manifest_path)
    f = pd.read_parquet(fov_path)
    o = pd.read_parquet(optics_path)
    assert m.frame_id.tolist() == f.frame_id.tolist() == o.frame_id.tolist(), "manifest, fov and optics rows differ"
    d = m.merge(f, on="frame_id").merge(o, on="frame_id")
    metrics = list(optics_lib.METRICS)
    d["r_over_min_side"] = d.fov_r / np.minimum(d.width, d.height)

    def found_rate(g):
        return dict(n=int(len(g)), found=int(g.fov_found.sum()), rate=float(g.fov_found.mean()))

    rate = float(d.fov_found.mean())
    report = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        n_frames=int(len(d)),
        found=dict(overall=found_rate(d), min_found_rate=min_found_rate, pass_overall=bool(rate >= min_found_rate),
                   by_domain={k: found_rate(g) for k, g in d.groupby("domain")},
                   by_partition={k: found_rate(g) for k, g in d.groupby("partition")}),
        method_counts=d.fov_method.value_counts().to_dict(),
        fov_r_by_domain={k: dict(r_px=_quant(g.fov_r[g.fov_found]), r_over_min_side=_quant(g.r_over_min_side[g.fov_found]),
                                 resid=_quant(g.fov_resid[g.fov_found])) for k, g in d.groupby("domain")},
        metrics_by_domain={k: {c: _quant(g[c]) for c in metrics} for k, g in d.groupby("domain")},
        metrics_by_partition={k: {c: _quant(g[c]) for c in metrics} for k, g in d.groupby("partition")},
        metrics_by_scenario={k: {c: _quant(g[c]) for c in metrics} for k, g in d[d.labeled].groupby("scenario")},
        nonfinite={c: dict(nan=int(d[c].isna().sum()), inf=int(np.isinf(d[c]).sum()),
                           nan_among_found=int(d[c][d.fov_found].isna().sum())) for c in metrics},
        runtime=dict(fov=_stage_runtime(parts_root, "fov"), metrics=_stage_runtime(parts_root, "metrics")),
    )
    cols = ["frame_id", "partition", "domain", "patient_id", "fov_method", "fov_r", "fov_resid", "image_path"]
    report["worst_residual_frames"] = worst_residual(m, f, 20)[cols].to_dict("records")
    report["not_found_frames"] = d[~d.fov_found][cols].head(50).to_dict("records")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(_clean(report), indent=1))
    _print_summary(report, log)
    return report


def _print_summary(rep, log):
    fnd = rep["found"]
    log(f"G1 FOV found: {fnd['overall']['found']}/{fnd['overall']['n']} = {100 * fnd['overall']['rate']:.2f}% "
        f"(target >= {100 * fnd['min_found_rate']:.0f}%: {'PASS' if fnd['pass_overall'] else 'FAIL'}); methods {rep['method_counts']}")
    for key in ("by_domain", "by_partition"):
        log("  " + "  ".join(f"{k}: {v['found']}/{v['n']} ({100 * v['rate']:.2f}%)" for k, v in fnd[key].items()))
    log("fov radius (px) min / p1 / p50 / p99 / max, and r over short side, by domain")
    for k, v in rep["fov_r_by_domain"].items():
        r, s = v["r_px"], v["r_over_min_side"]
        log(f"  {k}: {r['min']:.1f} / {r['p1']:.1f} / {r['p50']:.1f} / {r['p99']:.1f} / {r['max']:.1f}   "
            f"{s['min']:.3f} / {s['p50']:.3f} / {s['max']:.3f}   resid p50 {v['resid']['p50']:.4f} max {v['resid']['max']:.4f}")
    log("metric median [p1, p99] by domain")
    doms = list(rep["metrics_by_domain"])
    log("  " + f"{'metric':16s}" + "".join(f"{k:>28s}" for k in doms))
    for c in optics_lib.METRICS:
        cells = []
        for k in doms:
            q = rep["metrics_by_domain"][k][c]
            cells.append(f"{q['p50']:.3g} [{q['p1']:.3g}, {q['p99']:.3g}]" if q["p50"] is not None else "n/a")
        log("  " + f"{c:16s}" + "".join(f"{x:>28s}" for x in cells))
    nf = rep["nonfinite"]
    log(f"non-finite: NaN {sum(v['nan'] for v in nf.values())}, inf {sum(v['inf'] for v in nf.values())}, "
        f"NaN on found frames {sum(v['nan_among_found'] for v in nf.values())}")
    log(f"runtime: {rep['runtime']}")


# ---------------------------------------------------------------- overlays

def _pick(rng, pool, k):
    return pool.iloc[np.sort(rng.choice(len(pool), size=min(k, len(pool)), replace=False))] if len(pool) else pool


def select_overlay_frames(manifest, fov, seed, per_domain=8, n_stream=6, n_worst=6, n_notfound=12):
    """Seeded selection: per_domain labeled frames per domain (half Normal, half Difficult where available),
    n_stream stream frames, the n_worst found frames with the largest circle-fit residual and up to n_notfound
    frames without a FOV (labeled ones first)."""
    rng = np.random.default_rng(seed)
    lab = manifest[manifest.labeled]
    groups = {}
    for dom in sorted(lab.domain.unique()):
        g = lab[lab.domain == dom]
        half = per_domain // 2
        norm, diff = g[g.scenario == "Normal"], g[g.scenario == "Difficult"]
        take = pd.concat([_pick(rng, norm, half), _pick(rng, diff, half)])
        short = per_domain - len(take)
        if short > 0:
            rest = g[~g.frame_id.isin(take.frame_id)]
            take = pd.concat([take, _pick(rng, rest, short)])
        groups[dom] = take
    groups["stream"] = _pick(rng, manifest[~manifest.labeled], n_stream)
    groups["worst"] = worst_residual(manifest, fov, n_worst)[manifest.columns]
    nf = manifest[manifest.frame_id.isin(fov.frame_id[~fov.fov_found])]
    groups["notfound"] = pd.concat([nf[nf.labeled], _pick(rng, nf[~nf.labeled], n_notfound - int(nf.labeled.sum()))]).head(n_notfound)
    return groups


def _sub(x, scale):
    return (x + 0.5) * scale - 0.5      # pixel-centre coordinates -> resized image


def _tile(row, f, with_mask=True):
    img = read_image(abspath(row.image_path))
    h, w = img.shape[:2]
    s = min(TILE_W / w, TILE_H / h)
    tw, th = int(round(w * s)), int(round(h * s))
    small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    if with_mask and isinstance(row.mask_path, str):
        cnts, _ = cv2.findContours(read_mask(abspath(row.mask_path)).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = np.round(_sub(c.reshape(-1, 2).astype(np.float64), s)).astype(np.int32)
            cv2.polylines(small, [pts], True, GREEN, 1, cv2.LINE_AA)
    cx, cy = int(round(_sub(f.fov_cx, s))), int(round(_sub(f.fov_cy, s)))
    colour = YELLOW if f.fov_found else RED        # a red circle is the fallback geometry, not a fit
    cv2.circle(small, (cx, cy), int(round(f.fov_r * s)), colour, 1, cv2.LINE_AA)
    cv2.drawMarker(small, (cx, cy), colour, cv2.MARKER_CROSS, 10, 1)
    tile = np.zeros((TILE_H + LABEL_H, TILE_W, 3), np.uint8)
    y0, x0 = (TILE_H - th) // 2, (TILE_W - tw) // 2
    tile[y0:y0 + th, x0:x0 + tw] = small
    scen = row.scenario if isinstance(row.scenario, str) else "stream"
    inst = row.instrument if isinstance(row.instrument, str) else ""
    cv2.putText(tile, f"{row.frame_id}  {row.domain}  {scen} {inst}", (6, TILE_H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(tile, f"r={f.fov_r:.1f}px  {f.fov_method}  resid={f.fov_resid:.4f}  found={bool(f.fov_found)}",
                (6, TILE_H + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return tile


def render_overlays(manifest_path, fov_path, out_dir, seed, per_domain=8, n_stream=6, n_worst=6, n_notfound=12, log=print):
    """Montages of frames with the FOV circle (yellow; red when not found) and, where labeled, the ground-truth
    contour (green)."""
    m = pd.read_parquet(manifest_path)
    f = pd.read_parquet(fov_path).set_index("frame_id")
    groups = select_overlay_frames(m, f.reset_index(), seed, per_domain, n_stream, n_worst, n_notfound)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, selection = [], {}
    for name, g in groups.items():
        if not len(g):
            continue
        tiles = [_tile(r, f.loc[r.frame_id]) for r in g.itertuples()]
        rows = math.ceil(len(tiles) / MONTAGE_COLS)
        blank = np.zeros_like(tiles[0])
        tiles += [blank] * (rows * MONTAGE_COLS - len(tiles))
        grid = np.vstack([np.hstack(tiles[i * MONTAGE_COLS:(i + 1) * MONTAGE_COLS]) for i in range(rows)])
        path = out_dir / f"overlay_montage_{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        written.append(path)
        selection[name] = g.frame_id.tolist()
    (out_dir / "overlay_selection.json").write_text(json.dumps(selection, indent=1))
    log(f"overlays: wrote {len(written)} montages to {out_dir}")
    return written


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["fov", "metrics", "g1", "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=250, help="frames per task and per part file")
    args = ap.parse_args(argv)
    cfg = load_config()
    p = cfg["paths"]
    derived, logs = ROOT / p["derived"], ROOT / p["logs"]
    manifest, fov_path, optics_path = derived / "manifest.parquet", derived / "fov.parquet", derived / "optics.parquet"
    parts = logs / "optics_parts"
    stages = ["fov", "metrics", "g1"] if args.stage == "all" else [args.stage]
    if "fov" in stages:
        run_fov(manifest, fov_path, parts, args.workers, args.chunk, _make_log(logs / "optics_fov.log"))
    if "metrics" in stages:
        run_metrics(manifest, fov_path, optics_path, parts, args.workers, args.chunk, _make_log(logs / "optics_metrics.log"))
    if "g1" in stages:
        log = _make_log(logs / "optics_g1.log")
        g1_report(manifest, fov_path, optics_path, parts, logs / "g1_report.json", cfg["fov"]["min_found_rate"], log)
        render_overlays(manifest, fov_path, ROOT / p["v1"] / "work" / "qc" / "g1", cfg["seeds"]["global"], log=log)


if __name__ == "__main__":
    main()
