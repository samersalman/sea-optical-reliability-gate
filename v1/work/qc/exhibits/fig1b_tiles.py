"""Pixel work for Figure 1B (R draws the figure; this only reads images and masks).

Input  : a candidate CSV from R/10_fig1.R (columns cell, rank, frame_id, image_path, mask_path, fov_cx, fov_cy, fov_r),
         candidates ranked per cell by closeness to the cell median.
Output : in OUT_DIR
  fig1b_candidates.csv  every candidate that was inspected, with its ground-truth area share of the FOV disk
  fig1b_chosen.csv      per cell the first candidate whose instrument covers min_area..max_area of the FOV disk, with the
                        FOV circle in tile pixels (tile_cx, tile_cy, tile_r)
  tile_<frame_id>.png   the FOV-square crop resized to tile_px
  contours_<frame_id>.csv  ground-truth outline(s) in tile pixels (x right, y down), clipped to the tile square: part, order, x, y

Run: v1/.venv/bin/python v1/work/qc/exhibits/fig1b_tiles.py <candidates.csv> <out_dir> [--tile-px 600 --min-area 0.04 --max-area 0.40]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PY = Path(__file__).resolve().parents[3] / "python"
sys.path.insert(0, str(PY))
import common  # noqa: E402
import fov  # noqa: E402


def area_share(mask, cx, cy, r):
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return float((mask & disk).sum() / disk.sum())


def clip_segment(p0, p1, size):
    """Liang-Barsky: the part of segment p0-p1 inside the square [0, size]^2 as (start, end), or None. Endpoints that
    are already inside are returned unchanged."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, size - x0), (-dy, y0), (dy, size - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    start = p0 if t0 == 0.0 else (x0 + t0 * dx, y0 + t0 * dy)
    end = p1 if t1 == 1.0 else (x0 + t1 * dx, y0 + t1 * dy)
    return start, end


def clip_ring(pts, size):
    """Parts of a closed ring (first point repeated at the end) that lie inside [0, size]^2, as open polylines. A ring
    that never leaves the square comes back whole and still closed; one that does is cut where it crosses the edge."""
    parts, cur = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = clip_segment(tuple(a), tuple(b), size)
        if seg is None:
            if cur:
                parts.append(cur)
            cur = []
            continue
        if cur and cur[-1] == seg[0]:
            cur.append(seg[1])
        else:
            if cur:
                parts.append(cur)
            cur = [seg[0], seg[1]]
        if seg[1] != tuple(b):                    # left the square inside this segment
            parts.append(cur)
            cur = []
    if cur:
        parts.append(cur)
    if len(parts) > 1 and parts[-1][-1] == parts[0][0]:     # the cut fell mid-ring: rejoin the ends
        parts[0] = parts.pop()[:-1] + parts[0]
    return [pt for pt in parts if len(pt) > 1]


def tile_contours(mask, x0, y0, side, tile_px):
    """Outer contours of the native-resolution mask in tile pixel edge coordinates, lightly simplified and clipped to
    the tile square (an instrument can extend past the crop: the inscribed-circle crop of a 720 x 576 frame is narrower
    than the frame). Rows: part, order, x, y; every clipped piece is its own part."""
    scale = tile_px / side
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rows = []
    part = 0
    for c in cnts:
        if cv2.contourArea(c) < 0.0005 * mask.size:   # specks below 0.05% of the frame are not drawn
            continue
        c = cv2.approxPolyDP(c, 0.8, True)[:, 0, :].astype(float)
        pts = np.vstack([c, c[:1]])                       # close the ring
        ring = [((x - x0 + 0.5) * scale, (y - y0 + 0.5) * scale) for x, y in pts]
        for piece in clip_ring(ring, tile_px):
            for k, (x, y) in enumerate(piece):
                rows.append((part, k, x, y))
            part += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("out_dir")
    ap.add_argument("--tile-px", type=int, default=600)
    ap.add_argument("--min-area", type=float, default=0.04)
    ap.add_argument("--max-area", type=float, default=0.40)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(a.candidates, newline="") as fh:
        cand = list(csv.DictReader(fh))
    cells = list(dict.fromkeys(r["cell"] for r in cand))
    seen, chosen = [], []
    for cell in cells:
        rows = sorted((r for r in cand if r["cell"] == cell), key=lambda r: int(r["rank"]))
        for r in rows:
            cx, cy, rad = float(r["fov_cx"]), float(r["fov_cy"]), float(r["fov_r"])
            mask = common.read_mask(common.abspath(r["mask_path"]))
            share = area_share(mask, cx, cy, rad)
            seen.append({"cell": cell, "rank": r["rank"], "frame_id": r["frame_id"], "gt_area_share": round(share, 4)})
            if not (a.min_area <= share <= a.max_area):
                continue
            img = common.read_image(common.abspath(r["image_path"]))
            x0, y0, side = fov.crop_box(cx, cy, rad)
            tile = fov.crop_square(img, cx, cy, rad, a.tile_px, "area")
            cv2.imwrite(str(out / f"tile_{r['frame_id']}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            with open(out / f"contours_{r['frame_id']}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["part", "order", "x", "y"])
                w.writerows(tile_contours(mask, x0, y0, side, a.tile_px))
            scale = a.tile_px / side
            chosen.append({"cell": cell, "rank": r["rank"], "frame_id": r["frame_id"], "gt_area_share": round(share, 4),
                           "tile_cx": (cx - x0 + 0.5) * scale, "tile_cy": (cy - y0 + 0.5) * scale, "tile_r": rad * scale,
                           "tile_px": a.tile_px})
            break
        else:
            raise SystemExit(f"no candidate in cell {cell} has an instrument share in [{a.min_area}, {a.max_area}]")

    for name, rows in (("fig1b_candidates.csv", seen), ("fig1b_chosen.csv", chosen)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"chosen {len(chosen)} tiles from {len(seen)} inspected candidates")


if __name__ == "__main__":
    main()
