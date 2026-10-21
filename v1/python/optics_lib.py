"""Optical quality metrics for one FOV-normalised endoscopic frame.

Input is the 512 x 512 float RGB canvas (gamma-encoded, 0-1) from
fov.canonical_canvas and the eroded FOV disk. Every metric is computed over
M = disk (and bg_mask when given). The ground-truth instrument mask is never an
input; bg_mask is the complement of the *predicted* mask (sensitivity set).

Metric definitions (Table 2). Y is Rec.709 luma 0.2126 R + 0.7152 G + 0.0722 B
of the gamma-encoded values, V = max(R, G, B), S = (V - min(R, G, B)) / V.

| metric          | definition                                                         | physical reading        |
|-----------------|--------------------------------------------------------------------|-------------------------|
| lum_mean        | mean of Y over M                                                   | exposure                |
| contrast_rms    | standard deviation of Y over M                                     | global contrast         |
| contrast_local  | mean over 16 px blocks of the block std of Y (blocks with >= half  | local texture / detail  |
|                 | their pixels in M)                                                 |                         |
| sharp_tenengrad | mean of gx^2 + gy^2, Sobel 3x3 on Y scaled to luma per pixel,      | focus / motion blur     |
|                 | over pixels whose 3x3 neighbourhood lies inside M                  | (noise raises it too)   |
| under_frac      | share of M with Y < 0.10                                           | under-exposure          |
| sat_frac        | share of M with any channel >= 0.98                                | clipping / over-exposure|
| spec_frac       | share of M with Y >= T_Y and S <= T_S (below)                      | glare / specular        |
| haze_dcp        | mean dark channel, min over 15 px patch and RGB, patch limited     | scatter, smoke, turbid  |
|                 | to M                                                               | irrigation              |
| red_dom         | mean of R / (R + G + B) over pixels of M with R + G + B >= 0.06    | blood absorption        |
| illum_slope     | OLS slope of Y on normalised radius (0 canvas centre, 1 disk edge) | vignetting / uneven     |
|                 | over M; negative = darker towards the rim                          | illumination            |
| lum_entropy     | Shannon entropy (bits) of Y over M, 64 equal bins on [0, 1]        | tonal range            |

Specular threshold. Tissue brightness varies by an order of magnitude between
frames, so a fixed cut-off would flag white bone as glare in bright frames and
miss glare in dark ones. With med and sd the median and the MAD-based robust
SD of Y over M:
    T_Y = max( clip(med + 3 sd, 0.80, 0.97), med + 0.10 )
    T_S = clip(0.5 * median(S over M), 0.10, 0.25)
A glare pixel must be bright in absolute terms (floor 0.80), stand out from the
frame's own tissue brightness (3 robust SD, at least 0.10 above the median, so
a uniformly bright or clipped frame is over-exposure, not glare) and be
desaturated relative to the frame's own colour (half the median saturation, so
coloured tissue and blood-red highlights are not counted).
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis_plan.yaml"
with open(_CFG_PATH) as fh:
    METRICS = tuple(yaml.safe_load(fh)["optical_metrics"])

BLOCK = 16                 # local-contrast block (px on the 512 canvas)
BLOCK_MIN_FRAC = 0.5       # share of a block that must lie in M
DCP_PATCH = 15
UNDER_Y = 0.10
SAT_LEVEL = 0.98
N_BINS = 64
CHROMA_MIN_SUM = 0.06      # R + G + B below this has no usable chromaticity
MIN_PIXELS = 256           # smaller M returns NaN for every metric
SPEC_Y_FLOOR, SPEC_Y_CAP, SPEC_K, SPEC_EXCESS = 0.80, 0.97, 3.0, 0.10
SPEC_S_FRAC, SPEC_S_MIN, SPEC_S_MAX = 0.5, 0.10, 0.25
MAD_TO_SD = 1.4826

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_NAN = {name: float("nan") for name in METRICS}


@lru_cache(maxsize=4)
def _radius_grid(size):
    c = (size - 1) / 2.0
    yy, xx = np.indices((size, size), dtype=np.float32)
    return np.hypot(xx - c, yy - c)


def _block_contrast(y, m):
    """Mean of the per-block std of y over blocks with at least half their pixels in m."""
    h, w = (y.shape[0] // BLOCK) * BLOCK, (y.shape[1] // BLOCK) * BLOCK
    yb = y[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    mb = m[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    n = mb.sum(axis=(1, 3))
    ok = n >= BLOCK_MIN_FRAC * BLOCK * BLOCK
    if not ok.any():
        return float("nan")
    mean = (yb * mb).sum(axis=(1, 3)) / np.maximum(n, 1)
    dev = (yb - mean[:, None, :, None]) * mb
    std = np.sqrt((dev * dev).sum(axis=(1, 3)) / np.maximum(n, 1))
    return float(std[ok].mean())


def _entropy_bits(yv):
    idx = np.clip((yv * N_BINS).astype(np.int32), 0, N_BINS - 1)
    p = np.bincount(idx, minlength=N_BINS).astype(np.float64)
    p = p[p > 0] / yv.size
    return float(-(p * np.log2(p)).sum())


def _spec_frac(y, s, m):
    yv = y[m]
    med = float(np.median(yv))
    sd = MAD_TO_SD * float(np.median(np.abs(yv - med)))
    t_y = max(min(max(med + SPEC_K * sd, SPEC_Y_FLOOR), SPEC_Y_CAP), med + SPEC_EXCESS)
    t_s = min(max(SPEC_S_FRAC * float(np.median(s[m])), SPEC_S_MIN), SPEC_S_MAX)
    return float(np.mean((yv >= t_y) & (s[m] <= t_s)))


def compute_metrics(rgb01, disk_mask, bg_mask=None):
    """The 11 optical metrics for one canvas, as a dict in the order of analysis_plan.yaml.

    rgb01: (H, W, 3) float, gamma-encoded 0-1 (H = W = 512 in the pipeline).
    disk_mask: bool (H, W), the eroded FOV disk.
    bg_mask: optional bool (H, W); metrics are computed over disk_mask & bg_mask.
    Returns all-NaN when fewer than MIN_PIXELS pixels remain.
    """
    rgb = np.ascontiguousarray(rgb01, dtype=np.float32)
    m = np.asarray(disk_mask, dtype=bool)
    if bg_mask is not None:
        m = m & np.asarray(bg_mask, dtype=bool)
    if int(m.sum()) < MIN_PIXELS:
        return dict(_NAN)

    y = rgb @ _LUMA
    yv = y[m].astype(np.float64)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    s = (mx - mn) / np.maximum(mx, 1e-6)
    out = {}

    out["lum_mean"] = float(yv.mean())
    out["contrast_rms"] = float(yv.std())
    out["contrast_local"] = _block_contrast(y, m)

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.0)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.0)
    inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out["sharp_tenengrad"] = float((gx * gx + gy * gy)[inner].mean()) if inner.any() else float("nan")

    out["under_frac"] = float(np.mean(yv < UNDER_Y))
    out["sat_frac"] = float(np.mean(mx[m] >= SAT_LEVEL))
    out["spec_frac"] = _spec_frac(y, s, m)

    # pixels outside M count as bright so the patch minimum only sees M
    dark = cv2.erode(np.where(m, mn, 1.0).astype(np.float32), np.ones((DCP_PATCH, DCP_PATCH), np.uint8))
    out["haze_dcp"] = float(dark[m].mean())

    tot = rgb.sum(axis=2)
    ok = m & (tot >= CHROMA_MIN_SUM)
    out["red_dom"] = float(np.mean(rgb[..., 0][ok] / tot[ok])) if ok.any() else float("nan")

    radius = _radius_grid(y.shape[0])
    edge = np.sqrt(np.asarray(disk_mask, dtype=bool).sum() / np.pi)
    rn = radius[m].astype(np.float64) / edge
    var = rn.var()
    out["illum_slope"] = float(np.mean((rn - rn.mean()) * (yv - yv.mean())) / var) if var > 0 else float("nan")

    out["lum_entropy"] = _entropy_bits(yv)

    assert set(out) == set(METRICS), "metric names drifted from analysis_plan.yaml"
    return {name: out[name] for name in METRICS}
