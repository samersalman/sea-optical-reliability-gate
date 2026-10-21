"""Field-of-view (FOV) detection and canonical crop for endoscopic frames.

The endoscope image is a bright disk on a dark surround, with a small notch at
the bottom of the circle. Detection runs in two steps:

1. coarse: threshold a downsampled copy just above the surround level, take the
   outer contour of the largest blob (holes such as dark instruments are
   ignored) and fit a circle to the contour with RANSAC, so the notch and any
   arc cut off by the image border count as outliers;
2. refine: sample intensity along rays at native resolution in a narrow band
   around the coarse radius, take the steepest bright-to-dark step on each ray
   (sub-pixel) and fit again with RANSAC, repeating from the new circle while
   it still moves (a dark instrument that merges with the surround ruins the
   coarse fit but not the ray fit). The last plausible fit wins.

The refined fit is an axis-aligned ellipse when most of the rim is visible,
reported as its inscribed circle (centre = ellipse centre, r = shorter
semi-axis). Center 2 (720 x 576) stores the FOV about 4% taller than wide, and
a plain circle fit there wobbles in cy by up to 5% of r from frame to frame.
Circular FOVs give the same answer as a circle fit to 0.1%.

Geometry is in native pixel-centre coordinates (pixel (i, j) has its centre at
x = i, y = j). A frame is "found" only if the fit is plausible: radius, centre,
inlier share, angular coverage and residual all within bounds. When not found,
cx, cy, r hold a fallback (image centre, half the short side) so callers never
crash; they must still read `found`.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis_plan.yaml"
with open(_CFG_PATH) as fh:
    _FOV_CFG = yaml.safe_load(fh)["fov"]

CANON_PX = int(_FOV_CFG["canonical_diameter_px"])
ERODE_FRAC = float(_FOV_CFG["erode_frac"])

# Detector internals. Not in analysis_plan.yaml; chosen from the real sample
# (D1/D2/D3 and stream frames) and checked by the unit tests.
COARSE_PX = 256            # long side of the downsampled copy
THR_MIN_STEP = 6           # gray levels above the surround for the coarse threshold
THR_FRAC = 0.2             # or this share of (p95 - surround), whichever is larger
MIN_CONTRAST = 12          # p95 - surround below this means no visible FOV
N_RAYS = 144
BAND_FRAC = 0.12           # ray band half-width as a share of the coarse radius
EDGE_SIGMA_PX = 1.5        # smoothing along each ray (native px)
EDGE_MIN_DROP = 4.0        # min gray-level drop per px for a ray to count
RANSAC_ITERS = 200
TOL_FRAC = 0.012           # inlier tolerance as a share of r
TOL_MIN_PX = 1.5
R_FRAC_MIN, R_FRAC_MAX = 0.25, 0.75   # r / min(h, w): plausible FOV size
INLIER_MIN = 0.5           # share of candidate points that must be inliers
COVER_MIN = 0.3            # share of 10-degree bins holding an inlier
RESID_MAX = 0.02           # RMS radial residual / r
LOOSE_FRAC = 0.03          # first-pass inlier tolerance before the ellipse trim
ELL_COVER_MIN = 0.7        # an ellipse needs at least this share of the circle visible
ELL_RATIO_MAX = 1.10       # longer / shorter semi-axis; beyond this keep the circle
REFINE_ROUNDS = 3          # a poor coarse fit (dark instrument merging with the surround) needs a second pass
REFINE_CONVERGED = 0.01    # stop when centre + radius move less than this share of r


def _as_uint8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.dtype.kind == "f":
        return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _kasa(pts):
    """Algebraic least-squares circle through pts (N, 2); returns cx, cy, r."""
    mu = pts.mean(0)
    p = pts - mu
    a = np.c_[p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    return cx + mu[0], cy + mu[1], r


def _ransac_circle(pts, tol, r_lo, r_hi, rng):
    """Robust circle fit. Returns (cx, cy, r, inlier_mask) or None."""
    n = len(pts)
    if n < 20:
        return None
    idx = rng.integers(0, n, size=(RANSAC_ITERS, 3))
    a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    good = np.abs(d) > 1e-9
    d = np.where(good, d, 1.0)
    a2, b2, c2 = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
    ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
    uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    r = np.hypot(a[:, 0] - ux, a[:, 1] - uy)
    good &= (r >= r_lo) & (r <= r_hi)
    if not good.any():
        return None
    dist = np.abs(np.hypot(pts[None, :, 0] - ux[:, None], pts[None, :, 1] - uy[:, None]) - r[:, None])
    score = np.where(good, (dist < tol).sum(1), -1)
    k = int(np.argmax(score))
    cx, cy, rr = ux[k], uy[k], r[k]
    inl = dist[k] < tol
    for _ in range(3):
        if inl.sum() < 10:
            return None
        cx, cy, rr = _kasa(pts[inl])
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rr) < tol
    return cx, cy, rr, inl


def _score(pts, cx, cy, r, inl, n_candidates):
    """Residual (RMS / r), inlier share and angular coverage of a fit."""
    p = pts[inl]
    resid = float(np.sqrt(np.mean((np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r) ** 2)) / r)
    ang = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
    bins = np.unique(((ang + np.pi) / (2 * np.pi) * 36).astype(int) % 36)
    return resid, float(inl.sum() / n_candidates), len(bins) / 36.0


def _ellipse(pts, inl, tol, n_rays):
    """Axis-aligned ellipse by trimmed least squares, reported as its inscribed circle.

    Some devices (Center 2, 720 x 576) store the FOV about 4% taller than wide,
    so a circle fit wobbles in cy by up to 5% of r between frames. The ellipse
    centre does not; r = the shorter semi-axis keeps the disk inside the FOV.
    Returns the fit dict or None when the arc is too short or too elongated.
    """
    for _ in range(4):
        if inl.sum() < 20:
            return None
        mu = pts[inl].mean(0)
        sc = float(np.ptp(pts[inl][:, 0]) + np.ptp(pts[inl][:, 1])) / 2 or 1.0
        q = (pts[inl] - mu) / sc
        p1, p2, p3, p4 = np.linalg.lstsq(np.c_[q[:, 0] ** 2, q[:, 1] ** 2, q[:, 0], q[:, 1]], np.ones(len(q)), rcond=None)[0]
        if p1 <= 0 or p2 <= 0:
            return None
        ux, uy = -p3 / (2 * p1), -p4 / (2 * p2)
        k = 1 + p1 * ux ** 2 + p2 * uy ** 2
        a, b = sc * np.sqrt(k / p1), sc * np.sqrt(k / p2)
        cx, cy = mu[0] + sc * ux, mu[1] + sc * uy
        dev = (np.hypot((pts[:, 0] - cx) / a, (pts[:, 1] - cy) / b) - 1.0) * np.sqrt(a * b)
        inl = np.abs(dev) < tol
    if inl.sum() < 20 or max(a, b) / min(a, b) > ELL_RATIO_MAX:
        return None
    ang = np.arctan2(pts[inl][:, 1] - cy, pts[inl][:, 0] - cx)
    cover = len(np.unique(((ang + np.pi) / (2 * np.pi) * 36).astype(int) % 36)) / 36.0
    if cover < ELL_COVER_MIN:
        return None
    r = float(min(a, b))
    return dict(cx=cx, cy=cy, r=r, resid=float(np.sqrt(np.mean(dev[inl] ** 2)) / r),
                share=float(inl.sum() / max(n_rays, 1)), cover=cover)


def _coarse(img, rng):
    """Circle from the thresholded contour of a downsampled copy, in native px."""
    h, w = img.shape[:2]
    k = max(1, max(h, w) // (2 * COARSE_PX))      # stride first: INTER_AREA at 1080p is slow
    sub = img[::k, ::k]
    f = COARSE_PX / max(sub.shape[:2])
    sh, sw = max(int(round(sub.shape[0] * f)), 8), max(int(round(sub.shape[1] * f)), 8)
    ax, ay = k * sub.shape[1] / sw, k * sub.shape[0] / sh   # native px per coarse px
    gray = cv2.resize(cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
    q = max(int(0.08 * min(sh, sw)), 2)
    corners = np.concatenate([gray[:q, :q].ravel(), gray[:q, -q:].ravel(), gray[-q:, :q].ravel(), gray[-q:, -q:].ravel()])
    surround = float(np.median(corners))
    hi = float(np.percentile(gray, 95))
    if hi - surround < MIN_CONTRAST:
        return None
    thr = surround + max(THR_MIN_STEP, THR_FRAC * (hi - surround))
    mask = (gray > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    on_border = (pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= sw - 2) | (pts[:, 1] >= sh - 2)
    pts = pts[~on_border]
    # coarse pixel centres -> native pixel-centre coordinates
    pts = np.c_[ax * (pts[:, 0] + 0.5) - 0.5 * k, ay * (pts[:, 1] + 0.5) - 0.5 * k]
    m = min(h, w)
    tol = max(TOL_MIN_PX, 1.5 * max(ax, ay))
    fit = _ransac_circle(pts, tol, R_FRAC_MIN * m * 0.9, R_FRAC_MAX * m * 1.1, rng)
    if fit is None:
        return None
    cx, cy, r, inl = fit
    r += 0.25 * (ax + ay)              # boundary pixels sit half a coarse pixel inside the edge
    resid, share, cover = _score(pts, cx, cy, r, inl, len(pts))
    return dict(cx=cx, cy=cy, r=r, resid=resid, share=share, cover=cover)


def _refine(img, cx, cy, r, rng):
    """Re-fit the circle on sub-pixel edges sampled along rays at native resolution."""
    h, w = img.shape[:2]
    band = BAND_FRAC * r
    rr = np.arange(r - band, r + band + 1e-6, 1.0)
    ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    xs = cx + np.cos(ang)[:, None] * rr[None]
    ys = cy + np.sin(ang)[:, None] * rr[None]
    inside = ((xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)).all(1)
    if inside.sum() < 20:
        return None
    x0, x1 = int(max(np.floor(xs.min()) - 2, 0)), int(min(np.ceil(xs.max()) + 3, w))
    y0, y1 = int(max(np.floor(ys.min()) - 2, 0)), int(min(np.ceil(ys.max()) + 3, h))
    gray = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    prof = cv2.remap(gray, (xs - x0).astype(np.float32), (ys - y0).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=EDGE_SIGMA_PX, sigmaY=0.01)
    d = np.gradient(prof, axis=1)
    k = np.argmin(d, axis=1)
    rows = np.arange(len(k))
    dmin = d[rows, k]
    valid = inside & (dmin <= -EDGE_MIN_DROP) & (k > 0) & (k < len(rr) - 1)
    if valid.sum() < 20:
        return None
    km1, k0, kp1 = d[rows, np.clip(k - 1, 0, None)], dmin, d[rows, np.clip(k + 1, None, len(rr) - 1)]
    denom = km1 - 2 * k0 + kp1
    off = np.divide(0.5 * (km1 - kp1), denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-9)
    rad = rr[k] + np.clip(off, -1, 1)
    pts = np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)][valid]
    tol = max(TOL_MIN_PX, TOL_FRAC * r)
    n_valid = len(pts)                  # rays with a clear edge; weak-edge rays are judged by coverage instead
    r_lo, r_hi = r * (1 - BAND_FRAC), r * (1 + BAND_FRAC)
    loose = _ransac_circle(pts, max(TOL_MIN_PX, LOOSE_FRAC * r), r_lo, r_hi, rng)
    if loose is None:
        return None
    ell = _ellipse(pts, loose[3], tol, n_valid)
    if ell is not None:
        return ell
    fit = _ransac_circle(pts, tol, r_lo, r_hi, rng)
    if fit is None:
        return None
    ncx, ncy, nr, inl = fit
    resid, _, cover = _score(pts, ncx, ncy, nr, inl, len(pts))
    return dict(cx=ncx, cy=ncy, r=nr, resid=resid, share=float(inl.sum() / n_valid), cover=cover)


def _plausible(fit, h, w):
    m = min(h, w)
    return (fit is not None
            and R_FRAC_MIN * m <= fit["r"] <= R_FRAC_MAX * m
            and 0 <= fit["cx"] < w and 0 <= fit["cy"] < h
            and fit["share"] >= INLIER_MIN and fit["cover"] >= COVER_MIN
            and fit["resid"] <= RESID_MAX)


def detect_fov(img_rgb):
    """Fit the circular endoscope FOV.

    img_rgb: (H, W, 3) uint8 RGB (float arrays in 0-1 are accepted).
    Returns dict(found, cx, cy, r, method, resid): centre and radius in native
    pixels (inscribed circle, see module docstring); method is "ray"
    (sub-pixel refined), "mask" (coarse only) or "fallback" (not found); resid
    is the RMS radial residual of the inlier boundary points divided by r
    (NaN if no fit was attempted).
    """
    img = _as_uint8(img_rgb)
    h, w = img.shape[:2]
    rng = np.random.default_rng(0)
    coarse = _coarse(img, rng)
    out = dict(found=False, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, r=0.5 * min(h, w), method="fallback", resid=float("nan"))
    if coarse is None:
        return out
    cur, best, method = coarse, (coarse if _plausible(coarse, h, w) else None), "mask"
    for _ in range(REFINE_ROUNDS):
        ref = _refine(img, cur["cx"], cur["cy"], cur["r"], rng)
        if ref is None:
            break
        moved = (np.hypot(ref["cx"] - cur["cx"], ref["cy"] - cur["cy"]) + abs(ref["r"] - cur["r"])) / cur["r"]
        cur = ref
        if _plausible(ref, h, w):
            best, method = ref, "ray"
        if moved < REFINE_CONVERGED:
            break
    fit = best if best is not None else cur
    out["resid"] = fit["resid"]
    if best is not None:
        out.update(found=True, cx=float(fit["cx"]), cy=float(fit["cy"]), r=float(fit["r"]), method=method)
    return out


def crop_box(cx, cy, r):
    """Integer square crop (x0, y0, side) centred on (cx, cy) with side = round(2r)."""
    side = int(round(2 * r))
    return int(round(cx - (side - 1) / 2.0)), int(round(cy - (side - 1) / 2.0)), side


_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def _crop_raw(arr, x0, y0, side):
    h, w = arr.shape[:2]
    out = np.zeros((side, side) + arr.shape[2:], dtype=arr.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = arr[ya:yb, xa:xb]
    return out


def _resize(a, out_size, interp):
    side = a.shape[0]
    if side == out_size:
        return a
    dtype = a.dtype
    if dtype == bool:
        interp = "nearest"
        a = a.astype(np.uint8)
    elif dtype.kind in "iu" and dtype.itemsize == 8:
        a = a.astype(np.int32)
    elif dtype.kind == "f" and dtype.itemsize == 2:
        a = a.astype(np.float32)
    if interp == "auto":
        interp = "area" if out_size < side else "linear"
    out = cv2.resize(a, (out_size, out_size), interpolation=_INTERP[interp])
    if out.ndim < a.ndim:                      # cv2 drops a trailing singleton channel
        out = out[..., None]
    return out.astype(dtype) if out.dtype != dtype else out


def crop_square(arr, cx, cy, r, out_size, interp="auto"):
    """Square crop of side round(2r) centred on (cx, cy), zero-padded outside the image, resized to out_size.

    Works for RGB images, single-channel arrays and masks. interp is "auto"
    (area when shrinking, linear when enlarging), "nearest" (use for masks),
    "linear", "cubic" or "area". bool masks are always resized with nearest.
    The dtype is preserved.
    """
    x0, y0, side = crop_box(cx, cy, r)
    return _resize(_crop_raw(np.asarray(arr), x0, y0, side), int(out_size), interp)


def fov_disk_mask(size=CANON_PX, erode_frac=ERODE_FRAC):
    """Boolean (size, size) disk filling the canvas, radius shrunk by erode_frac of the radius."""
    c = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    rad = size / 2.0 * (1.0 - erode_frac)
    return (xx - c) ** 2 + (yy - c) ** 2 <= rad ** 2


def _geom(fov):
    if "cx" in fov:
        return fov["cx"], fov["cy"], fov["r"]
    return fov["fov_cx"], fov["fov_cy"], fov["fov_r"]


def canonical_canvas(img_rgb, fov, size=CANON_PX, erode_frac=ERODE_FRAC):
    """FOV-centred square canvas for the optics metrics.

    fov: dict with cx, cy, r (the output of detect_fov) or fov_cx, fov_cy, fov_r.
    Returns (canvas, disk): float32 (size, size, 3) in 0-1 with zeros where the
    crop leaves the image, and the eroded FOV disk mask (bool).
    """
    img = np.asarray(img_rgb)
    x0, y0, side = crop_box(*_geom(fov))
    crop = _crop_raw(img, x0, y0, side)
    if crop.dtype == np.uint8:
        crop = crop.astype(np.float32) / 255.0
    else:
        crop = crop.astype(np.float32)
    return np.ascontiguousarray(_resize(crop, size, "auto")), fov_disk_mask(size, erode_frac)
