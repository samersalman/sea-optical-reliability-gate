"""Physically parameterised image degradations for the perturbation experiment.

Contract: apply(rgb, disk_mask, family, severity, rng) -> float32 HxWx3 in [0, 1].
Pixels change only where disk_mask is True; everywhere else the output is bit-identical to
the input. Severity 0 is the identity (a copy). Severities 1-5 step up a fixed, monotone
grid per family. CONDITIONS / list_conditions() enumerate (family, severity, param).

Conventions
- The array is the 512 px FOV canvas (analysis_plan.yaml: fov.canonical_diameter_px). Lengths
  (defocus sigma) are in pixels of that canvas; nothing is rescaled inside.
- Input values are display-referred sRGB. Every operator works in linear light (sRGB transfer
  function decoded, operated on, encoded again, clipped to [0, 1]) because exposure, scattering,
  absorption, PSF blur and photon noise are all linear in radiance. The camera's real tone curve
  is unknown; sRGB is the stated approximation.
- Only glare (blob layout) and noise (field of normals) consume rng. To get nested conditions
  for one frame (same blobs, same noise field, only the strength changes), build the rng from
  the same seed for every severity of that (frame, family).
- Operators read only pixels inside disk_mask (defocus uses normalised convolution), so content
  outside the disk (black surround, padding, colour cast) cannot leak in.

Families and parameters (severity 0 is the identity value)

family            param (unit)                    sev 1    2     3     4     5     model
defocus           psf_sigma (px, 512 canvas)      1.0      2.0   3.5   5.0   7.0   Gaussian PSF
haze              beta (optical depth)            0.04     0.10  0.20  0.36  0.55  I = J t + A (1 - t), t = exp(-beta)
exposure_under    gain (x linear radiance)        0.75     0.55  0.40  0.28  0.18  I = g J, g < 1 (-0.4 to -2.5 EV)
exposure_over     gain (x linear radiance)        1.3      1.7   2.3   3.2   4.5   I = min(g J, 1), g > 1 (+0.4 to +2.2 EV)
blood_absorption  s (path x conc., k_G = 1)       0.10     0.25  0.50  0.80  1.20  I_c = J_c exp(-k_c s)
glare             area_frac (of disk area)        0.005    0.015 0.03  0.06  0.10  saturated feathered blobs
noise             sigma_mid (linear units)        0.01     0.02  0.035 0.055 0.08  Poisson-Gaussian, var = a J + b

Severity-5 check on 120 random labeled frames (crude FOV crops, JPG and PNG, all centres).
Medians, baseline in brackets: mean luminance Y and sd of Y inside the disk, fraction of pixels
with any channel >= 0.98, fraction with Y < 0.10, R/(R+G+B).
- defocus: sharpness only; Y and Y sd barely move (0.40, 0.18 [0.40, 0.19]), fine detail is gone.
- haze: Y 0.69 [0.40], Y sd 0.061 [0.189]; no frame above Y 0.9.
- exposure_under: Y 0.17, under-exposed fraction 0.25 [0.02]; no frame below Y 0.06.
- exposure_over: Y 0.73, saturated fraction 0.42 [0.006]; 4 of 120 frames above Y 0.9.
- blood_absorption: Y 0.26, R/(R+G+B) 0.57 [0.40].
- glare: saturated fraction 0.074 [0.006].
- noise: under-exposed fraction 0.09 [0.02] (noise pushes shadows to zero before the clip).

Fixed constants and why severity 5 is clearly degraded but not black or white
- haze: A = (0.80, 0.80, 0.80) in linear light (pale grey-white veil, as for cautery smoke or turbid
  irrigation fluid under a white light source). t is uniform over the FOV (no depth map is
  available). Severity 5 keeps t = 0.58 of the scene signal, so contrast falls to about a third
  and the image turns milky; structure stays visible because the veil is A (1 - t), not white.
- exposure_under: gain 0.18 at severity 5 is -2.5 EV: dim, shadows crushed, tissue still visible.
  exposure_over: gain 4.5 is +2.2 EV; bright tissue clips, red first (as on a real sensor), shadows
  keep detail. Larger gains (6.5) turned the median frame nearly white and were rejected.
- blood_absorption: k = (0.05, 1.00, 1.30) for (R, G, B) per unit s. These are effective,
  approximate channel coefficients set from the shape of the oxyhaemoglobin extinction spectrum
  (about 20-fold lower at ~610 nm than at ~540 and ~450 nm). They are NOT calibrated to the SEA
  camera, whose spectral sensitivities are unknown. s is a dimensionless path-times-
  concentration scalar, uniform over the FOV. At severity 5 transmission is 0.94 (R), 0.30 (G),
  0.21 (B): a strongly red frame that is still lit.
- glare: 5 elliptical blobs (aspect 1-2.2, random orientation) at positions drawn inside the disk
  from rng; the half-opacity area sums to area_frac x disk area (before overlap and clipping by
  the disk). Cores are pure white, edges feather over +/-25% of the radius, so the saturated
  fraction realised is about 0.65 of area_frac. Blobs are stylised ellipses, not real specular
  patterns; severity 4-5 are the least realistic. Severity 5 covers about a tenth of the FOV.
- noise: var(J) = sigma_mid^2 (0.75 J / 0.18 + 0.25) in linear light: 75% of the variance at
  mid-grey (J = 0.18) is signal-dependent shot noise, 25% is signal-independent read noise,
  independent per pixel and channel, added before the display clip. Severity 5 is SNR of
  about 2 at mid-grey: heavy chroma grain, but the scene is readable.
- Everything is uniform over the disk (no spatially varying haze, bleeding or defocus).
"""
from __future__ import annotations

import cv2
import numpy as np

FAMILIES = ("defocus", "haze", "exposure_under", "exposure_over", "blood_absorption", "glare", "noise")
SEVERITIES = (0, 1, 2, 3, 4, 5)

# family -> (param_name, unit, values for severities 0..5)
_SPECS = {
    "defocus": ("psf_sigma", "px", (0.0, 1.0, 2.0, 3.5, 5.0, 7.0)),
    "haze": ("beta", "optical depth", (0.0, 0.04, 0.10, 0.20, 0.36, 0.55)),
    "exposure_under": ("gain", "x linear radiance", (1.0, 0.75, 0.55, 0.40, 0.28, 0.18)),
    "exposure_over": ("gain", "x linear radiance", (1.0, 1.3, 1.7, 2.3, 3.2, 4.5)),
    "blood_absorption": ("path_conc", "k_G = 1", (0.0, 0.10, 0.25, 0.50, 0.80, 1.20)),
    "glare": ("area_frac", "fraction of disk", (0.0, 0.005, 0.015, 0.03, 0.06, 0.10)),
    "noise": ("sigma_mid", "linear units at 0.18", (0.0, 0.01, 0.02, 0.035, 0.055, 0.08)),
}

CONDITIONS = [
    {"family": f, "severity": s, "param": _SPECS[f][2][s], "param_name": _SPECS[f][0], "unit": _SPECS[f][1]}
    for f in FAMILIES
    for s in SEVERITIES
]


def list_conditions(include_baseline=True):
    """Copy of CONDITIONS; include_baseline=False drops the severity-0 rows."""
    return [dict(c) for c in CONDITIONS if include_baseline or c["severity"] > 0]


_HAZE_LIGHT = np.array([0.80, 0.80, 0.80], np.float32)  # atmospheric light A, linear
_BLOOD_K = np.array([0.05, 1.00, 1.30], np.float32)  # effective absorption per unit s, (R, G, B)
_GLARE_BLOBS = 5
_GLARE_FEATHER = 0.25  # half-width of the edge ramp as a fraction of the blob radius
_GLARE_MARGIN = 0.15  # blob centres stay this far (x disk radius) from the disk edge
_NOISE_MID = 0.18
_NOISE_SHOT_SHARE = 0.75


def _to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def _defocus(lin, mask, sigma, rng):
    m = mask.astype(np.float32)
    kw = dict(sigmaX=sigma, borderType=cv2.BORDER_CONSTANT)
    num = cv2.GaussianBlur(lin * m[..., None], (0, 0), **kw)
    den = cv2.GaussianBlur(m, (0, 0), **kw)
    return num / np.maximum(den, 1e-6)[..., None]


def _haze(lin, mask, beta, rng):
    t = np.exp(-beta)
    return lin * t + _HAZE_LIGHT * (1.0 - t)


def _gain(lin, mask, g, rng):
    return lin * g


def _blood(lin, mask, s, rng):
    return lin * np.exp(-_BLOOD_K * s)


def _glare(lin, mask, frac, rng):
    h, w = mask.shape
    area = float(mask.sum())
    r_disk = np.sqrt(area / np.pi)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    cand = np.flatnonzero(dist > _GLARE_MARGIN * r_disk)
    if cand.size == 0:
        cand = np.flatnonzero(mask)
    # the layout draws are the same for every severity so the blobs nest as frac grows
    cy, cx = np.divmod(cand[rng.integers(cand.size, size=_GLARE_BLOBS)], w)
    share = rng.dirichlet(np.full(_GLARE_BLOBS, 3.0))
    aspect = rng.uniform(1.0, 2.2, _GLARE_BLOBS)
    theta = rng.uniform(0.0, np.pi, _GLARE_BLOBS)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = np.zeros((h, w), np.float32)
    for i in range(_GLARE_BLOBS):
        blob_area = frac * area * share[i]
        a = np.sqrt(blob_area * aspect[i] / np.pi)  # semi-major; alpha = 0.5 on the ellipse a x b
        b = blob_area / (np.pi * a)
        dx, dy = xx - cx[i], yy - cy[i]
        u = dx * np.cos(theta[i]) + dy * np.sin(theta[i])
        v = -dx * np.sin(theta[i]) + dy * np.cos(theta[i])
        rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        t = np.clip((rho - (1.0 - _GLARE_FEATHER)) / (2.0 * _GLARE_FEATHER), 0.0, 1.0)
        alpha = 1.0 - (1.0 - alpha) * (t * t * (3.0 - 2.0 * t))
    return lin * (1.0 - alpha[..., None]) + alpha[..., None]


def _noise(lin, mask, sigma_mid, rng):
    z = rng.standard_normal(lin.shape, dtype=np.float32)
    var = sigma_mid**2 * (_NOISE_SHOT_SHARE * lin / _NOISE_MID + (1.0 - _NOISE_SHOT_SHARE))
    return lin + np.sqrt(var) * z


_OPS = {
    "defocus": _defocus,
    "haze": _haze,
    "exposure_under": _gain,
    "exposure_over": _gain,
    "blood_absorption": _blood,
    "glare": _glare,
    "noise": _noise,
}


def apply(rgb_float01, disk_mask, family, severity, rng):
    """Degrade rgb_float01 (HxWx3, 0-1) inside disk_mask (HxW bool); see module docstring."""
    if family not in _OPS:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    rgb = np.asarray(rgb_float01, dtype=np.float32)
    mask = np.asarray(disk_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError(f"need HxWx3 image and HxW mask, got {rgb.shape} and {mask.shape}")
    out = rgb.copy()
    if severity == 0 or not mask.any():
        return out
    if family in ("glare", "noise") and not isinstance(rng, np.random.Generator):
        raise TypeError(f"{family} needs a numpy Generator for rng")
    lin = _OPS[family](_to_linear(rgb), mask, _SPECS[family][2][severity], rng)
    out[mask] = _to_srgb(lin[mask])
    return out


def montage(rgb, disk_mask, seed=0, tile=288):
    """Grid of one frame: rows = FAMILIES, columns = severities 0-5, as uint8 RGB with labels."""
    left, top, pad = 150, 26, 2
    n_r, n_c = len(FAMILIES), len(SEVERITIES)
    canvas = np.full((top + n_r * (tile + pad), left + n_c * (tile + pad), 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r, fam in enumerate(FAMILIES):
        y0 = top + r * (tile + pad)
        cv2.putText(canvas, fam, (6, y0 + tile // 2), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, _SPECS[fam][0], (6, y0 + tile // 2 + 18), font, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        for c, sev in enumerate(SEVERITIES):
            img = apply(rgb, disk_mask, fam, sev, np.random.default_rng(seed))
            small = cv2.resize((img * 255 + 0.5).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_AREA)
            x0 = left + c * (tile + pad)
            canvas[y0 : y0 + tile, x0 : x0 + tile] = small
            label = f"{_SPECS[fam][2][sev]:g}"
            cv2.putText(canvas, label, (x0 + 5, y0 + 16), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (x0 + 5, y0 + 16), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for c, sev in enumerate(SEVERITIES):
        cv2.putText(canvas, f"severity {sev}", (left + c * (tile + pad) + 5, 17), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas
