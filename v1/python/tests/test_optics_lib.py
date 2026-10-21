import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import optics_lib as ol

ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Spine endoscopic atlas"
SIZE = 512
YY, XX = np.indices((SIZE, SIZE))
RAD = np.hypot(XX - 255.5, YY - 255.5)
DISK = RAD <= 0.98 * 256
EDGE = np.sqrt(DISK.sum() / np.pi)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def tissue(seed=0, scale=1.0, colour=(0.85, 0.55, 0.50), noise=0.02):
    """Smooth pink texture with mild noise; luma about 0.55 * scale."""
    rng = np.random.default_rng(seed)
    low = cv2.resize(rng.random((12, 12)).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    img = (0.6 + 0.4 * low)[..., None] * np.array(colour, np.float32) * scale
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def flat(v):
    return np.full((SIZE, SIZE, 3), v, np.float32)


def blob_mask(centres, radius):
    m = np.zeros((SIZE, SIZE), bool)
    for cx, cy in centres:
        m |= np.hypot(XX - cx, YY - cy) <= radius
    return m


def test_metric_names_and_order_follow_the_plan():
    cfg = yaml.safe_load(open(ROOT / "v1" / "config" / "analysis_plan.yaml"))["optical_metrics"]
    out = ol.compute_metrics(tissue(), DISK)
    assert len(cfg) == 11 and list(out) == cfg
    assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())


def test_constant_gray_has_no_contrast_texture_or_gradient():
    out = ol.compute_metrics(flat(0.5), DISK)
    assert out["lum_mean"] == pytest.approx(0.5, abs=1e-6)
    for k in ("contrast_rms", "contrast_local", "sharp_tenengrad", "under_frac", "sat_frac", "spec_frac", "lum_entropy"):
        assert out[k] == pytest.approx(0.0, abs=1e-6), k
    assert out["haze_dcp"] == pytest.approx(0.5, abs=1e-6)
    assert out["red_dom"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["illum_slope"] == pytest.approx(0.0, abs=1e-6)


def test_contrast_rms_matches_known_noise_and_local_contrast_ignores_large_steps():
    rng = np.random.default_rng(1)
    g = np.clip(0.5 + rng.normal(0, 0.05, (SIZE, SIZE)), 0, 1).astype(np.float32)
    out = ol.compute_metrics(np.repeat(g[..., None], 3, axis=2), DISK)
    assert out["contrast_rms"] == pytest.approx(0.05, rel=0.03)
    assert out["contrast_local"] == pytest.approx(0.05, rel=0.03)
    tiles = (((XX // 128) + (YY // 128)) % 2).astype(np.float32) * 0.6 + 0.2     # 128 px checkerboard, aligned to blocks
    out = ol.compute_metrics(np.repeat(tiles[..., None], 3, axis=2), DISK)
    assert out["contrast_rms"] > 0.25 and out["contrast_local"] < 0.02


def test_illum_slope_sign_and_size():
    falloff = 0.8 - 0.4 * RAD / 256
    for sign, y in ((-1, falloff), (+1, 0.4 + 0.4 * RAD / 256)):
        out = ol.compute_metrics(np.repeat(y[..., None].astype(np.float32), 3, axis=2), DISK)
        assert np.sign(out["illum_slope"]) == sign
        assert abs(out["illum_slope"]) == pytest.approx(0.4 * EDGE / 256, rel=0.02)


def test_sat_and_under_fraction_equal_patch_area():
    img = flat(0.5)
    img[200:264, 200:264] = 1.0
    img[300:340, 300:380] = 0.05
    n = DISK.sum()
    out = ol.compute_metrics(img, DISK)
    assert out["sat_frac"] == pytest.approx(64 * 64 / n, abs=1e-6)
    assert out["under_frac"] == pytest.approx(40 * 80 / n, abs=1e-6)
    one_channel = flat(0.5)
    one_channel[200:264, 200:264, 0] = 0.99               # any channel counts
    assert ol.compute_metrics(one_channel, DISK)["sat_frac"] == pytest.approx(64 * 64 / n, abs=1e-6)


def test_blur_lowers_sharpness_and_local_contrast():
    sharp = tissue(2, noise=0.05)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.5)
    a, b = ol.compute_metrics(sharp, DISK), ol.compute_metrics(blurred, DISK)
    assert b["sharp_tenengrad"] < 0.25 * a["sharp_tenengrad"]
    assert b["contrast_local"] < a["contrast_local"]


def test_haze_raises_dark_channel():
    rng = np.random.default_rng(3)
    j = cv2.resize(rng.random((16, 16, 3)).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    t, a = 0.5, 0.9
    hazy = j * t + a * (1 - t)                                  # I = J t + A (1 - t)
    base, haz = ol.compute_metrics(j, DISK), ol.compute_metrics(hazy, DISK)
    assert haz["haze_dcp"] > base["haze_dcp"] + 0.3
    assert haz["haze_dcp"] == pytest.approx(t * base["haze_dcp"] + a * (1 - t), abs=1e-4)   # dark channel is linear in t
    assert haz["contrast_rms"] == pytest.approx(t * base["contrast_rms"], rel=0.02)


def test_red_tint_raises_red_dominance():
    base = tissue(4)
    tinted = np.clip(base * np.array([1.3, 0.9, 0.8], np.float32), 0, 1)
    assert ol.compute_metrics(tinted, DISK)["red_dom"] > ol.compute_metrics(base, DISK)["red_dom"] + 0.02
    colour = flat(0)
    colour[...] = (0.6, 0.3, 0.1)
    assert ol.compute_metrics(colour, DISK)["red_dom"] == pytest.approx(0.6, abs=1e-6)


def test_entropy_bits():
    rng = np.random.default_rng(5)
    uni = np.repeat(rng.random((SIZE, SIZE, 1)).astype(np.float32), 3, axis=2)
    assert ol.compute_metrics(uni, DISK)["lum_entropy"] == pytest.approx(6.0, abs=0.02)
    two = flat(0.25)
    two[:, :256] = 0.75
    assert ol.compute_metrics(two, DISK)["lum_entropy"] == pytest.approx(1.0, abs=0.01)


# specular fraction ---------------------------------------------------------

BLOBS = [(200, 200), (330, 260), (250, 340), (300, 190)]


def with_glare(img, radius=10, colour=(1.0, 0.99, 0.97)):
    m = blob_mask(BLOBS, radius)
    out = img.copy()
    out[m] = colour
    return out, m


def test_spec_frac_counts_glare_blobs_in_normal_dark_and_bright_frames():
    for scale, colour in ((1.0, (0.85, 0.55, 0.50)), (0.3, (0.85, 0.55, 0.50)), (1.0, (0.92, 0.90, 0.88))):
        base = tissue(6, scale=scale, colour=colour)
        img, m = with_glare(base)
        truth = (m & DISK).sum() / DISK.sum()
        assert ol.compute_metrics(base, DISK)["spec_frac"] < 0.002, (scale, colour)
        assert ol.compute_metrics(img, DISK)["spec_frac"] == pytest.approx(truth, abs=0.003), (scale, colour)


def test_spec_frac_ignores_uniformly_bright_saturated_and_coloured_regions():
    assert ol.compute_metrics(flat(0.92), DISK)["spec_frac"] == 0.0           # over-exposed frame is not glare
    assert ol.compute_metrics(flat(1.0), DISK)["spec_frac"] == 0.0
    red, _ = with_glare(tissue(7), colour=(1.0, 0.15, 0.15))                    # bright but strongly coloured
    assert ol.compute_metrics(red, DISK)["spec_frac"] < 0.002
    bone = tissue(8, colour=(0.97, 0.94, 0.90))                                # bright pale tissue, no glare
    assert ol.compute_metrics(bone, DISK)["spec_frac"] < 0.005


def test_spec_frac_grows_with_glare_area():
    base = tissue(9)
    fr = [ol.compute_metrics(with_glare(base, radius=r)[0], DISK)["spec_frac"] for r in (4, 8, 14)]
    assert fr[0] < fr[1] < fr[2]


# masks ---------------------------------------------------------------------

def test_bg_mask_restricts_every_metric_to_the_background():
    img = flat(0.2)
    img[:, :256] = 0.8                                     # bright left half
    right = XX >= 256
    out = ol.compute_metrics(img, DISK, right)
    assert out["lum_mean"] == pytest.approx(0.2, abs=1e-6)
    assert out["haze_dcp"] == pytest.approx(0.2, abs=1e-6)  # patch minimum never sees the left half
    both = ol.compute_metrics(img, DISK)
    assert both["lum_mean"] == pytest.approx(0.5, abs=0.01) and both["contrast_rms"] > 0.25
    assert out["contrast_rms"] < 1e-6


def test_bg_mask_removes_instrument_pixels_from_saturation_haze_and_gradient_metrics():
    base = tissue(10)
    inst = blob_mask([(256, 256)], 70)
    img = base.copy()
    img[inst] = (1.0, 1.0, 1.0)
    full = ol.compute_metrics(img, DISK)
    bg = ol.compute_metrics(img, DISK, ~inst)
    ref = ol.compute_metrics(base, DISK, ~inst)
    assert full["sat_frac"] > 0.05 and bg["sat_frac"] == pytest.approx(0.0, abs=1e-6)
    assert bg["sharp_tenengrad"] == pytest.approx(ref["sharp_tenengrad"], rel=0.02)   # no instrument edge leaks in
    assert full["sharp_tenengrad"] > 1.2 * bg["sharp_tenengrad"]
    dark = base.copy()
    dark[inst] = 0.0
    assert ol.compute_metrics(dark, DISK, ~inst)["haze_dcp"] == pytest.approx(ref["haze_dcp"], abs=1e-6)
    assert ol.compute_metrics(dark, DISK)["haze_dcp"] < ref["haze_dcp"] - 0.01


def test_disk_edge_does_not_leak_into_metrics():
    # surround outside the disk is black; none of it may reach the dark channel or the gradients
    img = flat(0.5)
    img[~DISK] = 0.0
    out = ol.compute_metrics(img, DISK)
    assert out["haze_dcp"] == pytest.approx(0.5, abs=1e-6)
    assert out["sharp_tenengrad"] == pytest.approx(0.0, abs=1e-9)
    assert out["under_frac"] == 0.0


def test_too_few_pixels_returns_nan_and_inputs_are_not_modified():
    out = ol.compute_metrics(tissue(), DISK, np.zeros((SIZE, SIZE), bool))
    assert list(out) == list(ol.METRICS) and all(np.isnan(v) for v in out.values())
    img, disk, bg = tissue(11), DISK.copy(), (XX > 100)
    img0, bg0 = img.copy(), bg.copy()
    ol.compute_metrics(img, disk, bg)
    assert np.array_equal(img, img0) and np.array_equal(disk, DISK) and np.array_equal(bg, bg0)
    f64 = ol.compute_metrics(tissue(11).astype(np.float64), DISK)
    f32 = ol.compute_metrics(tissue(11), DISK)
    assert all(f64[k] == pytest.approx(f32[k], rel=1e-4, abs=1e-6) for k in f32)


def test_speed_512():
    img, bg = tissue(12), XX > 120
    ol.compute_metrics(img, DISK, bg)
    ts = []
    for _ in range(9):
        t = time.perf_counter()
        ol.compute_metrics(img, DISK, bg)
        ts.append(time.perf_counter() - t)
    # order-of-magnitude guard: bound = 10x the median measured on an idle machine (about 12 ms)
    assert np.median(ts) < 0.120


@pytest.mark.skipif(not ATLAS.exists(), reason="dataset not present")
def test_real_frames_give_finite_plausible_metrics():
    from PIL import Image

    import fov
    pick = lambda xs: [xs[i] for i in np.linspace(0, len(xs) - 1, 8).astype(int)]
    paths = pick(sorted((ATLAS / "classified").rglob("*.jpg"))) + pick(sorted((ATLAS / "classified").rglob("*.png")))
    assert len(paths) >= 12
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        det = fov.detect_fov(img)
        canvas, disk = fov.canonical_canvas(img, det)
        out = ol.compute_metrics(canvas, disk)
        assert all(np.isfinite(v) for v in out.values()), p.name
        assert 0.02 < out["lum_mean"] < 0.95 and 0 <= out["under_frac"] <= 1 and 0 <= out["spec_frac"] < 0.3, (p.name, out)
        assert 0.30 < out["red_dom"] < 0.95 and 0 < out["lum_entropy"] <= 6, (p.name, out)
