import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import fov

ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Spine endoscopic atlas"


def synth_frame(w, h, cx, cy, rx, ry=None, surround=(0, 0, 0), cast=(1.0, 1.0, 1.0), notch=True, seed=0):
    """Textured pink disk on a flat surround; edge anti-aliased with the true edge at distance 0."""
    ry = rx if ry is None else ry
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    low = cv2.resize(rng.random((9, 16)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    tissue = (0.45 + 0.4 * low)[..., None] * np.array([235, 150, 130], np.float32) * np.array(cast, np.float32)
    tissue += rng.normal(0, 4, tissue.shape).astype(np.float32)
    d = (np.hypot((xx - cx) / rx, (yy - cy) / ry) - 1.0) * min(rx, ry)
    alpha = np.clip(0.5 - d, 0, 1)[..., None]
    img = alpha * tissue + (1 - alpha) * np.array(surround, np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if notch:
        tri = np.array([[cx - 0.04 * ry, cy + ry + 3], [cx + 0.04 * ry, cy + ry + 3], [cx, cy + 0.9 * ry]], np.int32)
        cv2.fillConvexPoly(img, tri, tuple(int(c) for c in surround))
    return img


CASES = [
    ("D1", 1920, 1080, 940.0, 540.0, 430.0),
    ("D1-offset", 1920, 1080, 985.0, 520.0, 400.0),
    ("D2", 1280, 720, 640.0, 360.0, 290.0),
    ("D3", 720, 576, 365.0, 288.0, 265.0),
]


@pytest.mark.parametrize("notch", [True, False])
@pytest.mark.parametrize("name,w,h,cx,cy,r", CASES, ids=[c[0] for c in CASES])
def test_synthetic_circle_found(name, w, h, cx, cy, r, notch):
    out = fov.detect_fov(synth_frame(w, h, cx, cy, r, notch=notch))
    assert out["found"] and out["method"] == "ray"
    assert abs(out["cx"] - cx) < 1.5 and abs(out["cy"] - cy) < 1.5
    assert abs(out["r"] - r) < 0.006 * r + 1.0
    assert out["resid"] < 0.01


def test_brown_surround_and_colour_cast():
    img = synth_frame(720, 576, 360.0, 288.0, 268.0, surround=(60, 35, 20), cast=(1.0, 0.8, 0.6))
    out = fov.detect_fov(img)
    assert out["found"]
    assert abs(out["r"] - 268.0) < 3 and abs(out["cx"] - 360) < 2 and abs(out["cy"] - 288) < 2


def test_dark_instrument_merging_with_surround():
    # a black bar across the top rim: the coarse threshold loses that arc, the ray fit must not
    img = synth_frame(1920, 1080, 960.0, 540.0, 420.0)
    cv2.rectangle(img, (800, 100), (1120, 330), (0, 0, 0), -1)
    out = fov.detect_fov(img)
    assert out["found"]
    assert abs(out["r"] - 420.0) < 4 and abs(out["cx"] - 960) < 3 and abs(out["cy"] - 540) < 3


def test_circle_clipped_by_image_border():
    out = fov.detect_fov(synth_frame(1920, 1080, 960.0, 540.0, 580.0))
    assert out["found"]
    assert abs(out["r"] - 580.0) < 6 and abs(out["cx"] - 960) < 4 and abs(out["cy"] - 540) < 8


def test_anamorphic_fov_reports_inscribed_circle_with_stable_centre():
    # Center 2 stores the FOV ~4% taller than wide: centre must be the ellipse centre, r the shorter axis
    for seed, cy in [(0, 280.0), (1, 292.0)]:
        out = fov.detect_fov(synth_frame(720, 576, 362.0, cy, 267.0, ry=277.0, seed=seed))
        assert out["found"]
        assert abs(out["cx"] - 362) < 1.5 and abs(out["cy"] - cy) < 1.5
        assert abs(out["r"] - 267.0) < 3


@pytest.mark.parametrize("img", [
    np.zeros((576, 720, 3), np.uint8),
    np.full((576, 720, 3), 120, np.uint8),
    np.random.default_rng(0).integers(0, 255, (576, 720, 3)).astype(np.uint8),
], ids=["black", "uniform", "noise"])
def test_not_found_returns_fallback(img):
    out = fov.detect_fov(img)
    assert out["found"] is False and out["method"] == "fallback"
    assert out["r"] == 0.5 * min(img.shape[:2])


def test_float_input_matches_uint8():
    img = synth_frame(1280, 720, 640.0, 360.0, 290.0)
    a, b = fov.detect_fov(img), fov.detect_fov(img.astype(np.float32) / 255.0)
    assert a["found"] and b["found"] and abs(a["r"] - b["r"]) < 0.5


def test_detect_speed_1080p():
    img = synth_frame(1920, 1080, 960.0, 540.0, 420.0)
    fov.detect_fov(img)
    # order-of-magnitude guard: bound = 10x the median measured on an idle machine (about 3 ms)
    med = float(np.median([_time(lambda: fov.detect_fov(img)) for _ in range(9)]))
    assert med < 0.030


def _time(f):
    t = time.perf_counter()
    f()
    return time.perf_counter() - t


def test_crop_box_side_is_round_two_r():
    assert fov.crop_box(100.0, 100.0, 50.4)[2] == 101
    x0, y0, side = fov.crop_box(100.0, 80.0, 50.0)                    # even side: centre within half a pixel
    assert abs(x0 + (side - 1) / 2 - 100.0) <= 0.5 and abs(y0 + (side - 1) / 2 - 80.0) <= 0.5
    x0, y0, side = fov.crop_box(100.0, 80.0, 50.4)                    # odd side: exact
    assert (x0 + (side - 1) / 2, y0 + (side - 1) / 2) == (100.0, 80.0)


def test_crop_square_zero_pads_outside_image():
    img = np.full((100, 100, 3), 200, np.uint8)
    out = fov.crop_square(img, 10.0, 50.0, 40.0, 80, "area")      # crop reaches 30 px left of the image
    assert out.shape == (80, 80, 3) and out.dtype == np.uint8
    assert out[:, :20].max() == 0 and out[:, 50:].min() == 200
    assert fov.crop_square(img, 500.0, 500.0, 20.0, 40).max() == 0  # fully outside


def test_crop_square_mask_nearest_keeps_labels_and_dtype():
    m = np.zeros((200, 300), np.uint8)
    m[50:100, 100:160] = 1
    m[120:160, 200:240] = 255
    out = fov.crop_square(m, 150.0, 100.0, 90.0, 64, "nearest")
    assert out.shape == (64, 64) and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 1, 255} and {1, 255} <= set(np.unique(out))
    b = fov.crop_square(m > 0, 150.0, 100.0, 90.0, 64, "linear")  # bool is forced to nearest
    assert b.dtype == bool and b.shape == (64, 64) and b.any()
    f = fov.crop_square(m.astype(np.float32), 150.0, 100.0, 90.0, 64)
    assert f.dtype == np.float32 and f.shape == (64, 64)


def test_fov_disk_mask_geometry():
    m = fov.fov_disk_mask(512, 0.02)
    assert m.shape == (512, 512) and m.dtype == bool
    assert m[255, 255] and not m[0, 0] and not m[0, 511]
    assert np.array_equal(m, m[::-1, ::-1])
    assert abs(m.mean() - np.pi / 4 * 0.98 ** 2) < 0.003
    assert abs(fov.fov_disk_mask(512, 0.0).mean() - np.pi / 4) < 0.003
    assert fov.fov_disk_mask(512, 0.1).sum() < fov.fov_disk_mask(512, 0.02).sum()


def test_canonical_canvas_matches_fov_and_keeps_surround_out_of_the_disk():
    img = synth_frame(1920, 1080, 985.0, 520.0, 400.0, surround=(60, 35, 20), notch=False)
    det = fov.detect_fov(img)
    canvas, disk = fov.canonical_canvas(img, det)
    assert canvas.shape == (512, 512, 3) and canvas.dtype == np.float32
    assert 0.0 <= canvas.min() and canvas.max() <= 1.0 and disk.shape == (512, 512)
    lum = canvas.mean(axis=2)
    assert lum[disk].min() > 0.20                     # surround is 0.15; tissue is brighter
    ring = (~fov.fov_disk_mask(512, -0.06)) & (fov.fov_disk_mask(512, -0.30))
    outer = np.hypot(*(np.indices((512, 512)) - 255.5)) > 256 * 1.04
    assert abs(np.median(lum[outer & ring]) - 0.15) < 0.01
    same = fov.canonical_canvas(img, dict(fov_cx=det["cx"], fov_cy=det["cy"], fov_r=det["r"]))[0]
    assert np.array_equal(canvas, same)


def _atlas_samples():
    labeled = sorted((ATLAS / "classified").rglob("*.jpg")) + sorted((ATLAS / "classified").rglob("*.png"))
    want, got = {(1920, 1080): [], (1280, 720): [], (720, 576): []}, []
    from PIL import Image
    for p in labeled[::53]:
        with Image.open(p) as im:
            k = im.size
        if k in want and len(want[k]) < 5:
            want[k].append(p)
    got = [p for v in want.values() for p in v]
    stream = sorted((ATLAS / "unclassified").glob("P*/*.jpg"))
    return got + stream[::9000][:4]


@pytest.mark.skipif(not ATLAS.exists(), reason="dataset not present")
def test_real_frames_across_domains_and_stream():
    from PIL import Image
    paths = _atlas_samples()
    assert len(paths) >= 12
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        out = fov.detect_fov(img)
        assert out["found"], p.name
        frac = out["r"] / min(img.shape[:2])
        assert 0.35 < frac < 0.6 and out["resid"] < 0.01, (p.name, out)
        canvas, disk = fov.canonical_canvas(img, out)
        rn = np.hypot(*(np.indices((512, 512)) - 255.5)) / 256
        assert np.median(canvas.mean(axis=2)[(rn > 1.04) & (rn < 1.15)]) < 0.08, p.name   # outside the disk is surround
