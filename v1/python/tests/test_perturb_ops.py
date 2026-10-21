import cv2
import numpy as np
import pytest
import yaml

import perturb_ops as po
from conftest import V1

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
STRICT = [s for s in po.SEVERITIES if s > 0]


def make_scene(seed=0, size=512):
    """Reddish textured 'tissue' with vignette, a sharp-edged grey instrument and junk outside the disk."""
    r = np.random.default_rng(seed)

    def smooth(sigma, scale):
        return cv2.GaussianBlur(r.standard_normal((size, size)).astype(np.float32), (0, 0), sigma) * scale

    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rad = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    img = np.stack([0.62 + smooth(40, 6), 0.38 + smooth(40, 6), 0.33 + smooth(40, 6)], -1)
    img += smooth(2.5, 0.45)[..., None]
    img += np.clip(smooth(25, 40), 0, None)[..., None] * 0.5  # bright sheen so highlights start to clip early
    band = np.abs((xx - 300) * 0.8 + (yy - 200) * 0.6) < 30
    img[band] = 0.55 + smooth(2.5, 0.3)[band][:, None]
    img *= (1.0 - 0.35 * rad**2)[..., None]
    img = np.clip(img, 0.02, 0.95).astype(np.float32)

    mask = rad <= 240 / 256
    mask[470:, 246:266] = False  # notch like the real FOV
    img[~mask] = r.uniform(0.2, 0.9, (int((~mask).sum()), 3)).astype(np.float32)
    return img, mask


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def inner(scene):
    return cv2.erode(scene[1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0


def lum(img):
    return img @ LUM


def lap_var(img, m):
    return float(cv2.Laplacian(lum(img), cv2.CV_32F)[m].var())


def dark_channel_mean(img, m):
    dc = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8))
    return float(dc[m].mean())


def rms_contrast(img, m):
    return float(lum(img)[m].std())


def mean_lum(img, m):
    return float(lum(img)[m].mean())


def sat_frac(img, m):
    return float((img.max(axis=2) >= 0.98)[m].mean())


def red_dom(img, m):
    return float((img[..., 0] / np.maximum(img.sum(axis=2), 1e-6))[m].mean())


def run(scene, family, severity, seed=11):
    return po.apply(scene[0], scene[1], family, severity, np.random.default_rng(seed))


def test_conditions_cover_grid_and_are_monotone():
    conds = po.list_conditions()
    assert len(conds) == len(po.FAMILIES) * len(po.SEVERITIES) == 42
    assert all(set(c) == {"family", "severity", "param", "param_name", "unit"} for c in conds)
    assert len(po.list_conditions(include_baseline=False)) == 35
    for fam in po.FAMILIES:
        params = [c["param"] for c in conds if c["family"] == fam]
        assert [c["severity"] for c in conds if c["family"] == fam] == list(range(6))
        d = np.diff(params)
        assert np.all(d > 0) or np.all(d < 0), fam
    assert next(c["param"] for c in conds if c["family"] == "exposure_under" and c["severity"] == 0) == 1.0
    assert next(c["param"] for c in conds if c["family"] == "exposure_under" and c["severity"] == 5) < 1.0
    assert next(c["param"] for c in conds if c["family"] == "exposure_over" and c["severity"] == 5) > 1.0
    assert po.CONDITIONS is not po.list_conditions()


def test_families_and_severities_match_analysis_plan():
    cfg = yaml.safe_load((V1 / "config" / "analysis_plan.yaml").read_text())["perturbation"]
    fams = [f for f in cfg["families"] if f != "exposure"]
    fams += [f"exposure_{arm}" for arm in cfg["exposure_arms"]]
    assert sorted(fams) == sorted(po.FAMILIES)
    assert list(cfg["severities"]) == list(po.SEVERITIES)


@pytest.mark.parametrize("family", po.FAMILIES)
def test_severity0_is_identity_copy(scene, family):
    out = run(scene, family, 0)
    assert np.array_equal(out, scene[0])
    assert out.dtype == np.float32
    assert not np.shares_memory(out, scene[0])


@pytest.mark.parametrize("family", po.FAMILIES)
@pytest.mark.parametrize("severity", STRICT)
def test_outside_disk_bit_identical_and_range(scene, family, severity):
    img, mask = scene
    out = run(scene, family, severity)
    assert out.dtype == np.float32 and out.shape == img.shape
    assert np.array_equal(out[~mask], img[~mask])
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    assert np.abs(out[mask] - img[mask]).max() > 0.0  # a perturbation actually happened


@pytest.mark.parametrize("family", po.FAMILIES)
def test_no_leak_from_outside_the_disk(scene, family):
    img, mask = scene
    other = img.copy()
    other[~mask] = np.random.default_rng(99).uniform(0.0, 1.0, (int((~mask).sum()), 3)).astype(np.float32)
    for severity in (2, 5):
        a = po.apply(img, mask, family, severity, np.random.default_rng(5))
        b = po.apply(other, mask, family, severity, np.random.default_rng(5))
        assert np.array_equal(a[mask], b[mask])


@pytest.mark.parametrize("family", po.FAMILIES)
def test_deterministic_given_seed(scene, family):
    a, b = run(scene, family, 3, seed=7), run(scene, family, 3, seed=7)
    assert np.array_equal(a, b)
    c = run(scene, family, 3, seed=8)
    if family in ("glare", "noise"):
        assert not np.array_equal(a, c)
    else:
        assert np.array_equal(a, c)  # deterministic families ignore rng


@pytest.mark.parametrize("family", ["defocus", "haze", "exposure_under", "exposure_over", "blood_absorption"])
def test_deterministic_families_leave_rng_untouched(scene, family):
    rng = np.random.default_rng(3)
    po.apply(scene[0], scene[1], family, 4, rng)
    assert rng.bit_generator.state == np.random.default_rng(3).bit_generator.state


def test_input_not_modified_and_float64_accepted(scene):
    img, mask = scene
    before = img.copy()
    po.apply(img, mask, "haze", 5, np.random.default_rng(0))
    assert np.array_equal(img, before)
    out = po.apply(img.astype(np.float64), mask, "haze", 3, np.random.default_rng(0))
    assert out.dtype == np.float32


def test_bad_arguments(scene):
    img, mask = scene
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        po.apply(img, mask, "fog", 1, rng)
    with pytest.raises(ValueError):
        po.apply(img, mask, "haze", 6, rng)
    with pytest.raises(ValueError):
        po.apply(img, mask[:-1], "haze", 1, rng)
    with pytest.raises(TypeError):
        po.apply(img, mask, "noise", 1, None)


def test_empty_mask_returns_copy(scene):
    out = po.apply(scene[0], np.zeros_like(scene[1]), "glare", 5, np.random.default_rng(0))
    assert np.array_equal(out, scene[0])


# Family-appropriate statistic, direction of travel with severity.
MONOTONE = [
    ("defocus", "laplacian variance falls", lambda o, s, m: lap_var(o, m), -1),
    ("haze", "dark-channel mean rises", lambda o, s, m: dark_channel_mean(o, m), +1),
    ("haze", "RMS contrast falls", lambda o, s, m: rms_contrast(o, m), -1),
    ("exposure_under", "mean luminance falls", lambda o, s, m: mean_lum(o, m), -1),
    ("exposure_over", "mean luminance rises", lambda o, s, m: mean_lum(o, m), +1),
    ("exposure_over", "saturated fraction rises", lambda o, s, m: sat_frac(o, m), +1),
    ("blood_absorption", "R/(R+G+B) rises", lambda o, s, m: red_dom(o, m), +1),
    ("glare", "saturated fraction rises", lambda o, s, m: sat_frac(o, m), +1),
    ("noise", "residual std rises", lambda o, s, m: float((o - s)[m].std()), +1),
]


@pytest.mark.parametrize("family,name,stat,sign", MONOTONE, ids=[f"{m[0]}:{m[1]}" for m in MONOTONE])
def test_statistic_monotone_in_severity(scene, inner, family, name, stat, sign):
    vals = [stat(run(scene, family, s), scene[0], inner) for s in po.SEVERITIES]
    assert np.all(sign * np.diff(vals) > 0), (family, name, vals)


@pytest.mark.parametrize("family", po.FAMILIES)
def test_severity5_degraded_but_not_black_or_white(scene, inner, family):
    out = run(scene, family, 5)
    y = lum(out)[inner]
    assert 0.08 < y.mean() < 0.92
    assert y.std() > 0.02  # structure survives
    assert np.abs(lum(out) - lum(scene[0]))[inner].mean() > 0.01  # and the change is clear


def test_glare_saturated_area_tracks_nominal(scene, inner):
    base = sat_frac(scene[0], inner)
    area = scene[1].sum() / inner.sum()
    for sev in (3, 4, 5):
        nominal = po._SPECS["glare"][2][sev]
        added = [sat_frac(run(scene, "glare", sev, seed=s), inner) - base for s in range(8)]
        # feathered edges and overlaps: saturated core is a bit under the half-opacity area
        assert 0.35 * nominal * area < np.median(added) < 1.15 * nominal * area


def test_glare_layout_nests_across_severity(scene):
    lo, hi = run(scene, "glare", 2, seed=4), run(scene, "glare", 4, seed=4)
    assert np.all(lum(hi) >= lum(lo) - 1e-6)


def test_noise_field_nests_across_severity(scene, inner):
    d1 = (run(scene, "noise", 2, seed=4) - scene[0])[inner].ravel()
    d2 = (run(scene, "noise", 4, seed=4) - scene[0])[inner].ravel()
    assert np.corrcoef(d1, d2)[0, 1] > 0.8


def test_montage_shape(scene):
    m = po.montage(scene[0], scene[1], tile=32)
    assert m.dtype == np.uint8 and m.ndim == 3 and m.shape[2] == 3
    assert m.shape[0] > len(po.FAMILIES) * 32 and m.shape[1] > len(po.SEVERITIES) * 32
