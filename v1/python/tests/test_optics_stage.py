import json
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, load_script

optics = load_script("02_optics.py")
sys.modules[optics.__name__] = optics   # fork workers look the worker functions up by module name

MANIFEST = ROOT / "v1" / "derived" / "manifest.parquet"
FOV_COLS = ["frame_id", "fov_found", "fov_cx", "fov_cy", "fov_r", "fov_method", "fov_resid", "crop_x0", "crop_y0", "crop_side"]
METRICS = list(optics.optics_lib.METRICS)


def _disk_image(path, h=480, w=640, r=190, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    inside = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= r ** 2
    tex = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 3)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 170, 110, 90
    img += tex[..., None]
    img[~inside] = 4
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


@pytest.fixture
def synthetic(tmp_path):
    """Two endoscope-like frames and one black frame (no FOV), with absolute image paths."""
    paths = []
    for name, kind in (("a", "disk"), ("b", "black"), ("c", "disk")):
        p = tmp_path / f"{name}.png"
        if kind == "disk":
            _disk_image(p, seed=len(paths))
        else:
            cv2.imwrite(str(p), np.full((480, 640, 3), 3, np.uint8))
        paths.append(str(p))
    m = pd.DataFrame(dict(frame_id=["A", "B", "C"], image_path=paths))
    mp = tmp_path / "manifest.parquet"
    m.to_parquet(mp, index=False)
    return mp, tmp_path


@pytest.fixture(scope="module")
def real_subset(tmp_path_factory):
    if not MANIFEST.exists():
        pytest.skip("derived/manifest.parquet missing")
    m = pd.read_parquet(MANIFEST)
    if not (ROOT / m.image_path.iloc[0]).exists():
        pytest.skip("image data not available")
    rng = np.random.default_rng(3)
    pick = []
    for dom, k in (("D1", 7), ("D2", 8), ("D3", 8)):
        pick += rng.choice(m.index[m.labeled & (m.domain == dom)], k, replace=False).tolist()
    pick += rng.choice(m.index[~m.labeled], 7, replace=False).tolist()
    sub = m.loc[sorted(pick)].reset_index(drop=True)
    path = tmp_path_factory.mktemp("subset") / "manifest.parquet"
    sub.to_parquet(path, index=False)
    return path, sub


def _run(manifest, out, workers=2, chunk=8):
    parts, fov_p, opt_p = out / "parts", out / "fov.parquet", out / "optics.parquet"
    r1 = optics.run_fov(manifest, fov_p, parts, workers, chunk, log=lambda *_: None)
    r2 = optics.run_metrics(manifest, fov_p, opt_p, parts, workers, chunk, log=lambda *_: None)
    return r1, r2, pd.read_parquet(fov_p), pd.read_parquet(opt_p)


def test_synthetic_schema_order_and_unfound_frames(synthetic):
    manifest, tmp = synthetic
    _, _, fov, opt = _run(manifest, tmp / "out", chunk=2)
    assert list(fov.columns) == FOV_COLS and list(opt.columns) == ["frame_id", *METRICS]
    assert fov.frame_id.tolist() == opt.frame_id.tolist() == ["A", "B", "C"]
    assert fov.fov_found.tolist() == [True, False, True]
    b = fov.set_index("frame_id").loc["B"]
    assert b.fov_method == "fallback" and (b.fov_cx, b.fov_cy, b.fov_r) == (319.5, 239.5, 240.0)
    nan_row = opt.set_index("frame_id").loc["B"]
    assert nan_row.isna().all()
    ok = opt.set_index("frame_id").loc[["A", "C"]]
    assert np.isfinite(ok.to_numpy()).all()
    assert abs(fov.fov_r.iloc[0] - 190) < 2 and abs(fov.fov_cx.iloc[0] - 319.5) < 2
    assert (fov.crop_side == fov.fov_r.mul(2).round().astype(int)).all()
    assert fov.fov_found.dtype == bool and fov.crop_side.dtype.kind == "i"


def test_synthetic_resume_skips_finished_chunks_and_repairs_bad_parts(synthetic):
    manifest, tmp = synthetic
    out = tmp / "out"
    r1, r2, fov, opt = _run(manifest, out, chunk=2)
    assert (r1["done"], r1["skipped"], r2["done"], r2["skipped"]) == (2, 0, 2, 0)
    parts = out / "parts" / "fov"
    keep = {p.name: p.stat().st_mtime_ns for p in parts.glob("part_*.parquet")}
    (parts / "part_00001.parquet").unlink()
    (parts / "part_00000.parquet").write_bytes(b"not parquet")
    r1b, r2b, fov2, opt2 = _run(manifest, out, chunk=2)
    assert (r1b["done"], r1b["skipped"]) == (2, 0)          # the broken part is recomputed too
    assert (r2b["done"], r2b["skipped"]) == (0, 2)
    pd.testing.assert_frame_equal(fov, fov2, check_exact=True)
    pd.testing.assert_frame_equal(opt, opt2, check_exact=True)
    r1c, _, _, _ = _run(manifest, out, chunk=2)
    assert (r1c["done"], r1c["skipped"]) == (0, 2)
    assert set(keep) == {p.name for p in parts.glob("part_*.parquet")}


def test_changed_input_discards_parts(synthetic):
    manifest, tmp = synthetic
    out = tmp / "out"
    _run(manifest, out, chunk=2)
    m = pd.read_parquet(manifest)
    m.iloc[:2].to_parquet(manifest, index=False)
    r1, r2, fov, opt = _run(manifest, out, chunk=2)
    assert r1["skipped"] == 0 and r2["skipped"] == 0 and len(fov) == len(opt) == 2


def test_real_subset_schema_order_determinism_resume(real_subset, tmp_path):
    manifest, sub = real_subset
    r1, r2, fov, opt = _run(manifest, tmp_path / "one", workers=2, chunk=8)
    assert list(fov.columns) == FOV_COLS and list(opt.columns) == ["frame_id", *METRICS]
    assert fov.frame_id.tolist() == opt.frame_id.tolist() == sub.frame_id.tolist()
    assert len(fov) == 30 and r1["done"] == 4 and r2["done"] == 4
    assert fov.fov_found.all() and set(fov.fov_method) <= {"ray", "mask"}
    assert np.isfinite(opt[METRICS].to_numpy()).all()
    assert (fov.fov_r > 0.25 * np.minimum(sub.width, sub.height)).all()
    # different worker count and chunking give bit-identical output
    _, _, fov_b, opt_b = _run(manifest, tmp_path / "two", workers=3, chunk=7)
    pd.testing.assert_frame_equal(fov, fov_b, check_exact=True)
    pd.testing.assert_frame_equal(opt, opt_b, check_exact=True)
    # resume
    (tmp_path / "one" / "parts" / "metrics" / "part_00002.parquet").unlink()
    r1c, r2c, _, opt_c = _run(manifest, tmp_path / "one", workers=2, chunk=8)
    assert (r1c["done"], r2c["done"], r2c["skipped"]) == (0, 1, 3)
    pd.testing.assert_frame_equal(opt, opt_c, check_exact=True)


def test_real_subset_report_and_overlays(real_subset, tmp_path):
    manifest, sub = real_subset
    _run(manifest, tmp_path, workers=2, chunk=8)
    rep = optics.g1_report(manifest, tmp_path / "fov.parquet", tmp_path / "optics.parquet", tmp_path / "parts",
                           tmp_path / "g1.json", 0.99, log=lambda *_: None)
    on_disk = json.loads((tmp_path / "g1.json").read_text())
    assert on_disk["n_frames"] == 30 and on_disk["found"]["pass_overall"] is True
    assert set(on_disk["metrics_by_domain"]) == {"D1", "D2", "D3"} and set(on_disk["metrics_by_scenario"]) <= {"Normal", "Difficult"}
    assert len(on_disk["worst_residual_frames"]) == 20 and rep["nonfinite"]["lum_mean"]["nan"] == 0
    files = optics.render_overlays(manifest, tmp_path / "fov.parquet", tmp_path / "qc", 1, per_domain=4, n_stream=2, n_worst=2,
                                   log=lambda *_: None)
    assert {f.name for f in files} == {"overlay_montage_D1.png", "overlay_montage_D2.png", "overlay_montage_D3.png",
                                       "overlay_montage_stream.png", "overlay_montage_worst.png"}
    sel = json.loads((tmp_path / "qc" / "overlay_selection.json").read_text())
    again = optics.select_overlay_frames(sub, pd.read_parquet(tmp_path / "fov.parquet"), 1, 4, 2, 2, 3)
    assert {k: v.frame_id.tolist() for k, v in again.items() if len(v)} == sel
    assert cv2.imread(str(files[0])) is not None
