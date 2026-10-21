"""Tests for infer.py and the 05/06/07 scripts.

Unit tests need no weights. The smoke tests use work/models/{unet,segformer}_fold0_pilot.pt when they exist and
otherwise randomly initialised weights written to a temp dir (the real models directory is only read, never
written); `weights_note` says which. Smoke output goes to work/smoke/. Nothing here writes derived/*.parquet.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import dataset
import fov
import infer
import optics_lib
import perturb_ops
from common import abspath, load_config, read_image
from conftest import ROOT, V1, load_script
from models import MODEL_NAMES, build_model

predict = load_script("05_predict.py")
perturb = load_script("06_perturb.py")
stream = load_script("07_stream.py")
train = load_script("04_train.py")

SMOKE = V1 / "work" / "smoke"
FOLDS = pd.read_csv(infer.DERIVED / "folds.csv") if (infer.DERIVED / "folds.csv").exists() else None
needs_data = pytest.mark.skipif(
    not all((infer.DERIVED / f).exists() for f in ("manifest.parquet", "fov.parquet", "optics.parquet", "folds.csv")),
    reason="derived/ tables missing")
needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
ATLAS = ROOT / "Spine endoscopic atlas"
needs_images = pytest.mark.skipif(not ATLAS.exists(), reason="dataset not present")


# ---------------------------------------------------------------- helpers

def contract_section(name):
    text = (V1 / "config" / "CONTRACTS.md").read_text()
    return text.split(f"## derived/{name}")[1].split("\n## ")[0]


def split_top_level(s):
    """Split on commas that are not inside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return out + [cur.strip()]


def contract_columns(name):
    """Column list of predictions.parquet / perturb.parquet / stream.parquet as CONTRACTS.md states it."""
    sec = contract_section(name)
    metrics = list(infer.METRICS)
    if name.startswith("predictions") or name.startswith("perturb"):
        line = next(ln for ln in sec.splitlines() if ln.lstrip("`").startswith("frame_id"))
        cols = []
        for tok in split_top_level(line.strip("`").strip()):
            word = re.match(r"[^\s(]+", tok).group(0)
            if word.startswith("unet_bg_<"):
                cols += [f"unet_bg_{m}" for m in metrics]
            elif word.startswith("<11"):
                cols += metrics
            else:
                cols.append(word)
        return cols
    base = contract_columns("predictions.parquet")
    groups = re.findall(r"`([^`]+)`", next(ln for ln in sec.splitlines() if ln.startswith("Same as predictions")))
    minus = [re.match(r"[^\s(]+", t).group(0) for t in split_top_level(groups[0])]
    plus = [re.match(r"[^\s(]+", t).group(0) for g in groups[1:] for t in split_top_level(g)]
    return [c for c in base if c not in minus] + plus


@pytest.fixture(scope="module")
def pilot_dir(tmp_path_factory):
    """(dir with {name}_fold0_pilot.pt, names whose weights are the real pilot ones). Missing ones are random-init."""
    d = tmp_path_factory.mktemp("weights")
    real = []
    for name in MODEL_NAMES:
        src = infer.weights_path(name, "fold0_pilot")
        dst = d / src.name
        if src.exists():
            dst.symlink_to(src)
            real.append(name)
        else:
            torch.manual_seed(0)
            torch.save(build_model(name, pretrained=False).state_dict(), dst)
    return d, real


def weights_note(real):
    return "real pilot weights: " + (", ".join(real) or "none") + "; random-init: " + (", ".join(set(MODEL_NAMES) - set(real)) or "none")


def fold0_frames(n, seed=0):
    t = dataset.frame_table("center1_labeled")
    t = t[t.fold == 0].sample(n=n, random_state=seed).sort_values("frame_id").reset_index(drop=True)
    return infer.assign_models(t, pilot=True)


def disk_blob(size, radius, cy=None, cx=None):
    c = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    return (xx - (c if cx is None else cx)) ** 2 + (yy - (c if cy is None else cy)) ** 2 <= radius ** 2


# ---------------------------------------------------------------- metric definitions on synthetic masks

def test_binary_entropy_bits():
    h = infer.binary_entropy_bits(np.array([0.0, 1.0, 0.5, 0.25, 0.9, 1e-30], dtype=np.float32))
    assert h[0] == pytest.approx(0, abs=1e-9) and h[1] == pytest.approx(0, abs=1e-9)
    assert h[2] == pytest.approx(1.0) and h[3] == pytest.approx(0.811278, abs=1e-5)
    assert h[4] == pytest.approx(0.468996, abs=1e-5) and np.isfinite(h).all()
    assert infer.binary_entropy_bits(np.array([0.3]))[0] == pytest.approx(infer.binary_entropy_bits(np.array([0.7]))[0])


def test_overlap_cases():
    region = np.ones((20, 20), bool)
    gt = np.zeros((20, 20), bool)
    gt[2:12, 2:12] = True                                               # 100 px
    assert infer.overlap(gt, gt, region) == (1.0, 1.0)                    # full overlap
    empty = np.zeros_like(gt)
    assert infer.overlap(empty, gt, region) == (0.0, 0.0)                 # empty prediction
    assert infer.overlap(empty, empty, region) == (1.0, 1.0)              # both empty
    half = np.zeros_like(gt)
    half[2:12, 2:7] = True                                              # 50 px inside gt
    iou, dice = infer.overlap(half, gt, region)
    assert iou == pytest.approx(0.5) and dice == pytest.approx(2 * 50 / 150)
    shifted = np.zeros_like(gt)
    shifted[7:17, 7:17] = True                                          # 25 px overlap, union 175
    iou, dice = infer.overlap(shifted, gt, region)
    assert iou == pytest.approx(25 / 175) and dice == pytest.approx(50 / 200)
    assert infer.overlap(gt, ~gt, region) == (0.0, 0.0)                   # disjoint


def test_overlap_counts_only_pixels_inside_the_region():
    gt = np.zeros((20, 20), bool)
    pred = np.zeros((20, 20), bool)
    gt[:, :10] = True
    pred[:, :10] = True
    pred[:, 15:] = True                                                 # false positive outside the region
    region = np.zeros((20, 20), bool)
    region[:, :14] = True
    assert infer.overlap(pred, gt, region) == (1.0, 1.0)
    region[:] = True
    assert infer.overlap(pred, gt, region)[0] == pytest.approx(200 / 300)
    # a mask that lies entirely outside the region counts as empty
    out = np.zeros((20, 20), bool)
    out[:, 16:] = True
    region[:, 14:] = False
    assert infer.overlap(out, np.zeros((20, 20), bool), region) == (1.0, 1.0)


def test_ens_dis_cases():
    region = np.ones((16, 16), bool)
    a = np.zeros((16, 16), bool)
    a[:8] = True
    b = np.zeros((16, 16), bool)
    b[8:] = True
    assert infer.ens_dis(a, a, region) == 0.0                           # identical masks
    assert infer.ens_dis(a, b, region) == 1.0                           # disjoint
    assert infer.ens_dis(np.zeros_like(a), np.zeros_like(a), region) == 0.0   # both empty
    assert infer.ens_dis(a, np.zeros_like(a), region) == 1.0            # one empty
    c = np.zeros((16, 16), bool)
    c[:8, :4] = True                                                    # half of a
    assert infer.ens_dis(a, c, region) == pytest.approx(1 - 2 * 32 / (128 + 32))
    assert infer.ens_dis(a, c, region) == infer.ens_dis(c, a, region)   # symmetric


def test_roi_entropy_dilation_radius_and_empty_fallback():
    region = np.ones((100, 100), bool)
    pred = np.zeros((100, 100), bool)
    pred[50, 50] = True
    ent = np.zeros((100, 100), np.float32)
    ent[50, 50 + 15] = 1.0                                              # exactly 15 px away: inside the ROI
    near = infer.roi_entropy(ent, pred, region)
    ent[:] = 0
    ent[50, 50 + 20] = 1.0                                              # 20 px away: outside
    far = infer.roi_entropy(ent, pred, region)
    kernel_area = int(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)).sum())
    assert near == pytest.approx(1.0 / kernel_area) and far == 0.0
    assert infer.roi_entropy(ent, np.zeros_like(pred), region) is None  # empty prediction -> caller falls back


def test_seg_stats_native_grid_and_empty_prediction_fallback():
    side = 700                                                          # native crop larger than 384
    gt = disk_blob(side, 120, cy=330, cx=360)
    prob_gt = cv2.resize(gt.astype(np.float32), (384, 384), interpolation=cv2.INTER_AREA)
    prob = np.where(prob_gt > 0.5, 0.97, 0.03).astype(np.float32)
    row, pred384, pred_native, fallback = infer.seg_stats(prob, side, gt)
    assert pred_native.shape == (side, side) and pred384.shape == (384, 384) and not fallback
    assert row["iou"] > 0.95 and row["dice"] > row["iou"]
    dn = infer.disk(side)
    assert row["gt_area_frac"] == pytest.approx((gt & dn).sum() / dn.sum())
    assert row["pred_area_frac"] == pytest.approx(row["gt_area_frac"], abs=0.01)
    assert 0 < row["entropy_roi"] < 1 and 0 < row["entropy_fov"] < 1
    # empty prediction: IoU 0, area 0, entropy_roi falls back to entropy_fov and says so
    row0, p384, pnat, fb = infer.seg_stats(np.full((384, 384), 0.02, np.float32), side, gt)
    assert row0["iou"] == 0.0 and row0["dice"] == 0.0 and row0["pred_area_frac"] == 0.0 and fb and not pnat.any()
    assert row0["entropy_roi"] == row0["entropy_fov"]
    # uncertain pixels sit around the object, so the ROI average exceeds the whole-FOV average
    band = np.where(disk_blob(384, 60), 0.99, 0.01).astype(np.float32)
    band[disk_blob(384, 70) & ~disk_blob(384, 60)] = 0.5
    r, *_ = infer.seg_stats(band, 400)
    assert r["entropy_roi"] > r["entropy_fov"]
    # both empty
    both, *_ = infer.seg_stats(np.zeros((384, 384), np.float32), side, np.zeros((side, side), bool))
    assert both["iou"] == 1.0 and both["dice"] == 1.0


def test_grid_stats_matches_seg_stats_when_the_grids_agree():
    gt = disk_blob(384, 90, cy=170, cx=200)
    prob = np.where(disk_blob(384, 85, cy=172, cx=198), 0.9, 0.1).astype(np.float32)
    g = infer.grid_stats(prob, gt)
    s, *_ = infer.seg_stats(prob, 384, gt)
    for k in ("iou", "dice", "pred_area_frac", "entropy_fov"):
        assert g[k] == pytest.approx(s[k])


def test_bg_metrics_with_empty_prediction_equals_the_whole_fov_metrics():
    rng = np.random.default_rng(1)
    canvas = rng.random((512, 512, 3), dtype=np.float32)
    whole = optics_lib.compute_metrics(canvas, infer.disk(512))
    bg = infer.bg_metrics(canvas, np.zeros((600, 600), bool))
    assert list(bg) == infer.BG_COLS
    for m in infer.METRICS:
        assert bg[f"unet_bg_{m}"] == pytest.approx(whole[m], rel=1e-9, nan_ok=True)
    full = infer.bg_metrics(canvas, np.ones((600, 600), bool))             # nothing left: all NaN
    assert all(np.isnan(v) for v in full.values())
    part = np.zeros((512, 512), bool)
    part[:, :256] = True
    half = infer.bg_metrics(canvas, part)
    assert half["unet_bg_lum_mean"] != whole["lum_mean"]


def test_guard_output_refuses_smoke_runs_at_the_real_path(tmp_path):
    real = infer.DERIVED / "predictions.parquet"
    for kw in (dict(pilot=True, limit=None), dict(pilot=False, limit=5)):
        with pytest.raises(SystemExit):
            infer.guard_output(real, real, **kw)
        with pytest.raises(SystemExit):
            infer.guard_output(infer.DERIVED / "other.parquet", real, **kw)
        with pytest.raises(SystemExit):                                        # the logs directory holds the real summaries
            infer.guard_output(infer.LOGS_DIR / "predictions.parquet", real, **kw)
    assert infer.guard_output(tmp_path / "x.parquet", real, True, 5) == (tmp_path / "x.parquet").resolve()
    assert infer.guard_output(real, real, False, None) == real.resolve()


def test_smoke_paths_sit_next_to_out_and_never_in_the_real_run_directories(tmp_path, monkeypatch):
    parts_root, logs = tmp_path / "work_parts", tmp_path / "work_logs"
    monkeypatch.setattr(infer, "PARTS_ROOT", parts_root)
    monkeypatch.setattr(infer, "LOGS_DIR", logs)
    out = tmp_path / "smoke" / "predictions.parquet"
    assert infer.default_parts(out, smoke=True) == out.parent / "predictions_parts"
    assert infer.default_parts(out) == parts_root / "predictions"                  # real-run location
    assert infer.write_summary(out, smoke=True, a=1) == out.parent / "predictions_summary.json" and not logs.exists()
    assert infer.write_summary(out, a=1) == logs / "predictions_summary.json"
    for inside in (parts_root / "predictions", parts_root, logs / "x.csv", logs):  # an explicit --parts / --alloc there
        with pytest.raises(SystemExit):
            infer.guard_side_path(inside, True, None)
        with pytest.raises(SystemExit):
            infer.guard_side_path(inside, False, 5)
        assert infer.guard_side_path(inside, False, None) == inside.resolve()
    assert infer.guard_side_path(tmp_path / "smoke" / "parts", True, 5) == (tmp_path / "smoke" / "parts").resolve()


def test_constants_come_from_the_plan_yaml():
    cfg = load_config()
    assert infer.THRESHOLD == cfg["segmenters"]["mask_threshold"] == 0.5
    assert infer.ROI_DILATE_PX == cfg["entropy"]["roi_dilate_px"] == 15
    assert perturb.MIN_IOU == cfg["perturbation"]["min_oof_iou"] == 0.90


def post_inputs(side=600):
    """Two overlapping but different 384 maps, a native GT that differs from both, and a random canvas."""
    gt = disk_blob(side, 150, cy=250, cx=250)
    prob_u = np.where(disk_blob(384, 90, cy=190, cx=200), 0.93, 0.05).astype(np.float32)
    prob_s = np.where(disk_blob(384, 70, cy=170, cx=170), 0.88, 0.08).astype(np.float32)
    return prob_u, prob_s, np.random.default_rng(3).random((512, 512, 3), dtype=np.float32), gt, side


def test_stored_ens_dis_is_one_minus_dice_of_the_two_actual_masks():
    prob_u, prob_s, canvas, gt, side = post_inputs()
    rows, _ = infer._post_one(prob_u, prob_s, canvas, gt, side, {"frame_id": "f"})
    d = fov.fov_disk_mask(384, fov.ERODE_FRAC)                                  # recomputed without infer.overlap / ens_dis
    a, b = (prob_u > 0.5) & d, (prob_s > 0.5) & d
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())
    assert 0.05 < 1 - dice < 0.95                                               # partial overlap: not a degenerate case
    assert rows[0]["ens_dis"] == rows[1]["ens_dis"] == pytest.approx(1 - dice, abs=1e-12)
    assert [r["model"] for r in rows] == ["unet", "segformer"]
    empty = np.zeros((384, 384), np.float32)
    assert infer._post_one(empty, empty, canvas, gt, side, {})[0][0]["ens_dis"] == 0.0         # both masks empty
    full = np.ones((384, 384), np.float32)
    disjoint = np.where(disk_blob(384, 60, cy=100, cx=100), 0.9, 0.1).astype(np.float32)
    far = np.where(disk_blob(384, 60, cy=290, cx=290), 0.9, 0.1).astype(np.float32)
    assert infer._post_one(disjoint, far, canvas, gt, side, {})[0][0]["ens_dis"] == 1.0
    assert infer._post_one(full, full, canvas, gt, side, {})[0][0]["ens_dis"] == 0.0


def test_unet_bg_uses_the_unet_predicted_mask_not_the_ground_truth(monkeypatch):
    prob_u, prob_s, canvas, gt, side = post_inputs()
    seen, real_bg = [], infer.bg_metrics
    monkeypatch.setattr(infer, "bg_metrics", lambda c, m: (seen.append(m.copy()), real_bg(c, m))[1])
    rows, _ = infer._post_one(prob_u, prob_s, canvas, gt, side, {"frame_id": "f"})
    pred_u = cv2.resize(prob_u, (side, side), interpolation=cv2.INTER_LINEAR) > 0.5
    pred_s = cv2.resize(prob_s, (side, side), interpolation=cv2.INTER_LINEAR) > 0.5
    assert len(seen) == 1 and np.array_equal(seen[0], pred_u)
    assert not np.array_equal(seen[0], gt) and not np.array_equal(seen[0], pred_s)

    def bg_values(mask):
        bg = ~cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
        return optics_lib.compute_metrics(canvas, fov.fov_disk_mask(512, fov.ERODE_FRAC), bg)
    want, from_gt = bg_values(pred_u), bg_values(gt)
    for r in rows:
        for m in infer.METRICS:
            assert r[f"unet_bg_{m}"] == pytest.approx(want[m], rel=1e-9, nan_ok=True), m
    assert any(abs(want[m] - from_gt[m]) > 1e-9 for m in infer.METRICS)         # the GT would have given other numbers


# ---------------------------------------------------------------- model selection (real folds.csv)

def tables_from_folds():
    f = FOLDS.copy()
    part = {"cv": "center1_labeled", "external": "center2_labeled", "stream": "stream"}
    return pd.DataFrame({"frame_id": f.patient_id + "-x", "patient_id": f.patient_id, "partition": f.role.map(part), "fold": f.fold})


@needs_data
def test_trained_on_follows_the_fold_membership():
    cv = FOLDS[FOLDS.role == "cv"]
    for k in range(5):
        seen = infer.trained_on(f"fold{k}", FOLDS)
        held = set(cv[cv.fold == k].patient_id)
        assert not seen & held and seen | held == set(cv.patient_id)
        assert infer.trained_on(f"fold{k}_pilot", FOLDS) == seen
    assert infer.trained_on("final", FOLDS) == set(cv.patient_id)
    assert not infer.trained_on("final", FOLDS) & set(FOLDS[FOLDS.role != "cv"].patient_id)
    with pytest.raises(ValueError):
        infer.trained_on("fold7", FOLDS)


@needs_data
def test_check_selection_accepts_the_planned_assignment_and_rejects_leaks():
    t = infer.assign_models(tables_from_folds())
    infer.check_selection(t, FOLDS)
    assert set(t.tag) == {f"fold{k}" for k in range(5)} | {"final"}
    assert (t[t.partition == "center1_labeled"].tag == "fold" + t[t.partition == "center1_labeled"].fold.astype(str)).all()
    assert set(t[t.partition != "center1_labeled"].source) == {"final"}
    # a Center 1 patient scored by another fold's model, or by the final model, is a leak
    i = t.index[t.partition == "center1_labeled"][0]
    for bad in ("fold%d" % ((int(t.fold[i]) + 1) % 5), "final"):
        t2 = t.copy()
        t2.loc[i, "tag"] = bad
        with pytest.raises(AssertionError, match="trained on their patient"):
            infer.check_selection(t2, FOLDS)
    # Center 2 / stream with a fold model is not a leak but is not the plan either
    j = t.index[t.partition == "stream"][0]
    t3 = t.copy()
    t3.loc[j, "tag"] = "fold0"
    with pytest.raises(AssertionError, match="final model"):
        infer.check_selection(t3, FOLDS)
    # a wrong fold column is caught too
    t4 = t.copy()
    t4.loc[i, "fold"] = (int(t.fold[i]) + 1) % 5
    t4.loc[i, "tag"] = "fold%d" % t4.fold[i]
    with pytest.raises(AssertionError):
        infer.check_selection(t4, FOLDS)


@needs_data
def test_pilot_assignment_only_passes_for_patients_the_pilot_did_not_train_on():
    t = tables_from_folds()
    ok = t[~t.patient_id.isin(infer.trained_on("fold0_pilot", FOLDS))]
    infer.check_selection(infer.assign_models(ok, pilot=True), FOLDS)
    assert set(ok[ok.partition == "center1_labeled"].fold) == {0}
    with pytest.raises(AssertionError, match="trained on their patient"):
        infer.check_selection(infer.assign_models(t, pilot=True), FOLDS)


@needs_data
def test_real_tables_pass_the_selection_check_and_the_exclusion_counts():
    table, dropped = predict.build_table()
    assert dropped == {"center1_labeled": 3, "center2_labeled": 0}
    assert len(table) == 10659 and table.frame_id.is_unique
    assert (table.partition == "center1_labeled").sum() == 9590 and (table.partition == "center2_labeled").sum() == 1069
    assert not table.frame_id.isin(["P5-9-1066", "P49-1-0019", "P57-1-0016"]).any()
    infer.check_selection(table, FOLDS)
    c1 = table[table.partition == "center1_labeled"]
    assert (c1.groupby("patient_id").tag.nunique() == 1).all() and c1.source.eq("oof").all()
    c2 = table[table.partition == "center2_labeled"]
    assert c2.tag.eq("final").all() and c2.fold.eq(-1).all() and c2.source.eq("final").all()
    st, dropped_st = stream.build_table()
    assert dropped_st == 25 and len(st) == 37823 and st.tag.eq("final").all() and st.fold.eq(-2).all()
    infer.check_selection(st, FOLDS)


@needs_data
def test_stream_t_index_is_consecutive_per_video_in_frame_order():
    st, _ = stream.build_table()
    assert st.patient_id.nunique() == 20 and (st.video_patient == st.patient_id).all()
    for p, g in st.groupby("patient_id"):
        assert g.t_index.tolist() == list(range(len(g))) and g.frame_idx.is_monotonic_increasing, p
    assert st.t_index.dtype == np.int32
    small, _ = stream.build_table(pilot=True, limit=30)
    assert len(small) == 30 and (small.tag == "fold0_pilot").all()


# ---------------------------------------------------------------- perturbation selection (06)

def make_cells(counts):
    rows = []
    for (d, s, ins), n in counts.items():
        rows += [dict(domain=d, scenario=s, instrument=ins, patient_id=f"{d}{s}{ins}{i % 7}", frame_id=f"{d}{s}{ins}{i}") for i in range(n)]
    return pd.DataFrame(rows)


def test_allocate_equal_share_with_nearest_fill_and_remainder():
    rng = np.random.default_rng(0)
    avail = {("D1", "N", "a"): 100, ("D1", "N", "b"): 100, ("D1", "S", "a"): 100, ("D2", "N", "a"): 2, ("D2", "S", "a"): 100}
    taken = perturb.allocate(avail, 52, rng)
    assert sum(taken.values()) == 52 and all(taken[c] <= avail[c] for c in avail)
    assert taken[("D2", "N", "a")] == 2                                   # short cell gives everything
    # ask = 52 // 5 = 10; the 8-frame shortfall goes to the nearest cell (same domain, other scenario)
    assert taken[("D2", "S", "a")] == 18
    assert sorted(taken[c] for c in avail if c[0] == "D1") == [10, 11, 11]   # the remainder of 2 goes to the lowest
    again = perturb.allocate(avail, 52, np.random.default_rng(0))
    assert again == taken                                                # seeded
    assert perturb.allocate(avail, 1000, rng) == avail                   # not enough frames: take all


def test_select_frames_spreads_over_patients_and_reports_the_allocation():
    counts = {("D1", "N", "a"): 40, ("D1", "S", "a"): 40, ("D2", "N", "a"): 5, ("D2", "S", "a"): 40}
    el = make_cells(counts)
    ids, alloc = perturb.select_frames(el, 60, 3)
    assert len(ids) == 60 and len(set(ids)) == 60
    sel = el[el.frame_id.isin(ids)]
    assert list(alloc.columns) == ["domain", "scenario", "instrument", "available", "asked", "taken", "short_by", "extra"]
    assert alloc.taken.sum() == 60 and alloc[(alloc.domain == "D2") & (alloc.scenario == "N")].short_by.iloc[0] == 10
    assert sel[sel.domain == "D1"].groupby(["scenario"]).patient_id.nunique().min() == 7   # every patient represented
    ids2, _ = perturb.select_frames(el, 60, 3)
    assert ids2 == ids
    ids3, _ = perturb.select_frames(el, 60, 4)
    assert ids3 != ids
    few, alloc_few = perturb.select_frames(el, 500, 3)
    assert len(few) == len(el) and (alloc_few.taken == alloc_few.available).all()


# ---------------------------------------------------------------- loading workers and the canvas

@needs_images
@needs_data
def test_load_job_reproduces_the_training_crop_the_optics_canvas_and_the_native_gt():
    t = fold0_frames(2, seed=3)
    optics = pd.read_parquet(infer.DERIVED / "optics.parquet").set_index("frame_id")
    for r in t.itertuples():
        crop, canvas, gt, secs = infer.load_job((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, int(r.crop_side)))
        img, msk = dataset._build_one((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, infer.SIZE))
        assert np.array_equal(crop, img) and crop.dtype == np.uint8 and crop.shape == (384, 384, 3)
        assert canvas.shape == (512, 512, 3) and canvas.dtype == np.float32
        assert gt.shape == (r.crop_side, r.crop_side) and gt.dtype == bool and gt.any()
        assert fov.crop_box(r.fov_cx, r.fov_cy, r.fov_r)[2] == r.crop_side
        # the 384 training mask is the nearest-neighbour shrink of the native one: areas agree to a fraction of a percent
        assert abs(msk.mean() - gt.mean()) < 0.005
        got = optics_lib.compute_metrics(canvas, infer.disk(512))
        for m in infer.METRICS:
            assert got[m] == pytest.approx(optics.loc[r.frame_id, m], rel=1e-5, abs=1e-9)
        assert secs > 0


# ---------------------------------------------------------------- resumable chunks

def save_random_weights(directory, tag):
    for name in MODEL_NAMES:
        torch.manual_seed(1)
        torch.save(build_model(name, pretrained=False).state_dict(), Path(directory) / f"{name}_{tag}.pt")


@needs_images
@needs_data
def test_killed_run_resumes_and_matches_an_uninterrupted_run(tmp_path):
    save_random_weights(tmp_path, "fold0_pilot")
    table = fold0_frames(7, seed=5)
    kw = dict(chunk=3, log=lambda m: None)
    with infer.Engine("cpu", 0, tmp_path, batch=3) as eng:
        df, stats, new = infer.predict_table(table, True, eng, tmp_path / "a", max_new=1, **kw)    # "killed" after one chunk
        assert df is None and stats is None and new == 1
        assert len(list((tmp_path / "a").glob("*.parquet"))) == 1 and not list((tmp_path / "a").glob("*.tmp"))
        df, stats, new = infer.predict_table(table, True, eng, tmp_path / "a", **kw)
        assert new == 2 and len(df) == 14                                     # only the missing chunks ran
        again, _, new = infer.predict_table(table, True, eng, tmp_path / "a", **kw)
        assert new == 0
        straight, _, _ = infer.predict_table(table, True, eng, tmp_path / "b", **kw)
        with pytest.raises(RuntimeError, match="different run"):               # another plan in the same directory
            infer.predict_table(table, True, eng, tmp_path / "a", chunk=2, log=lambda m: None)
    pd.testing.assert_frame_equal(df, straight)
    pd.testing.assert_frame_equal(df, again)
    assert stats["n_frames"] == 7 and list(df.columns) == infer.PRED_COLS


@needs_images
@needs_data
def test_threads_and_worker_processes_do_not_change_the_numbers(tmp_path):
    save_random_weights(tmp_path, "fold0_pilot")
    table = fold0_frames(4, seed=11)
    out = {}
    for name, workers, threads in (("serial", 0, 1), ("threads", 0, 4), ("pool", 2, 4)):
        with infer.Engine("cpu", workers, tmp_path, batch=3, threads=threads) as eng:
            out[name], _, _ = infer.predict_table(table, True, eng, tmp_path / name, chunk=4, log=lambda m: None)
    pd.testing.assert_frame_equal(out["serial"], out["threads"])
    pd.testing.assert_frame_equal(out["serial"], out["pool"])


@needs_images
@needs_data
def test_retrained_weights_invalidate_finished_chunks(tmp_path):
    save_random_weights(tmp_path, "fold0_pilot")
    table = fold0_frames(2, seed=9)
    with infer.Engine("cpu", 0, tmp_path, batch=2) as eng:
        first, _, new = infer.predict_table(table, True, eng, tmp_path / "p", chunk=2, log=lambda m: None)
        assert new == 1
        torch.manual_seed(99)                                               # "retrain": same name, different weights
        torch.save(build_model("unet", pretrained=False).state_dict(), tmp_path / "unet_fold0_pilot.pt")
        with pytest.raises(RuntimeError, match="different run"):
            infer.predict_table(table, True, eng, tmp_path / "p", chunk=2, log=lambda m: None)
        eng._tag = None                                                     # drop the model cache of the old weights
        second, _, new = infer.predict_table(table, True, eng, tmp_path / "p", chunk=2, restart=True, log=lambda m: None)
    assert new == 1 and not np.allclose(first[first.model == "unet"].entropy_fov, second[second.model == "unet"].entropy_fov)


def test_missing_weights_stop_the_run_before_any_chunk(tmp_path):
    table = infer.assign_models(tables_from_folds().query("partition == 'center2_labeled'").assign(
        image_path="x", mask_path="x", fov_cx=0.0, fov_cy=0.0, fov_r=1.0, crop_side=2))
    torch.save({}, tmp_path / "unet_final.pt")
    with infer.Engine("cpu", 0, tmp_path) as eng:
        with pytest.raises(FileNotFoundError, match="segformer_final.pt"):
            infer.predict_table(table, True, eng, tmp_path / "parts", log=lambda m: None)
    assert not (tmp_path / "parts").exists()


def test_run_chunks_restart_discards_other_plans(tmp_path):
    calls = []

    def work(p):
        calls.append(p)
        return pd.DataFrame({"x": [p]}), {"n_frames": 1}
    infer.run_chunks([("c0", 0), ("c1", 1)], work, tmp_path, {"k": 1}, log=lambda m: None)
    infer.run_chunks([("c0", 0), ("c1", 1)], work, tmp_path, {"k": 1}, log=lambda m: None)
    assert calls == [0, 1]
    with pytest.raises(RuntimeError):
        infer.run_chunks([("c0", 0)], work, tmp_path, {"k": 2}, log=lambda m: None)
    infer.run_chunks([("c0", 0)], work, tmp_path, {"k": 2}, restart=True, log=lambda m: None)
    assert calls == [0, 1, 0]
    df, tot = infer.read_parts(tmp_path, ["c0"])
    assert df.x.tolist() == [0] and tot["n_frames"] == 1
    with pytest.raises(RuntimeError, match="not finished"):
        infer.read_parts(tmp_path, ["c0", "c1"])


def test_code_fingerprint_covers_the_four_files_and_tracks_their_content(tmp_path, monkeypatch):
    assert [p.name for p in infer.CODE_FILES] == ["infer.py", "optics_lib.py", "fov.py", "fov.parquet"]
    assert all(p.exists() for p in infer.CODE_FILES[:3])
    fp = infer.code_fingerprint()
    assert set(fp) == {"infer.py", "optics_lib.py", "fov.py", "fov.parquet"}
    assert fp["infer.py"] == hashlib.sha256(Path(infer.__file__).read_bytes()).hexdigest()[:16]
    extra = tmp_path / "perturb_ops.py"
    extra.write_text("x = 1")
    assert set(infer.code_fingerprint(extra=(extra,))) == set(fp) | {"perturb_ops.py"}


@needs_data
def test_changed_code_or_fov_geometry_invalidates_finished_parts(tmp_path, monkeypatch):
    stand_ins = []
    for p in infer.CODE_FILES:                                                  # small stand-ins so the test can edit them
        q = tmp_path / "code" / p.name
        q.parent.mkdir(exist_ok=True)
        q.write_bytes(b"v1")
        stand_ins.append(q)
    monkeypatch.setattr(infer, "CODE_FILES", tuple(stand_ins))
    for name in MODEL_NAMES:                                                    # only hashed and checked for existence: max_new=0 never runs a chunk
        (tmp_path / f"{name}_fold0_pilot.pt").write_bytes(b"w")
    t = tables_from_folds().query("partition == 'center1_labeled'")
    table = infer.assign_models(t[t.fold == 0].reset_index(drop=True), pilot=True)
    kw = dict(chunk=5, max_new=0, log=lambda m: None)
    runs = {"predict": lambda parts, **more: infer.predict_table(table, True, eng, parts, **kw, **more),
            "perturb": lambda parts, **more: infer.perturb_table(table, eng, parts, **kw, **more)}
    with infer.Engine("cpu", 0, tmp_path) as eng:
        for kind, run in runs.items():
            parts = tmp_path / f"parts_{kind}"
            run(parts)
            plan = json.loads((parts / "plan.json").read_text())
            assert set(plan["code"]) == {"infer.py", "optics_lib.py", "fov.py", "fov.parquet"} | ({"perturb_ops.py"} if kind == "perturb" else set())
            run(parts)                                                          # same code: the plan is accepted
            for victim in stand_ins:                                            # any one of the four changing stales the parts
                victim.write_bytes(b"v2")
                with pytest.raises(RuntimeError, match="different run"):
                    run(parts)
                victim.write_bytes(b"v1")
                run(parts)
            stand_ins[3].write_bytes(b"v2")                                     # fov.parquet re-detected
            with pytest.raises(RuntimeError, match="different run"):
                run(parts)
            run(parts, restart=True)                                            # --restart accepts the new geometry
            assert json.loads((parts / "plan.json").read_text())["code"]["fov.parquet"] == hashlib.sha256(b"v2").hexdigest()[:16]
            stand_ins[3].write_bytes(b"v1")


# ---------------------------------------------------------------- smoke: pilot weights, schema, GPU parity

@needs_images
@needs_data
def test_predictions_smoke_matches_the_contract_schema(pilot_dir):
    d, real = pilot_dir
    SMOKE.mkdir(parents=True, exist_ok=True)
    table = fold0_frames(40, seed=1)
    with infer.Engine(infer.pick_device(), 2, d) as eng:
        df, stats, _ = infer.predict_table(table, True, eng, SMOKE / "parts_test_predict", chunk=20, restart=True, log=lambda m: None)
    df.to_parquet(SMOKE / "predictions_smoke.parquet", index=False)
    assert list(df.columns) == contract_columns("predictions.parquet") == infer.PRED_COLS
    assert len(df) == 80 and set(df.model) == {"unet", "segformer"} and df.groupby("frame_id").size().eq(2).all()
    assert (df.source == "oof").all() and (df.fold == 0).all() and (df.partition == "center1_labeled").all()
    assert df.fold.dtype == np.int32
    for c in ("iou", "dice", "pred_area_frac", "gt_area_frac", "entropy_fov", "entropy_roi", "ens_dis"):
        assert df[c].between(0, 1).all(), c
    assert (df.gt_area_frac > 0).all() and (df.dice >= df.iou - 1e-12).all()
    g = df.groupby("frame_id")
    assert (g.ens_dis.nunique() == 1).all()                                  # same value on both model rows
    assert all((g[c].nunique(dropna=False) == 1).all() for c in infer.BG_COLS)
    assert df.iou.notna().all() and df.entropy_roi.notna().all()
    assert stats["n_frames"] == 40 and stats["bg_nan"] <= 1
    if "unet" in real:
        assert df[df.model == "unet"].iou.median() > 0.8, weights_note(real)
    print(weights_note(real), "smoke written to", SMOKE / "predictions_smoke.parquet")


@needs_images
@needs_data
def test_stream_and_center2_smoke_columns(pilot_dir):
    d, real = pilot_dir
    st, _ = stream.build_table(pilot=True, limit=6)
    c2 = dataset.frame_table("center2_labeled").sample(n=4, random_state=0).sort_values("frame_id").reset_index(drop=True)
    c2 = infer.assign_models(c2, pilot=True)
    with infer.Engine(infer.pick_device(), 2, d) as eng:
        s, _, _ = infer.predict_table(st, False, eng, SMOKE / "parts_test_stream", chunk=6, restart=True, log=lambda m: None)
        c, _, _ = infer.predict_table(c2, True, eng, SMOKE / "parts_test_c2", chunk=4, restart=True, log=lambda m: None)
    s = s.merge(st[["frame_id", "video_patient", "t_index"]], on="frame_id")
    assert list(s.columns) == contract_columns("stream.parquet") == infer.STREAM_COLS
    assert len(s) == 12 and not {"iou", "dice", "gt_area_frac"} & set(s.columns)
    assert (s.partition == "stream").all() and (s.source == "final").all() and (s.fold == -2).all()
    assert list(c.columns) == infer.PRED_COLS and (c.fold == -1).all() and (c.source == "final").all()
    assert (c.partition == "center2_labeled").all() and c.iou.between(0, 1).all()


@needs_images
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
@needs_data
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_mps_forward_equals_cpu_forward(pilot_dir, name):
    d, real = pilot_dir
    table = fold0_frames(8, seed=2)
    crops = np.stack([infer.load_job((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, int(r.crop_side)))[0] for r in table.itertuples()])
    outs = {}
    for dev in ("cpu", "mps"):
        m = infer.load_models("fold0_pilot", dev, d)[name]
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div_(255).contiguous().to(dev)
        with torch.inference_mode():
            outs[dev] = m(x).float().cpu()
    diff = (outs["cpu"] - outs["mps"]).abs().max().item()
    rtol = 0 if name in real else 1e-3          # random-init logits are large, so only trained weights get the strict absolute bound
    assert torch.allclose(outs["cpu"], outs["mps"], atol=1e-3, rtol=rtol), (
        f"{name} ({'real' if name in real else 'random-init'} weights): max logit difference {diff}")


@needs_images
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
@needs_data
def test_iou_from_mps_logits_equals_iou_from_cpu_logits(pilot_dir):
    d, real = pilot_dir
    table = fold0_frames(10, seed=4)
    loaded = [infer.load_job((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, int(r.crop_side))) for r in table.itertuples()]
    crops = np.stack([x[0] for x in loaded])
    ious = {}
    for dev in ("cpu", "mps"):
        probs = infer.forward_probs(infer.load_models("fold0_pilot", dev, d)["unet"], crops)
        ious[dev] = np.array([infer.seg_stats(p, x[2].shape[0], x[2])[0]["iou"] for p, x in zip(probs, loaded)])
    assert np.abs(ious["cpu"] - ious["mps"]).max() < 1e-3, (ious, weights_note(real))
    if "unet" in real:
        # independent check of the native-grid pipeline against the training metric on the 384 grid
        masks = np.stack([dataset._build_one((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, 384))[1] for r in table.itertuples()])
        probs = infer.forward_probs(infer.load_models("fold0_pilot", "cpu", d)["unet"], crops)
        iou384, _ = train.batch_metrics(torch.from_numpy(probs[:, None] - 0.5), torch.from_numpy(masks[:, None]).float(),
                                        torch.from_numpy(infer.disk(384).copy()))
        assert np.abs(iou384.numpy() - ious["cpu"]).max() < 0.05


@needs_images
@needs_data
def test_stored_ens_dis_matches_an_independent_forward_pass_and_dice(pilot_dir, tmp_path):
    d, real = pilot_dir
    table = fold0_frames(4, seed=8)
    with infer.Engine("cpu", 0, d, batch=4, threads=1) as eng:
        df, _, _ = infer.predict_table(table, True, eng, tmp_path / "parts", chunk=4, log=lambda m: None)
    crops = np.stack([dataset._build_one((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, 384))[0] for r in table.itertuples()])
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255).contiguous()
    masks = {}
    for name in MODEL_NAMES:                                                    # plain torch, no infer.forward_probs / load_models
        net = build_model(name, pretrained=False)
        net.load_state_dict(torch.load(d / f"{name}_fold0_pilot.pt", map_location="cpu", weights_only=True))
        with torch.inference_mode():
            masks[name] = (torch.sigmoid(net.eval()(x))[:, 0] > 0.5).numpy()
    disc = fov.fov_disk_mask(384, fov.ERODE_FRAC)
    want = []
    for i in range(len(table)):
        a, b = masks["unet"][i] & disc, masks["segformer"][i] & disc
        want.append(0.0 if a.sum() + b.sum() == 0 else 1 - 2 * (a & b).sum() / (a.sum() + b.sum()))
    got = df[df.model == "unet"].set_index("frame_id").ens_dis.loc[table.frame_id].to_numpy()
    assert np.allclose(got, want, atol=1e-9), (got, want)
    assert np.allclose(df[df.model == "segformer"].set_index("frame_id").ens_dis.loc[table.frame_id], want, atol=1e-9)
    if len(real) == 2:
        assert 0 < max(want) < 0.5, want                                        # real models disagree a little, not everywhere


@needs_images
@needs_mps
@needs_data
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_mps_parity_on_a_full_batch_of_32_plus_a_tail_of_4(pilot_dir, name):
    d, real = pilot_dir
    table = fold0_frames(36, seed=6)
    crops = np.stack([infer.load_job((r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r, int(r.crop_side)))[0] for r in table.itertuples()])
    probs = {}
    for dev in ("cpu", "mps"):
        m = infer.load_models("fold0_pilot", dev, d)[name]
        probs[dev] = np.concatenate([infer.forward_probs(m, crops[:infer.BATCH]), infer.forward_probs(m, crops[infer.BATCH:])])
        probs[dev + "_one"] = infer.forward_probs(m, crops)                    # the same 36 frames in a single batch
    assert infer.BATCH == 32 and probs["cpu"].shape == (36, 384, 384)
    why = f"{name} ({'real' if name in real else 'random-init'} weights)"
    for key in ("mps", "mps_one"):
        assert np.abs(probs["cpu"] - probs[key]).max() < 1e-3, why
        assert np.abs(probs["cpu"][32:] - probs[key][32:]).max() < 1e-3, f"tail of 4: {why}"
        assert ((probs["cpu"] > 0.5) != (probs[key] > 0.5)).mean() < 1e-3, why


def snapshot(*dirs):
    """{path: (mtime_ns, size)} of every file under the directories (missing directories are empty)."""
    return {str(f): (f.stat().st_mtime_ns, f.stat().st_size) for d in dirs for f in Path(d).rglob("*") if f.is_file()}


def run_main(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [mod.__name__ + ".py"] + [str(a) for a in argv])
    mod.main()


@needs_images
@needs_data
def test_main_of_each_script_in_smoke_mode_writes_next_to_out_and_leaves_real_run_files_alone(pilot_dir, tmp_path, monkeypatch):
    d, real = pilot_dir
    real_logs, real_parts = infer.LOGS_DIR, infer.PARTS_ROOT
    before = snapshot(real_logs, real_parts, infer.DERIVED)
    fake_logs, fake_parts = tmp_path / "work_logs", tmp_path / "work_parts"     # a stray write to the real-run locations would land here
    monkeypatch.setattr(infer, "LOGS_DIR", fake_logs)
    monkeypatch.setattr(infer, "PARTS_ROOT", fake_parts)
    out = tmp_path / "smoke"
    common = ["--pilot", "--workers", 2, "--device", "cpu", "--models-dir", d]

    run_main(predict, ["--limit", 10, "--chunk", 10, "--out", out / "predictions.parquet", *common], monkeypatch)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert list(pred.columns) == infer.PRED_COLS and len(pred) == 20 and pred.groupby("frame_id").size().eq(2).all()

    run_main(stream, ["--limit", 8, "--chunk", 8, "--out", out / "stream.parquet", *common], monkeypatch)
    st = pd.read_parquet(out / "stream.parquet")
    assert list(st.columns) == infer.STREAM_COLS and len(st) == 16 and st.t_index.dtype == np.int32

    # random-init pilot weights would leave no frame at IoU >= 0.90, so give the selection step eligible rows
    pred.assign(iou=0.95).to_parquet(out / "pred_for_perturb.parquet", index=False)
    run_main(perturb, ["--pred", out / "pred_for_perturb.parquet", "--limit", 2, "--chunk", 2,
                       "--out", out / "perturb.parquet", *common], monkeypatch)
    pt = pd.read_parquet(out / "perturb.parquet")
    assert list(pt.columns) == infer.PERTURB_COLS and len(pt) == 2 * 2 * 7 * 6

    for stem in ("predictions", "stream", "perturb"):
        assert (out / f"{stem}_parts" / "plan.json").exists()
        summary = json.loads((out / f"{stem}_summary.json").read_text())
        assert summary["pilot"] is True and summary["output"] == str(out / f"{stem}.parquet")
    assert (out / "perturb_allocation.csv").exists()
    assert not fake_logs.exists() and not fake_parts.exists()
    assert snapshot(real_logs, real_parts, infer.DERIVED) == before            # real-named files: nothing created, changed or wiped

    # a smoke run cannot be pointed at the real-run directories either, whatever --restart says
    for bad in (["--parts", fake_parts / "predictions"], ["--parts", fake_logs / "x"]):
        with pytest.raises(SystemExit):
            run_main(predict, ["--limit", 10, "--restart", "--out", out / "predictions.parquet", *common, *bad], monkeypatch)


# ---------------------------------------------------------------- perturbation run

def test_family_rng_is_deterministic_per_frame_and_family_and_layouts_nest():
    a = infer.family_rng("P1-1-0001", "glare").random(4)
    assert np.array_equal(a, infer.family_rng("P1-1-0001", "glare").random(4))
    assert not np.array_equal(a, infer.family_rng("P1-1-0002", "glare").random(4))
    assert not np.array_equal(a, infer.family_rng("P1-1-0001", "noise").random(4))
    canvas = np.full((512, 512, 3), 0.3, np.float32)
    full = fov.fov_disk_mask(512, 0.0)
    changed, sat = [], []
    for sev in range(1, 6):                                                # a fresh Generator per severity, as the run does
        out = perturb_ops.apply(canvas, full, "glare", sev, infer.family_rng("P1-1-0001", "glare"))
        changed.append((out != canvas).any(axis=2))
        sat.append((out >= 0.98).all(axis=2))
    for lo, hi in zip(changed, changed[1:]):
        assert (lo & ~hi).sum() == 0 and hi.sum() > lo.sum()               # glare footprint only grows
    for lo, hi in zip(sat, sat[1:]):
        assert (lo & ~hi).sum() == 0 and hi.sum() >= lo.sum()
    assert sat[0].any()
    base = perturb_ops.apply(canvas, full, "noise", 0, infer.family_rng("P1-1-0001", "noise"))
    n1 = perturb_ops.apply(canvas, full, "noise", 1, infer.family_rng("P1-1-0001", "noise")) - base
    n5 = perturb_ops.apply(canvas, full, "noise", 5, infer.family_rng("P1-1-0001", "noise")) - base
    inside = np.broadcast_to(full[..., None], n1.shape)
    assert np.corrcoef(n1[inside], n5[inside])[0, 1] > 0.9                  # same noise field, only the gain changes


def test_condition_order():
    assert infer.COND[0] == (None, 0) and len(infer.COND) == 1 + 7 * 5
    assert infer.COND[1] == ("defocus", 1) and infer.COND[5] == ("defocus", 5) and infer.COND[-1] == ("noise", 5)
    assert [f for f, _ in infer.COND[1:]][::5] == list(perturb_ops.FAMILIES)


@needs_images
@needs_data
def test_perturb_job_baseline_is_the_unperturbed_canvas_and_metrics_match_optics():
    r = next(fold0_frames(1, seed=7).itertuples())
    images, metrics, gt, secs = infer.perturb_job((r.frame_id, r.image_path, r.mask_path, r.fov_cx, r.fov_cy, r.fov_r))
    assert images.shape == (36, 384, 384, 3) and images.dtype == np.float32 and gt.shape == (384, 384) and gt.dtype == bool
    assert len(metrics) == 36 and images.min() >= 0 and images.max() <= 1
    optics = pd.read_parquet(infer.DERIVED / "optics.parquet").set_index("frame_id").loc[r.frame_id]
    for m in infer.METRICS:
        assert metrics[0][m] == pytest.approx(optics[m], rel=1e-5, abs=1e-9)   # severity 0 = the optics.parquet canvas path
    # the baseline image is the 512 canvas shrunk to 384 with area interpolation
    img = read_image(abspath(r.image_path))
    canvas, _ = fov.canonical_canvas(img, dict(fov_cx=r.fov_cx, fov_cy=r.fov_cy, fov_r=r.fov_r))
    assert np.array_equal(images[0], cv2.resize(canvas, (384, 384), interpolation=cv2.INTER_AREA))
    # severity 5 differs from the baseline for every family
    for k in range(1, 36):
        if infer.COND[k][1] == 5:
            assert np.abs(images[k] - images[0]).max() > 0.01, infer.COND[k]
    # outside the full FOV disk the perturbed canvases are untouched (checked on the 384 image away from the rim)
    outside = ~fov.fov_disk_mask(384, 0.0)
    outside = cv2.erode(outside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert outside.any() and all(np.array_equal(images[k][outside], images[0][outside]) for k in range(1, 36))


# metric that must move with severity per family, and its direction (T3 handoff)
EXPECT = {"defocus": ("sharp_tenengrad", -1), "haze": ("haze_dcp", +1), "exposure_under": ("lum_mean", -1),
          "exposure_over": ("sat_frac", +1), "blood_absorption": ("red_dom", +1), "glare": ("sat_frac", +1),
          "noise": ("sharp_tenengrad", +1)}


@needs_images
@needs_data
def test_perturbation_smoke_three_frames(pilot_dir):
    d, real = pilot_dir
    SMOKE.mkdir(parents=True, exist_ok=True)
    t = dataset.frame_table("center1_labeled")
    t = t[t.fold == 0]
    pick = [g.sample(n=1, random_state=2).iloc[0] for _, g in t.groupby("domain")][:3]
    rest = t[~t.frame_id.isin([p.frame_id for p in pick])].sample(n=3 - len(pick), random_state=10)
    frames = infer.assign_models(pd.concat([pd.DataFrame(pick), rest]).reset_index(drop=True), pilot=True)
    with infer.Engine(infer.pick_device(), 2, d, batch=18) as eng:
        df, stats, _ = infer.perturb_table(frames, eng, SMOKE / "parts_test_perturb", chunk=2, restart=True, log=lambda m: None)
    df.to_parquet(SMOKE / "perturb_smoke.parquet", index=False)
    assert list(df.columns) == contract_columns("perturb.parquet") == infer.PERTURB_COLS
    assert len(df) == 3 * 2 * 7 * 6 and df.severity.dtype == np.int32
    perturb.check_grid(df, frames)                                          # grid, baseline repetition, metric identity across models
    # param constant within family x severity and equal to the operator table
    assert df.groupby(["family", "severity"]).param.nunique().eq(1).all()
    spec = {(c["family"], c["severity"]): c["param"] for c in perturb_ops.CONDITIONS}
    assert all(spec[(r.family, r.severity)] == r.param for r in df.drop_duplicates(["family", "severity"]).itertuples())
    # severity 0: identical across the 7 family labels, one baseline per frame and model
    s0 = df[df.severity == 0]
    assert s0.groupby(["frame_id", "model"]).iou.nunique().eq(1).all() and len(s0) == 3 * 2 * 7
    # metric dose-response: the matching optical metric moves monotonically with severity
    for fam, (metric, sign) in EXPECT.items():
        for fid, g in df[(df.family == fam) & (df.model == "unet")].groupby("frame_id"):
            v = g.sort_values("severity")[metric].to_numpy() * sign
            assert np.all(np.diff(v) >= -1e-9) and v[-1] > v[0], (fam, metric, fid, v)
    for col in ("iou", "dice", "pred_area_frac", "entropy_fov"):
        assert df[col].between(0, 1).all(), col
    if "unet" in real:
        u = df[df.model == "unet"].groupby(["family", "severity"]).iou.mean().unstack()
        assert (u[0] > 0.6).all()
        assert (u[5] <= u[0] + 0.02).all(), u                                 # no family helps at severity 5
        assert (u[5] < u[0] - 0.02).sum() >= 3, u                             # several clearly hurt
    print(weights_note(real), "smoke written to", SMOKE / "perturb_smoke.parquet")


# ---------------------------------------------------------------- real-run guards of 06 (no smoke switches)

def test_real_perturb_run_needs_the_complete_predictions_file_and_400_frames():
    assert perturb.PRED_ROWS == 21318 == 2 * (10659)
    rows = pd.DataFrame(index=range(perturb.PRED_ROWS))
    perturb.check_real_pred(perturb.DEFAULT_PRED, rows)
    with pytest.raises(SystemExit, match="predictions.parquet"):
        perturb.check_real_pred(infer.DERIVED / "predictions_smoke.parquet", rows)
    with pytest.raises(SystemExit, match="21318"):
        perturb.check_real_pred(perturb.DEFAULT_PRED, rows.iloc[:-2])
    perturb.check_real_frames(400, 400)
    with pytest.raises(SystemExit, match="exactly 400"):
        perturb.check_real_frames(399, 400)


@needs_data
def test_real_perturb_run_stops_before_any_work_on_a_bad_input(tmp_path, monkeypatch):
    existed = (infer.DERIVED / "perturb.parquet").exists()

    def no_work(*a, **k):
        pytest.fail("work started")
    monkeypatch.setattr(infer, "Engine", no_work)
    monkeypatch.setattr(infer, "write_final", no_work)
    monkeypatch.setattr(infer, "LOGS_DIR", tmp_path / "logs")                   # the allocation CSV of a real run goes to LOGS_DIR
    monkeypatch.setattr(infer, "PARTS_ROOT", tmp_path / "parts")
    t = dataset.frame_table("center1_labeled")
    t = t[t.fold == 0].head(60)
    sel = pd.DataFrame({"frame_id": t.frame_id, "model": "unet", "source": "oof", "iou": 0.95})
    sel.to_parquet(tmp_path / "p.parquet", index=False)
    # a real run (no --pilot / --limit) given a file that is not derived/predictions.parquet
    monkeypatch.setattr(sys, "argv", ["06_perturb.py", "--pred", str(tmp_path / "p.parquet")])
    with pytest.raises(SystemExit, match="derived"):
        perturb.main()
    # the right path with the wrong number of rows
    monkeypatch.setattr(perturb, "DEFAULT_PRED", tmp_path / "p.parquet")
    monkeypatch.setattr(sys, "argv", ["06_perturb.py"])
    with pytest.raises(SystemExit, match="rows, expected 21318"):
        perturb.main()
    assert not (tmp_path / "logs").exists()                                     # failed before the allocation was written
    # right path and row count but only 60 frames qualify: the 400 are not reachable
    monkeypatch.setattr(perturb, "PRED_ROWS", len(sel))
    with pytest.raises(SystemExit, match="exactly 400"):
        perturb.main()
    assert (tmp_path / "logs" / "perturb_allocation.csv").exists()
    assert (infer.DERIVED / "perturb.parquet").exists() == existed                # the real output was not created by a refused run
