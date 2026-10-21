import numpy as np
import pandas as pd
import pytest

from conftest import load_script

folds = load_script("03_folds.py")


def synthetic_manifest(n1=14, n2=7, n_ext=4, n_stream=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n1 + n2):
        dom = "D1" if i < n1 else "D2"
        rows += [dict(patient_id=f"P{i + 1}", center=1, domain=dom, partition="center1_labeled")] * int(rng.integers(5, 200))
    for i in range(n_ext):
        rows += [dict(patient_id=f"T{i + 1:03d}", center=2, domain="D3", partition="center2_labeled")] * int(rng.integers(3, 30))
    for i in range(n_stream):
        rows += [dict(patient_id=f"P{100 + i}", center=1, domain="D1", partition="stream")] * 50
    return pd.DataFrame(rows)


def test_folds_cover_all_patients_without_overlap():
    m = synthetic_manifest()
    t = folds.build_folds(m, 5, 123)
    assert len(t) == m.patient_id.nunique() and t.patient_id.is_unique
    cv = t[t.role == "cv"]
    assert len(cv) == 21 and set(cv.fold) == {0, 1, 2, 3, 4}
    assert (t[t.role == "external"].fold == -1).all() and (t[t.role == "stream"].fold == -2).all()
    expected = dict(labeled_patients=25, center1_patients=21, center2_patients=4, stream_patients=3,
                    labeled_frames=int(m[m.partition != "stream"].shape[0]), center1_frames=int((m.partition == "center1_labeled").sum()),
                    center2_frames=int((m.partition == "center2_labeled").sum()), stream_frames=150)
    assert folds.check(t, m, expected, 5) == expected


def test_domains_stratified_and_frames_balanced():
    m = synthetic_manifest()
    cv = folds.build_folds(m, 5, 7).query("role == 'cv'")
    per = cv.groupby(["fold", "domain"]).size().unstack(fill_value=0)
    assert per.D1.max() - per.D1.min() <= 1 and per.D2.max() - per.D2.min() <= 1
    frames = cv.groupby("fold").n_frames.sum()
    assert frames.max() - frames.min() <= 0.1 * frames.mean()


def test_deterministic_and_seed_dependent():
    m = synthetic_manifest()
    a, b = folds.build_folds(m, 5, 99), folds.build_folds(m, 5, 99)
    pd.testing.assert_frame_equal(a, b)
    assert not folds.build_folds(m, 5, 100).fold.equals(a.fold)


def test_check_catches_bad_split():
    m = synthetic_manifest()
    t = folds.build_folds(m, 5, 1)
    expected = dict(labeled_patients=25, center1_patients=21, center2_patients=4, stream_patients=3, labeled_frames=0,
                    center1_frames=0, center2_frames=0, stream_frames=0)
    with pytest.raises(AssertionError, match="count mismatch"):
        folds.check(t, m, expected, 5)
    dup = pd.concat([t, t.iloc[[0]]])
    with pytest.raises(AssertionError):
        folds.check(dup, m, expected, 5)


def test_mixed_resolution_patient_rejected():
    m = synthetic_manifest()
    m.loc[m.index[0], "domain"] = "D2"
    with pytest.raises(AssertionError, match="mixed resolutions"):
        folds.patient_table(m)
