"""Patient-grouped 5-fold split of the Center 1 labeled patients, plus the external and stream rosters.

Folds are stratified by domain (patient counts per domain differ by at most one across folds) and balanced
on frame count: random restarts of a greedy largest-first assignment, each polished by same-domain patient
swaps; the split with the smallest max-min fold frame count wins. Deterministic given seeds.folds.
Writes derived/folds.csv (role cv: fold 0-4, external: -1, stream: -2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

RESTARTS = 300
TOLERANCE = 0.01  # fold frame spread below this share of the mean fold size counts as balanced
ROLE_FOLD = {"external": -1, "stream": -2}


def _score(totals):
    """totals: folds x domains frame counts. Total frame spread first (within a 1% tolerance), then the spread
    of each domain's frames across folds, then the exact total spread."""
    t = totals.sum(axis=1)
    spread = t.max() - t.min()
    return (max(spread, TOLERANCE * t.mean()), float((totals.max(axis=0) - totals.min(axis=0)).sum()), spread)


def _one_restart(frames, dom_idx, n_folds, rng):
    n, n_dom = len(frames), dom_idx.max() + 1
    cap = np.zeros((n_dom, n_folds), dtype=int)
    for d in range(n_dom):
        k = int((dom_idx == d).sum())
        cap[d] = k // n_folds
        cap[d, rng.permutation(n_folds)[: k % n_folds]] += 1
    order = np.argsort(-(frames * (1 + 0.25 * rng.random(n))))
    fold = np.full(n, -1)
    totals = np.zeros((n_folds, n_dom))
    for i in order:
        open_ = np.flatnonzero(cap[dom_idx[i]] > 0)
        f = open_[np.argmin(totals[open_].sum(axis=1))]
        fold[i] = f
        cap[dom_idx[i], f] -= 1
        totals[f, dom_idx[i]] += frames[i]
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = fold[i], fold[j]
                if a == b or dom_idx[i] != dom_idx[j]:
                    continue
                t = totals.copy()
                t[a, dom_idx[i]] += frames[j] - frames[i]
                t[b, dom_idx[i]] += frames[i] - frames[j]
                if _score(t) < _score(totals):
                    fold[i], fold[j], totals = b, a, t
                    improved = True
    return fold, totals


def assign_folds(patients, n_folds, seed, restarts=RESTARTS):
    """patients: DataFrame with patient_id, domain, n_frames. Returns a Series fold (0..n_folds-1) by patient_id."""
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    frames = patients.n_frames.to_numpy(dtype=float)
    dom_idx = patients.domain.astype("category").cat.codes.to_numpy()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        fold, totals = _one_restart(frames, dom_idx, n_folds, rng)
        s = _score(totals)
        if best is None or s < best[0]:
            best = (s, fold)
    return pd.Series(best[1], index=patients.patient_id, name="fold")


def patient_table(manifest):
    g = manifest.groupby("patient_id")
    t = pd.DataFrame(dict(center=g.center.first().astype(int), n_frames=g.size(), n_domains=g.domain.nunique(),
                          domain=g.domain.agg(lambda s: s.mode().iat[0]),
                          partition=g.partition.first(), n_partitions=g.partition.nunique())).reset_index()
    assert (t.n_domains == 1).all(), f"patients with mixed resolutions: {t[t.n_domains > 1].patient_id.tolist()}"
    assert (t.n_partitions == 1).all(), "a patient appears in more than one partition"
    return t.drop(columns=["n_domains", "n_partitions"])


def build_folds(manifest, n_folds, seed):
    t = patient_table(manifest)
    t["role"] = t.partition.map({"center1_labeled": "cv", "center2_labeled": "external", "stream": "stream"})
    cv = t[t.role == "cv"]
    f = assign_folds(cv[["patient_id", "domain", "n_frames"]], n_folds, seed)
    t["fold"] = t.role.map(ROLE_FOLD)
    t.loc[t.role == "cv", "fold"] = t.loc[t.role == "cv", "patient_id"].map(f)
    t["fold"] = t.fold.astype(int)
    t["_order"] = t.role.map({"cv": 0, "external": 1, "stream": 2})
    t["_num"] = t.patient_id.str.extract(r"(\d+)", expand=False).astype(int)
    t = t.sort_values(["_order", "_num"]).drop(columns=["_order", "_num", "partition"]).reset_index(drop=True)
    return t[["patient_id", "center", "domain", "n_frames", "role", "fold"]]


def check(folds, manifest, expected, n_folds):
    cv = folds[folds.role == "cv"]
    sets = [set(cv[cv.fold == k].patient_id) for k in range(n_folds)]
    assert all(sets), "an empty fold"
    assert sum(len(s) for s in sets) == len(set().union(*sets)), "patient appears in more than one fold"
    assert not set(cv.patient_id) & set(folds[folds.role == "external"].patient_id), "cv/external overlap"
    assert folds.patient_id.is_unique and set(folds.patient_id) == set(manifest.patient_id), "patients not all assigned once"
    assert set(cv.fold) == set(range(n_folds)) and (folds[folds.role == "external"].fold == -1).all() and (folds[folds.role == "stream"].fold == -2).all()
    lab = folds[folds.role != "stream"]
    got = dict(labeled_patients=len(lab), center1_patients=len(cv), center2_patients=int((folds.role == "external").sum()),
               labeled_frames=int(lab.n_frames.sum()), center1_frames=int(cv.n_frames.sum()),
               center2_frames=int(folds[folds.role == "external"].n_frames.sum()),
               stream_patients=int((folds.role == "stream").sum()), stream_frames=int(folds[folds.role == "stream"].n_frames.sum()))
    bad = {k: (expected[k], v) for k, v in got.items() if expected[k] != v}
    assert not bad, f"count mismatch (expected, got): {bad}"
    return got


def main():
    cfg = load_config()
    p, n_folds = cfg["paths"], cfg["segmenters"]["cv_folds"]
    manifest = pd.read_parquet(ROOT / p["derived"] / "manifest.parquet")
    folds = build_folds(manifest, n_folds, cfg["seeds"]["folds"])
    got = check(folds, manifest, cfg["dataset_counts_expected"], n_folds)
    print("reconciled:", got)
    cv = folds[folds.role == "cv"]
    summ = cv.groupby("fold").apply(lambda g: pd.Series({
        "patients": len(g), "frames": g.n_frames.sum(),
        **{f"{d}_patients": (g.domain == d).sum() for d in ("D1", "D2")},
        **{f"{d}_frames": g.n_frames[g.domain == d].sum() for d in ("D1", "D2")}}), include_groups=False)
    print(summ.to_string())
    print(f"fold frames: min {summ.frames.min()}, max {summ.frames.max()}, range {summ.frames.max() - summ.frames.min()}, "
          f"mean {summ.frames.mean():.1f}")
    out = ROOT / p["derived"] / "folds.csv"
    folds.to_csv(out, index=False)
    print(f"wrote {out} ({len(folds)} rows)")


if __name__ == "__main__":
    main()
