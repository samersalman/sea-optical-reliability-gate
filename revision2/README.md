# Revision 2 analyses (post hoc, exploratory)

Analyses added during the second round of peer review. They were chosen after the external results were known and are exploratory. None of them changes the locked gate, its cutoffs or the failure threshold (IoU below 0.90). Both R scripts verify the lock hashes in `v1/config/analysis_lock.yaml` before running.

## Files

```
r2_posthoc.R        analyses on the stored per-frame tables in v1/derived (outputs out/s1_* to out/s10_*)
r2_tierb.R          statistics on the pixel-level reruns (outputs out/t1_* to out/t4_*)
tierb/tierb.py      pixel-level reruns with the locked segmenters (outputs out/tierb_*.parquet, out/tierb_*.json)
tierb/run_*.log     logs of the pixel-level reruns
run_posthoc.log, run_tierb.log   logs of the R scripts, run in this repository layout
out/                every output table; out/tierb_r_sessioninfo.txt records the R session
```

## What each output holds

| Output | Content |
|---|---|
| s1_threshold_sensitivity | Binary metrics at IoU thresholds 0.75, 0.80, 0.85, 0.90 and 0.95 |
| s2_fixed_cutoffs, s2_cutoffs_used | Flags, sensitivity and missed failures at the development-derived cutoffs |
| s3_calibration_fifths | Calibration of the gate by fifths of predicted risk |
| s4_patient_macro, s4_design_effect | Patient-balanced estimates; intraclass correlation and effective sample size |
| s5_shift_extrapolation | External range of every metric, retained metrics flagged |
| s6_* | Domain-specific associations, metric-by-center interactions, score distributions, pairwise interactions |
| s7_matched_subgroups | Gates refitted on acquisition-matched training frames; scene strata |
| s8_ablation_variants | Leave-one-metric-out, single-metric, linear, clamped, robust, rank and domain-normalized gates |
| s9_perturbation_* | Failure rates and locked-gate behaviour by perturbation severity |
| s10_gate_* | Locked gate coefficients, spline knots and boundaries, settings and seeds |
| t1_boundary | Boundary IoU and HD95 as alternative error measures |
| t2_harmonization_* | Training-center frames downsampled to the external field-of-view size: gate, associations, domain separability |
| t3_perturbation_* | Localized and combined perturbations at severities 3 and 5 |
| t4_latency | Per-frame latency at batch size 1 (Apple M4 Pro laptop) |
| tierb_boundary.parquet | Per frame and segmenter: IoU, Dice, boundary IoU, HD95 (px), crop side, entropy, disagreement |
| tierb_harmonize.parquet | Per frame and segmenter, harmonized resolution: the 11 metrics, IoU, boundary IoU, HD95, entropy, disagreement, scale factor |
| tierb_perturb2.parquet | Per frame, segmenter and condition: IoU, Dice, entropy and the 11 metrics |

## Rerunning

The R scripts need only this repository (they read `v1/derived` and `v1/work/models`):

```
Rscript revision2/r2_posthoc.R
Rscript revision2/r2_tierb.R
```

`tierb/tierb.py` also needs the SEA images (see the main README) and the 12 checkpoints of release v.2.0.0 in `v1/work/models/`. It uses the code in `v1/python` read-only. The authors ran it with Python 3.12 and the package versions pinned in the main README. A rerun reproduced every stored IoU, entropy and disagreement value in `v1/derived/predictions.parquet` exactly.

```
PYTHONDONTWRITEBYTECODE=1 python revision2/tierb/tierb.py boundary
PYTHONDONTWRITEBYTECODE=1 python revision2/tierb/tierb.py perturb2
PYTHONDONTWRITEBYTECODE=1 python revision2/tierb/tierb.py harmonize
PYTHONDONTWRITEBYTECODE=1 python revision2/tierb/tierb.py latency
```

Set `OPTICS_V1` to the `v1` folder if it is not at `../v1` relative to this folder. Latency depends on the machine, and the gate-scoring time in `t4_latency.csv` varies slightly between runs.
