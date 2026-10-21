# SEA optical reliability gate

Analysis code, locked analysis settings, derived per-frame tables and aggregate results for a retrospective external-validation study. The question: can physically interpretable optical metrics of an endoscopic spine surgery frame (11 measures such as luminance, contrast, sharpness, saturation, glare, haze and red dominance, computed inside the detected circular field of view) flag the frames where an instrument-segmentation model fails? Two segmenters (U-Net with a ResNet34 encoder, and SegFormer-B0) are trained on the Center 1 patients of the public Spine Endoscopic Atlas (SEA) with patient-grouped five-fold cross-fitting. A penalized logistic "optical gate" is fitted on the out-of-fold results, locked, and then tested once on Center 2 (38 patients, 1,069 frames). A frame counts as a failure when its IoU is below 0.90. Secondary analyses perturb frames with seven physical degradations (defocus, haze, under- and over-exposure, blood absorption, glare, noise) and apply the locked gate to 20 unlabeled surgical videos.

Target paper: "Optical Quality Metrics for Detecting Instrument Segmentation Failure in Endoscopic Spine Surgery". Manuscript in preparation.

## Main result

The primary endpoint was not met. The endpoint compares the area under the risk-coverage curve (AURC; risk is 1 minus IoU, lower is better) of the locked gate with random ordering on Center 2. Intervals are 95% patient-clustered bootstrap intervals (2,000 resamples).

| Center 2, U-Net | AURC | AUROC for IoU below 0.90 |
|---|---|---|
| Optical gate | 0.048 (0.040 to 0.057) | 0.42 (0.37 to 0.48) |
| Random order | 0.046 (0.040 to 0.053) | not applicable |
| Pixel entropy | 0.022 (0.020 to 0.024) | 0.88 (0.84 to 0.91) |
| Ensemble disagreement | 0.022 (0.020 to 0.024) | 0.88 (0.83 to 0.92) |

Gate minus random AURC is 0.002 (-0.002 to 0.006), so the gate did no better than random order. Center 2 holds 109 failure frames (10.2%) from 31 patients. Scored out of fold on the training center, the gate did beat random order (AURC 0.042 against 0.054, difference -0.012, -0.015 to -0.007), but it did not transport. Post hoc diagnostics (labeled as such in the files) show a strong shift between centers: a classifier that separates Center 2 from Center 1 frames using only the 11 metrics reaches AUROC 0.99, and the locked gate keeps 4 of 33 candidate spline terms. No non-inferiority margin was set, so the comparisons with entropy and ensemble disagreement are descriptive.

Sources: `v1/results/parts/04b_external_eval.json` (keys `gate_ext_*` and `gate_int_*`), `v1/results/stats/gate_external.csv`, `v1/results/parts/34_diagnostics.json`.

## Layout

The repository mirrors the project folder, so all pipeline code sits under `v1/`. The R scripts find `v1/config/analysis_plan.yaml` from their own location or the working directory; set `OPTICS_V1` to override. The Python scripts take the root from the location of `v1/python/common.py` and have no override.

```
LICENSE  CITATION.cff  WEIGHTS_LICENSE.md  README.md  .gitignore
v1/
  python/    pixel work: 01_manifest, 02_optics (field of view and metrics), 03_folds, 04_train,
             05_predict, 06_perturb, 07_stream, shared modules, and tests/
  R/         all statistics, figures and tables (02 to 34), shared libraries, and tests/
  config/    analysis_plan.yaml, analysis_lock.yaml, G2_decision.md, CONTRACTS.md, history/
  scripts/   run_pilot.sh, run_t15.sh, run_t16.sh, g2_check.py, mps_bug_diagnostics/
  derived/   per-frame tables: manifest, fov, optics, predictions, perturb, stream (parquet), folds.csv
  results/   results.json, numbers-ledger.md, parts/*.json, stats/*.csv
  tables/    Table1 to Table3 and TableS1 to TableS3 as csv
  figures/   Figure1 to Figure3 as pdf
  work/      models/gate_*.rds (the four locked gate objects), qc/exhibits/fig1b_tiles.py
```

`config/history/` holds the original plan, a copy of the lock taken at lock time, `pip_freeze.txt`, `r_sessioninfo.txt` and `weights_sha256.txt`. `config/CONTRACTS.md` defines every derived table. `config/G2_decision.md` and `CONTRACTS.md` are historical records and name some files under `v1/work/` that are not shipped.

Not in this repository:
- SEA images and masks. The only SEA frames here are the six examples drawn in Figure 1B (`v1/figures/Figure1.pdf`).
- The 12 trained checkpoints (about 650 MB in total). They are assets of GitHub release v1.0.0.
- Manuscript drafts and the manuscript-assembly scripts `R/31_method_keys.R`, `R/32_text_keys.R`, `R/33_claim_check.R` and `R/40_render.R`. One exception: `R/30_bundle.R` is included because four tests source it. It merges `results/parts/*.json` into `results/results.json` and `results/numbers-ledger.md`. The parts `31_method_keys.json` and `32_text_keys.json` are shipped because `results.json` includes them, but the scripts that wrote them are not.
- Built exhibits: the Word, PNG and TIF versions of figures and tables. Only the PDF figures and CSV tables are kept. A rerun writes the other formats, and `.gitignore` excludes them.
- The Python environment (`v1/.venv`), logs, and the rest of `v1/work/`.

## Data

SEA is on Figshare, doi:10.6084/m9.figshare.27109312.v1, under CC0. The descriptor is Xu et al., Scientific Data 2025, doi:10.1038/s41597-025-05897-7. SEA holds 48,510 images: 10,662 annotated frames from 99 patients (Center 1: 61 patients, 9,593 frames; Center 2: 38 patients, 1,069 frames) and 37,848 unlabeled video frames from 20 further patients.

Unzip the download in the repository root so that a folder named `Spine endoscopic atlas` (holding `classified/` and `unclassified/`) sits next to `v1/`. The path comes from `paths.data` in `v1/config/analysis_plan.yaml`. Only the pixel stages and Figure 1B need it.

## Setup and run order

Commands run from the repository root. The authors used Python 3.12 in a virtual environment at `v1/.venv` and R 4.6.0. `R/10_fig1.R`, `scripts/run_pilot.sh` and `scripts/run_t15.sh` call `v1/.venv/bin/python` by that path.

```
python3.12 -m venv v1/.venv
v1/.venv/bin/pip install torch==2.10.0 torchvision==0.25.0 transformers==5.2.0 segmentation_models_pytorch==0.5.0 \
  numpy==2.3.5 pandas==2.3.3 pyarrow==23.0.0 opencv-python-headless==5.0.0.93 pillow==12.1.1 \
  pynrrd==1.1.3 PyYAML==6.0.3 pytest==9.0.2
```

`v1/config/history/pip_freeze.txt` freezes the authors' whole environment, so use it as a version reference only. R packages used: arrow, car, dplyr, flextable, ggplot2, glmnet, jsonlite, mgcv, officer, patchwork, pROC, png, ragg, sandwich, scales, systemfonts, tibble, tidyr, xgboost, yaml.

`v1/python/models.py` sets `HF_HUB_OFFLINE=1` unless you set it yourself, and it reads `nvidia/mit-b0` and the ImageNet ResNet34 encoder from the local Hugging Face cache. On a fresh machine, prefix the first training or inference run with `HF_HUB_OFFLINE=0` to fill the cache. Devices are chosen automatically: Apple MPS if available, otherwise CPU.

Every stage writes in place. Python stages overwrite `v1/derived/`, R stages overwrite `v1/results/`, `v1/tables/` and `v1/figures/`. Rerun in a copy of the repository, or review `git diff` afterwards.

A. Pixel stages (01, 02 and 04 to 07 read the SEA images; `run_t15.sh` also needs the 12 checkpoints in `v1/work/models/`, from `04_train.py` or from the release)
```
v1/.venv/bin/python v1/python/01_manifest.py       # needs SEA
v1/.venv/bin/python v1/python/03_folds.py          # reads derived/manifest.parquet only
v1/.venv/bin/python v1/python/02_optics.py all     # needs SEA: field of view, metrics, Gate G1 report
v1/.venv/bin/python v1/python/04_train.py --all    # 10 fold models and 2 final models, resumable
bash v1/scripts/run_t15.sh                         # 05_predict, 07_stream, 06_perturb
```
`scripts/run_pilot.sh` and `scripts/g2_check.py` are the Gate G2 pilot (fold 0, Center 1 only). The lock already holds its decisions, so reproduction does not need them.

B. Statistics (R; they read `v1/derived/` and `v1/results/`, not SEA pixels)
```
Rscript v1/R/02_cohort.R
bash v1/scripts/run_t16.sh                         # 03, 04b, 05, 06, 07, 08
Rscript v1/R/34_diagnostics.R                      # post hoc
Rscript v1/R/21_supplement.R                       # post hoc
for s in 11_fig2 12_fig3 20_tables 30_bundle; do Rscript v1/R/$s.R; done
Rscript v1/R/10_fig1.R                             # needs SEA and v1/.venv, see below
```
`R/04a_gate_fit_lock.R` fits and seals the gate. The shipped lock is already sealed, so the script stops with a message; `OPTICS_RELOCK=1` would refit the gate and replace the committed hashes. Do not use it when reproducing the paper.

Reproducing from the committed tables: with no SEA pixels and no weights, the R scripts 02, 03, 04b, 05, 06, 07, 08, 11, 12, 20, 21, 30 and 34 run in about 3 minutes in total. In a scratch copy, every file under `v1/results/` and every CSV under `v1/tables/` came out byte-identical to the committed one, and `03_folds.py` rewrote `folds.csv` identically. The PDF figures differ only in the embedded creation date and ID, so `git diff` flags them. Unless both the SEA folder and `v1/.venv` exist, `R/10_fig1.R` stops with an error and writes no figure, because panel B reads the images through `work/qc/exhibits/fig1b_tiles.py`. Before it calls that helper it writes `v1/work/qc/exhibits/fig1b_candidates_in.csv` (git-ignored). With both present it rebuilt Figure 1 with six tiles. The helper runs under `python -I`, so it writes `v1/python/__pycache__` (git-ignored).

Synthetic mode: `Rscript v1/R/tests/make_synthetic.R` writes deterministic stand-ins for the derived tables to `v1/work/synthetic/`. With `OPTICS_DATA=synthetic`, scripts read them and write only under `v1/work/synthetic/out/`; they refuse to touch the real folders. Figure 1B is skipped in this mode.

Not runnable as staged: the seven scripts in `scripts/mps_bug_diagnostics/` (historical; they expect a pilot checkpoint and a different folder depth) and `scripts/g2_check.py` (needs the fold-0 pilot checkpoints from `run_pilot.sh`, and SEA).

### Tests

Python tests, run from `v1/python` (it holds `pytest.ini`):

```
(cd v1/python && ../.venv/bin/python -m pytest -p no:cacheprovider -q -rs)
```

Without the SEA folder, 20 tests skip and say why ("dataset not present" or "image data not available"); the result was 218 passed, 20 skipped in about 3 minutes. With SEA unzipped at the repository root, `test_predict.py`, `test_train.py`, `test_fov.py`, `test_optics_lib.py` and `test_optics_stage.py` gave 136 passed and 0 skipped in 8 to 11 minutes (the model tests use Apple MPS when present).

R tests use synthetic data and never write real outputs. Run them one at a time, in this order (the whole sequence took about 3 minutes):

```
Rscript v1/R/tests/make_synthetic.R
for t in test_setup test_stats_lib test_gate_pipeline test_stream_sens test_assoc test_diag; do Rscript v1/R/tests/$t.R; done
for s in 04a_gate_fit_lock 03_association 04b_external_eval 05_transfer 06_perturb 07_stream 08_sensitivity 34_diagnostics 02_cohort 10_fig1 11_fig2 12_fig3 20_tables 21_supplement; do OPTICS_DATA=synthetic Rscript v1/R/$s.R; done
for t in test_tables test_fig23; do Rscript v1/R/tests/$t.R; done
```

`test_diag.R` needs `test_assoc.R` (or `03_association.R`) to have run first. `test_tables.R` and `test_fig23.R` need the pipeline loop. In the verification run every R test passed, with these check counts: test_setup 65, test_stats_lib 92, test_gate_pipeline 66, test_stream_sens 62, test_assoc 99, test_diag 53, test_tables 74, test_fig23 55.

## Weights

The 12 checkpoints are assets of release v1.0.0: `unet_fold0.pt` to `unet_fold4.pt`, `unet_final.pt`, `segformer_fold0.pt` to `segformer_fold4.pt` and `segformer_final.pt`. The release also carries `SHA256SUMS` and `LICENSE-NVIDIA-SegFormer.txt`. Fold k models score Center 1 patients in fold k; the final models trained on all Center 1 frames score Center 2 and the stream.

```
gh release download v1.0.0 -R samersalman/sea-optical-reliability-gate --dir v1/work/models
cd v1/work/models && shasum -a 256 -c SHA256SUMS
```

Without `gh`, fetch each file with `curl -L -O`, for example from `https://github.com/samersalman/sea-optical-reliability-gate/releases/download/v1.0.0/unet_fold0.pt`. The same hashes are in `v1/config/history/weights_sha256.txt`; from `v1/work/models` run `shasum -a 256 -c ../../config/history/weights_sha256.txt`. The `.pt` files are release assets and are not part of the repository source archive.

Licenses, as set out in `WEIGHTS_LICENSE.md`: the U-Net checkpoints are MIT for the authors' part, and the ImageNet-pretrained ResNet34 initialization carries its own terms. The SegFormer checkpoints are distributed under the NVIDIA Source Code License for SegFormer, which limits use to research or evaluation, with a copy of that license attached to the release. Read `WEIGHTS_LICENSE.md` before reuse.

## Lock and integrity

`v1/config/analysis_lock.yaml` freezes the failure threshold (0.9), the epochs (20 for each segmenter), the absence of a non-inferiority margin (`null`), the entropy baseline choice, and the path and SHA-256 of the four gate objects. `lock_fields_sha256` hashes every other field except `locked_at`. The lock was written on 2026-10-20 at 13:49:35 (-0500), and `v1/config/history/analysis_lock_at_lock.yaml` is the copy taken then. The gate was fitted on Center 1 only and sealed before any Center 2 outcome was analyzed. The segmenter inference that produced the Center 2 predictions had already run, but no script read those outcomes until the lock existed.

```
e442ba478ebfb5e938ab863d4984192926e6520ecd07776ce5a86d53fcfef249  v1/work/models/gate_optical.rds
89f909951a680cef94d397ec6ca0bda353df1862c1e251ccaab9f065db2644a7  v1/work/models/gate_area.rds
f56a2d503e5c072c5150d21a63e6f923afebe7930d1b7d425b1ed6c7df13da4f  v1/work/models/gate_stack.rds
a0920b07707556377fc5f7cee7423b94d40f8eb08537b7520c2eee40a1539994  v1/work/models/gate_folds.rds
```

Check the lock and the four files (prints `[1] TRUE`; it stops with a message on any mismatch):

```
Rscript -e 'source("v1/R/00_setup.R"); source("v1/R/stats_lib.R"); print(verify_lock_hashes(read_lock(file.path(V1, "config", "analysis_lock.yaml"))))'
```

The scripts that use the lock or open Center 2 outcomes (03, 04b, 05, 07, 08, 20, 21 and 34) call `verify_lock_hashes()`, directly or through `run_locked_evaluation()`, before they read those outcomes. The lock does not hash the segmenter weights; their hashes are in `v1/config/history/weights_sha256.txt`.

Gate G2 decisions, recorded in `v1/config/G2_decision.md` from a Center 1 pilot before any Center 2 data were opened: 20 epochs for both segmenters; failure threshold revised from 0.75 to 0.90 (at 0.75 only about 1% of frames fail, too few events for the gate and for external binary metrics); no non-inferiority margin.

An MPS finding shaped the code: on the authors' Apple GPU, PyTorch `grid_sample` returned dimmed images on real crops. All resampling and augmentation therefore run on the CPU, and only model forward passes use MPS. The diagnostic scripts are in `v1/scripts/mps_bug_diagnostics/`.

## License

- Code: MIT (`LICENSE`).
- U-Net checkpoints: MIT for the authors' part; SegFormer checkpoints: NVIDIA Source Code License for SegFormer (research or evaluation use). See `WEIGHTS_LICENSE.md`.
- SEA data: CC0, not redistributed here.
- The SEA descriptor article is CC BY-NC-ND 4.0; that license applies to the article text only, not to the data.

## Citation

See `CITATION.cff` (GitHub shows a "Cite this repository" button). Please also cite the SEA descriptor (Xu et al., doi:10.1038/s41597-025-05897-7) and, if you use the SegFormer checkpoints, Xie et al. 2021, arXiv:2105.15203.

Zenodo DOI: to be added at release
