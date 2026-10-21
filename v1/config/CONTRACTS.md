# Data contracts (v1)

Single source of truth for file schemas, so parallel tasks can build against each other before the upstream data exist.
If you must change a contract, edit this file AND append a note to every affected downstream task file in `v1/work/tasks/`.

Conventions
- Project root `ROOT` is the folder that contains `v1/` (the repository root). Paths stored in tables are RELATIVE to ROOT.
- Python: `v1/.venv/bin/python` (system-site-packages for torch). Python does pixel work only. R does every statistic, table, figure.
- Parquet via pyarrow (Python) and `arrow::read_parquet` (R). Column names are snake_case exactly as below.
- All scripts take paths from `v1/config/analysis_plan.yaml`; no hard-coded magic numbers. Seeds from the same file.
- `frame_id` is the join key everywhere. `patient_id` is the cluster for every resampling/CI.
- Disk: ~13 GB free. No large caches. Do not delete anything under `Spine endoscopic atlas/` or the `.zip`.

## derived/manifest.parquet  (python/01_manifest.py) - 48,510 rows
| column | type | notes |
|---|---|---|
| frame_id | str | unique key; the upper-cased file stem (34 stems are lower-case on disk, e.g. `p23-1-0140` -> `P23-1-0140`); if a stem repeats, suffix is `__2` etc. and documented |
| partition | str | `center1_labeled` \| `center2_labeled` \| `stream` |
| patient_id | str | upper-cased (`p46` -> `P46`); Center 2 = `T...` |
| center | int | 1 (P) or 2 (T) |
| domain | str | D1 (C1 1920x1080), D2 (C1 1280x720), D3 (C2 720x576); stream rows assigned by resolution |
| width, height | int | from image header |
| channel | str | `big` \| `small` \| NA (stream) |
| region | str | `cervical` \| `lumbar` \| NA |
| scenario | str | `Normal` \| `Difficult` \| NA |
| instrument | str | `forceps` `bipolar` `drill` `punch` `scissor` `dissector` \| NA (from FOLDER name, not segment name) |
| file_format | str | `jpg` \| `png`; NOT tied to the centre (Center 1 has both: 5,442 png frames; D2 is 100% png, D1 mixed, D3 100% png) |
| video_idx, frame_idx | int (nullable) | parsed robustly; 261 Center 2 names lack a video index |
| image_path, mask_path | str | relative to ROOT; mask_path NA for stream |
| labeled | bool | |
Binary mask = `mask > 0` (values {0,1} or {0,255}; 56 frames carry 2 segments - union them).

## derived/fov.parquet  (python/02_optics.py fov) - 48,510 rows
`frame_id, fov_found (bool), fov_cx, fov_cy, fov_r (float, native px), fov_method (str), fov_resid (float), crop_x0, crop_y0, crop_side (int, = round(2*fov_r))`
`fov_method` is `ray` (sub-pixel refined from boundary rays), `mask` (coarse mask-based fit only) or `fallback` (no field found: `fov_found` is false and the geometry is the image centre with r = half the shorter side).
For D3 the detector fits an axis-aligned ellipse and reports its inscribed circle (centre = ellipse centre, r = shorter semi-axis), so `fov_r` is the inscribed radius there.
Frames with `fov_found` false (28: 3 labeled Center 1, 25 stream) have NaN optics and no prediction rows. With `fov.exclude_not_found: true` (plan) they are excluded everywhere: `python/dataset.py::frame_table` drops and counts them (`attrs["dropped_no_fov"]`; false is refused), `R/01_load.R::load_frame_table(exclude_no_fov = )` drops them and returns the count as `attr(, "n_excluded_fov")`, and `stats_lib.R::read_eval_frames` drops them before its missing-value check.
Canonical FOV crop = square of side `crop_side` centred on (fov_cx, fov_cy), zero-padded where it leaves the image.
Library: `python/fov.py` with `detect_fov(img_rgb)` and `crop_square(arr, cx, cy, r, out_size, interp)` and `fov_disk_mask(size, erode_frac)`.

## derived/optics.parquet  (python/02_optics.py metrics) - 48,510 rows
`frame_id` + the 11 metrics in `analysis_plan.yaml:optical_metrics`, exactly those column names, computed inside the eroded FOV after resampling the FOV crop to 512 x 512. Library: `python/optics_lib.py::compute_metrics(rgb_512, disk_mask, bg_mask=None) -> dict` (`bg_mask` = boolean pixels to keep; used for the `_bg` sensitivity set). Ground-truth masks are NEVER used.
`sharp_tenengrad` is the mean squared Sobel gradient of luma scaled to luma units per pixel, so values are small (about 3e-4 on real frames); print it in scientific notation.

## derived/folds.csv  (python/03_folds.py) - 119 rows
`patient_id, center, domain, n_frames, role, fold` where role = `cv` (61 Center 1 labeled patients, fold 0-4), `external` (38 Center 2, fold = -1), `stream` (20, fold = -2). Domain for a patient with mixed resolutions: majority; assert none mixed.

## derived/predictions.parquet  (python/05_predict.py) - 10,659 analysed labeled frames x 2 models = 21,318 rows
`frame_id, patient_id, model (unet|segformer), source (oof|final), partition, fold (int; -1 external), iou, dice, pred_area_frac, gt_area_frac, entropy_fov, entropy_roi, ens_dis (1 - Dice(unet, segformer); same value on both model rows), unet_bg_<11 metrics> (optics on complement of the U-Net predicted mask; repeated on both rows)`
- Rows: 9,590 Center 1 + 1,069 Center 2 frames. The 3 Center 1 frames without a detected FOV (P5-9-1066, P49-1-0019, P57-1-0016) are excluded and counted, not substituted.
- `fold`: Center 1 = the patient's fold 0-4, Center 2 = -1 (stream = -2 in stream.parquet), as in folds.csv.
- IoU/Dice at native resolution (GT = FOV-square crop at crop_side px, probability map upsampled bilinearly, cut at 0.5), inside the eroded-FOV disk. Both masks empty gives IoU 1 and Dice 1 (not expected on labeled frames; counted as gt_empty_in_disk in the run summary). An empty prediction against a non-empty GT gives IoU 0 and Dice 0.
- `entropy_fov` = mean binary entropy (bits) of the sigmoid output over the eroded FOV disk, computed on the 384 x 384 grid.
- `entropy_roi` = same entropy over the predicted mask (0.5 cut on the 384 grid) dilated by an elliptical 15 px kernel (entropy.roi_dilate_px) and clipped to the disk. It falls back to the `entropy_fov` value when the predicted mask is empty (counted per model in the run summary).
- `ens_dis` = 1 - Dice of the two models' masks on the 384 grid inside the disk; both masks empty gives 0.
- `unet_bg_*`: the 11 optical metrics on the 512 canvas over the disk minus the U-Net predicted mask (never the ground truth). All 11 are NaN when fewer than 256 background pixels remain (counted in the run summary).
- `pred_area_frac` = predicted foreground / FOV area.
- `oof`: out-of-fold model of the patient's fold (Center 1). `final`: all-Center-1 model (Center 2).

## derived/stream.parquet  (python/07_stream.py) - 37,823 analysed frames x 2 models = 75,646 rows
Same as predictions.parquet minus `iou, dice, gt_area_frac`; plus `video_patient, t_index (0-based consecutive order within video)`.
- Rows: the 37,848 stream frames minus the 25 without a detected FOV (10 of them from P82). `fold` = -2, `source` = final.
- `t_index` = 0-based rank among the ANALYSED frames of the video, ordered by manifest frame_idx; gapless. The time axis (1 frame/s) is the manifest frame_idx, not t_index, because excluded or missing frames leave holes in frame_idx. `video_patient` equals `patient_id`.

## derived/perturb.parquet  (python/06_perturb.py) - 400 frames x 2 models x 7 families x severities 0-5 = 33,600 rows
`frame_id, patient_id, domain, scenario, instrument, model, family, severity (0-5), param (physical parameter value), iou, dice, pred_area_frac, entropy_fov, <11 optics metrics after perturbation>`
`family` in {defocus, haze, exposure_under, exposure_over, blood_absorption, glare, noise}; severity 0 = unperturbed baseline. Operators live in `python/perturb_ops.py`, signature `apply(rgb_float01, disk_mask, family, severity, rng) -> rgb_float01` (changes pixels only inside `disk_mask`).
- Frames: OOF Center 1 frames with U-Net IoU >= perturbation.min_oof_iou (0.90) in predictions.parquet, equal allocation over domain x scenario x instrument cells (realised table: work/logs/perturb_allocation.csv).
- `iou`, `dice`, `pred_area_frac` and `entropy_fov` are on the 384 grid inside the eroded 384 disk, against the nearest-neighbour 384 GT crop. They are NOT the native-resolution values of predictions.parquet: do not merge the two files.
- Severity 0 is recomputed on the same canvas path (native -> 512 canvas -> 384), not copied from predictions.parquet. It is one image per frame and model, repeated under every family label so each family has a full 0-5 curve; `param` at severity 0 is the identity value of the family (0, or gain 1.0 for exposure_under and exposure_over).
- The 11 optics columns describe the perturbed image, so they are identical on the unet and segformer rows of a (frame, family, severity).

## work/models/
`{unet,segformer}_fold{0..4}.pt`, `{unet,segformer}_final.pt`, `train_log_*.csv`; per-epoch checkpoints `*_ckpt.pt` (resumable). Gitignored, ~1 GB.

## results/results.json  (R/30_bundle.R)
Flat dictionary `key -> value` (string-formatted at publication precision plus raw numeric under `raw.<key>`). `numbers-ledger.md` lists key, value, source script. Manuscript templates use `{{key}}`.

## config/analysis_lock.yaml  (written at Gate G2, completed before external evaluation)
`failure_iou_threshold, epochs (per model), noninferiority_margin_aurc (or null), entropy_baseline_choice, gate_rds_path, gate_rds_sha256, gate_area_rds_*, gate_stack_rds_*, gate_folds_rds_* (path, sha256 each), locked_at (timestamp), lock_fields_sha256`.
`lock_fields_sha256` seals every frozen choice, not only the RDS files. Definition: SHA-256 (lower-case hex) of the UTF-8 bytes of the canonical JSON of all top-level lock keys EXCEPT `lock_fields_sha256` itself and `locked_at`. Canonical JSON: one object, keys sorted bytewise (`order(method = "radix")`, nested maps likewise), no whitespace, strings JSON-escaped, numbers printed with `%.15g` (so `30` and `30L` are both `30`, `0.75` is `0.75`), booleans `true`/`false`, null or NA as `null`, sequences as arrays. Reference implementation: `R/stats_lib.R::lock_fields_hash()`.
- `R/04a_gate_fit_lock.R` writes `lock_fields_sha256` as its LAST lock step, after the entropy choice and the RDS hashes are in.
- `verify_lock_hashes(lock)` (stats_lib.R) recomputes it and compares, then checks every `*_rds_sha256` against its file. `04b_external_eval.R` and `05_transfer.R` call it (through `run_locked_evaluation`) before any Center 2 row is read; `03_association.R` calls it whenever the lock carries a hash, before it uses `failure_iou_threshold`. Any future script that reads the threshold, margin, epochs or entropy choice from the lock must call it first. Changing the threshold, margin, epochs or entropy choice after locking therefore stops the run.
- `python/04_train.py` refuses a non-pilot `--epochs N` that differs from the frozen `epochs` of that model.
`R/04b_external_eval.R` must refuse to run unless both the field hash and the RDS hashes equal the recorded values.

## Task handoffs
Each task has a file `v1/work/tasks/Txx.md`. When your work creates a fact a downstream task must know (changed schema, path quirk, caveat, measured number), APPEND it with `cat >> v1/work/tasks/<downstream>.md <<'EOF'` under the heading `## Handoff notes` (append only, never rewrite another task's file).
