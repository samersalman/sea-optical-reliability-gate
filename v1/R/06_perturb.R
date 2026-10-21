# Perturbation experiment statistics, from derived/perturb.parquet (python/06_perturb.py).
#   (a) IoU dose-response per model, family and severity with patient-clustered bootstrap bands
#   (b) severity at which the median IoU first falls below the rule in analysis_plan.yaml (perturbation.rule_first_below_iou)
#   (c) metric sensitivity: Spearman rho between each optical metric and ordinal severity, per family
#   (d) the same rho as a long table for the heat-map (Figure 2C)
# Severity 0 is the unperturbed baseline, repeated once per family in the file, so every family has a full
# 0-5 curve; the checks below confirm that convention and print any deviation.
#
# Usage: Rscript R/06_perturb.R        (OPTICS_DATA=synthetic for development)
# Writes results/stats/perturb_{doseresponse,threshold,rho}.csv and results/parts/06_perturb.json.

.here <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else file.path(getwd(), "v1", "R")
})
source(file.path(.here, "00_setup.R"))
source(file.path(.here, "01_load.R"))
source(file.path(.here, "results_lib.R"))
source(file.path(.here, "assoc_lib.R"))

assoc_guard_results_root()
thr <- CFG$perturbation$rule_first_below_iou
MET <- OPTICAL_METRICS
REF_MODEL <- "unet"   # metrics come from the perturbed image, so they are the same on every model row
EXPECTED_FAMILIES <- c("defocus", "haze", "exposure_under", "exposure_over", "blood_absorption", "glare", "noise")

# Expected metric movements with severity (T3: python/perturb_ops.py). Noise raising Tenengrad is a known
# limitation of gradient energy as a sharpness measure, reported, not a bug.
EXPECTED <- data.frame(
  family = c("defocus", "defocus", "haze", "haze", "exposure_under", "exposure_under", "exposure_over", "exposure_over",
             "blood_absorption", "glare", "glare", "noise"),
  metric = c("sharp_tenengrad", "contrast_local", "haze_dcp", "contrast_rms", "lum_mean", "under_frac", "lum_mean",
             "sat_frac", "red_dom", "sat_frac", "spec_frac", "sharp_tenengrad"),
  expected_sign = c(-1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1),
  known_limitation = c(rep(FALSE, 11), TRUE),
  stringsAsFactors = FALSE
)

pt <- load_perturb()
if (!REF_MODEL %in% pt$model) stop("perturb.parquet has no '", REF_MODEL, "' rows")
if (anyNA(pt$iou)) stop("perturb.parquet has missing IoU values")

# ---- checks on the file (contract conventions) -------------------------------------------------

issues <- character()
note_issue <- function(txt) { issues <<- c(issues, txt); message("CONTRACT NOTE: ", txt) }

exp_cells <- n_distinct(pt$frame_id) * n_distinct(pt$model) * n_distinct(pt$family) * n_distinct(pt$severity)
if (nrow(pt) != exp_cells) note_issue(sprintf("grid is incomplete: %d rows, %d expected from frames x models x families x severities", nrow(pt), exp_cells))
if (!0L %in% pt$severity) stop("perturb.parquet has no severity 0 baseline rows")
if (!setequal(pt$family, EXPECTED_FAMILIES)) {
  note_issue(paste("families differ from the contract:", paste(sort(unique(pt$family)), collapse = ", ")))
}
base <- pt[pt$severity == 0L, ]
spread <- base |> group_by(frame_id, model) |> summarise(r = max(iou) - min(iou), .groups = "drop")
if (max(spread$r) > 1e-9) note_issue(sprintf("severity 0 IoU is not identical across families (max spread %.3g): baseline not repeated per family", max(spread$r)))
mdiff <- pt |> group_by(frame_id, family, severity) |>
  summarise(across(all_of(MET), ~ max(.x) - min(.x)), .groups = "drop")
mdiff_max <- max(unlist(mdiff[MET]))
if (mdiff_max > 1e-6) note_issue(sprintf("optical metrics differ between model rows of the same image (max %.3g); using the %s rows", mdiff_max, REF_MODEL))
pn <- pt |> group_by(family, severity) |> summarise(k = n_distinct(round(param, 10)), .groups = "drop")
if (any(pn$k > 1L)) note_issue("param is not constant within family x severity; param columns are NA where it varies")

# ---- (a), (b) dose-response and threshold ------------------------------------------------------

res <- assoc_perturb_curves(pt, thr = thr)
curves <- res$curves
curves <- curves |>
  group_by(model, family) |>
  mutate(median_rel_baseline = median_iou / median_iou[severity == 0L]) |>
  ungroup() |>
  as.data.frame()
curves <- curves[c("model", "family", "severity", "param", "n_frames", "n_patients", "median_iou", "median_lo", "median_hi",
                   "mean_iou", "mean_lo", "mean_hi", "median_rel_baseline")]
thresholds <- res$thresholds
thresholds$baseline_below_threshold <- thresholds$baseline_median_iou < thr
thresholds$never_reached <- is.na(thresholds$first_severity_below)
thresholds$median_iou_max_severity <- vapply(seq_len(nrow(thresholds)), function(i) {
  cv <- curves[curves$model == thresholds$model[i] & curves$family == thresholds$family[i], ]
  cv$median_iou[which.max(cv$severity)]
}, numeric(1))
thresholds <- thresholds[c("model", "family", "threshold_iou", "baseline_median_iou", "baseline_below_threshold",
                           "first_severity_below", "severity_interp", "param_at_first", "first_severity_lo",
                           "first_severity_hi", "prob_never", "never_reached", "median_iou_max_severity")]

# ---- (c), (d) metric sensitivity ---------------------------------------------------------------

pm <- pt[pt$model == REF_MODEL, c("frame_id", "patient_id", "domain", "family", "severity", MET)]
rho_raw <- assoc_perturb_rho(pm, MET)
rho <- data.frame(family = rho_raw$family, metric = rho_raw$variable, label = unname(METRIC_LABELS[rho_raw$variable]),
                  rho = rho_raw$est, rho_lo = rho_raw$lo, rho_hi = rho_raw$hi, n_obs = rho_raw$n,
                  n_frames = vapply(rho_raw$family, function(f) n_distinct(pm$frame_id[pm$family == f]), integer(1)),
                  n_patients = rho_raw$n_patients, stringsAsFactors = FALSE)
rho <- merge(rho, EXPECTED, by = c("family", "metric"), all.x = TRUE, sort = FALSE)
rho$expected_pair <- !is.na(rho$expected_sign)
rho$known_limitation[is.na(rho$known_limitation)] <- FALSE
rho$as_expected <- ifelse(rho$expected_pair, sign(rho$rho) == rho$expected_sign & abs(rho$rho) >= 0.5, NA)
rho$expected_sign[is.na(rho$expected_sign)] <- 0
rho$family <- factor(rho$family, levels = union(EXPECTED_FAMILIES, unique(rho$family)))
rho$metric <- factor(rho$metric, levels = MET)
rho <- rho[order(rho$family, rho$metric), ]
rho$family <- as.character(rho$family)
rho$metric <- as.character(rho$metric)
rho <- rho[c("family", "metric", "label", "rho", "rho_lo", "rho_hi", "n_obs", "n_frames", "n_patients",
             "expected_pair", "expected_sign", "known_limitation", "as_expected")]
rownames(rho) <- NULL

assoc_write_csv(curves, assoc_out("results", "stats", "perturb_doseresponse.csv"))
assoc_write_csv(thresholds, assoc_out("results", "stats", "perturb_threshold.csv"))
assoc_write_csv(rho, assoc_out("results", "stats", "perturb_rho.csv"))

# ---- registry ----------------------------------------------------------------------------------

frames <- distinct(pt, frame_id, patient_id)
reg("perturb_n_frames", nrow(frames))
reg("perturb_n_patients", n_distinct(frames$patient_id))
reg("perturb_n_families", n_distinct(pt$family))
reg("perturb_n_severities", n_distinct(pt$severity[pt$severity > 0L]))
reg("perturb_threshold_iou", thr, digits = 2)
for (m in sort(unique(pt$model))) {
  cm <- curves[curves$model == m, ]
  reg(paste0("perturb_baseline_iou_", m), cm$median_iou[cm$severity == 0L][1], digits = 3, note = "median IoU, severity 0, recomputed on the perturbation path")
  tm <- thresholds[thresholds$model == m, ]
  reg(paste0("perturb_n_reached_", m), sum(!tm$never_reached), note = "families whose median IoU falls below the rule within the severity grid")
  for (f in tm$family) {
    r <- tm[tm$family == f, ]
    key <- paste0("perturb_thr_", m, "_", f)
    if (r$never_reached) reg(key, "not reached", note = "median IoU never fell below the rule within the grid (string, not numeric)")
    else reg(key, r$first_severity_below, note = "first severity (ordinal) with median IoU below the rule")
    assoc_reg_num(paste0("perturb_thr_interp_", m, "_", f), r$severity_interp, digits = 1, note = "linearly interpolated crossing, severity units")
    reg(paste0("perturb_iou_max_sev_", m, "_", f), r$median_iou_max_severity, digits = 2, note = "median IoU at the highest severity")
  }
}
for (i in seq_len(nrow(rho))) {
  r <- rho[i, ]
  assoc_reg_ci(paste0("perturb_rho_", r$family, "_", r$metric), r$rho, r$rho_lo, r$rho_hi, 2,
               note = if (r$known_limitation) "known limitation: gradient energy counts noise as sharpness" else "Spearman rho vs ordinal severity")
}
ex <- rho[rho$expected_pair & !rho$known_limitation, ]
reg("perturb_n_expected_pairs", nrow(ex))
reg("perturb_n_expected_ok", sum(ex$as_expected, na.rm = TRUE), note = "expected pairs with the expected sign and |rho| >= 0.5")
assoc_reg_num("perturb_rho_abs_min_expected", min(abs(ex$rho)), digits = 2)

write_results_part("06_perturb")
message("06_perturb done: ", nrow(curves), " curve points, ", nrow(thresholds), " thresholds, ", nrow(rho), " rho rows",
        if (length(issues)) paste0("; ", length(issues), " contract note(s)") else "")
