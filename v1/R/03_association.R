# Table 2 statistics: how each of the 11 optical metrics relates to the expert Normal/Difficult label and
# to U-Net failure (IoU < threshold). Inference uses Center 1 out-of-fold rows only.
#
# Center 2 is descriptive. Its scenario labels and metrics are always summarised; its segmentation OUTCOMES
# (failure ORs, pooled sensitivity) are opened only after the gate lock is complete
# (analysis_lock.yaml: gate_rds_sha256), so run this script after 04a_gate_fit_lock.R to get that block.
# Synthetic mode: OPTICS_ASSOC_EXTERNAL=1 or 0 overrides the lock check (tests only; ignored on real data).
# A lock that carries any hash must pass verify_lock_hashes() (stats_lib.R) before its failure threshold is used, so a
# threshold edited after locking stops this script.
#
# Frames whose field of view was not found (fov_found == FALSE, fov.exclude_not_found) are dropped by load_frame_table()
# and counted (assoc_n_fov_excluded); they have NaN optics and no U-Net rows.
#
# Confounders: domain, channel, region, image format (file_format), instrument and the U-Net predicted area. Image format
# is not tied to the centre (both jpg and png occur within Center 1), so it is estimable in the Center 1 models. The
# aliasing report is kept as a defence: format is aliased with domain only in a model fitted on a subset where the two
# coincide (the synthetic test data, where png occurs only in D3, is such a subset), and the output then names the
# dropped term. Check the VIFs and that file_format is estimable within Center 1 on the real run.
#
# Usage: Rscript R/03_association.R        (OPTICS_DATA=synthetic for development)
# Writes results/stats/assoc_*.csv and results/parts/03_association.json.

.here <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else file.path(getwd(), "v1", "R")
})
source(file.path(.here, "00_setup.R"))
source(file.path(.here, "01_load.R"))
source(file.path(.here, "results_lib.R"))
source(file.path(.here, "assoc_lib.R"))
source(file.path(.here, "stats_lib.R"))

assoc_guard_results_root()
ext_forced <- SYNTH && nzchar(Sys.getenv("OPTICS_ASSOC_EXTERNAL", ""))
lock_file <- p("config", "analysis_lock.yaml")
lock_sealed <- file.exists(lock_file) && any(vapply(c("gate_rds_sha256", "lock_fields_sha256"),
                                                   function(k) .gl_nonempty(read_lock(lock_file)[[k]]), NA))
if (lock_sealed && !ext_forced) verify_lock_hashes(read_lock(lock_file))
thr <- assoc_failure_threshold()
ext_open <- if (ext_forced) identical(Sys.getenv("OPTICS_ASSOC_EXTERNAL"), "1") else assoc_lock_complete()
message("failure threshold IoU < ", thr, "; Center 2 outcomes ", if (ext_open) "OPEN (lock complete)" else "not opened (lock incomplete)")

MET <- OPTICAL_METRICS
QUANTITY <- c(
  lum_mean = "Mean luminance (exposure level; Rec. 709 Y, 0 to 1)",
  contrast_rms = "Global contrast (SD of luminance)",
  contrast_local = "Local contrast (mean 16 px block contrast)",
  sharp_tenengrad = "Sharpness (mean squared Sobel gradient of luminance)",
  under_frac = "Under-exposed area (fraction of pixels with Y below 0.10)",
  sat_frac = "Saturated area (fraction of pixels with any channel at or above 0.98)",
  spec_frac = "Specular glare area (adaptive brightness-aware threshold)",
  haze_dcp = "Haze or turbidity (mean dark channel, 15 px patch)",
  red_dom = "Blood absorption proxy (R / (R + G + B))",
  illum_slope = "Illumination fall-off (slope of luminance against normalised radius)",
  lum_entropy = "Tonal information (Shannon entropy of luminance, bits)"
)
stopifnot(setequal(names(QUANTITY), MET))

fmt_metric <- function(x, m) if (m == "sharp_tenengrad") sprintf("%.2e", x) else fmt_num(x, 3)
fmt_iqr <- function(med, q1, q3, m) paste0(fmt_metric(med, m), " (", fmt_metric(q1, m), " to ", fmt_metric(q3, m), ")")
or_scale <- function(m) if (m %in% ASSOC_LOG_METRICS) "per SD of log value" else "per SD"
q3 <- function(v) as.numeric(stats::quantile(v, c(0.5, 0.25, 0.75), names = FALSE))

# ---- data ---------------------------------------------------------------------------------------

ft <- load_frame_table(with_predictions = FALSE)   # manifest + fov + optics + folds: no segmentation outcomes
n_fov_excluded <- attr(ft, "n_excluded_fov")
message("excluded ", n_fov_excluded, " labeled frame(s) without a detected field of view")
c1 <- ft |>
  filter(role == "cv") |>
  inner_join(filter(assoc_read_unet("center1_labeled"), unet_source == "oof"), by = "frame_id")
if (nrow(c1) != sum(ft$role == "cv")) stop("Center 1 frames without an out-of-fold U-Net row")
if (anyNA(c1$scenario)) stop("labeled frames without a scenario")
c1$fail <- as.integer(c1$unet_iou < thr)
c1$difficult <- as.integer(c1$scenario == "Difficult")
c2 <- filter(ft, role == "external")
c2$difficult <- as.integer(c2$scenario == "Difficult")
message("Center 1 OOF: ", nrow(c1), " frames, ", n_distinct(c1$patient_id), " patients, ", sum(c1$fail), " failures")

# ---- scenario contrast: medians, AUROC for Difficult, within-domain AUROC -----------------------

auroc_stats <- function(d, domains) {
  X <- as.matrix(d[MET])
  lab <- d$difficult
  dom <- d$domain
  fn <- function(rows) {
    Xr <- X[rows, , drop = FALSE]
    yr <- lab[rows]
    dr <- dom[rows]
    out <- list(pooled = apply(Xr, 2, assoc_auroc, yr), within = apply(Xr, 2, assoc_auroc_strat, yr, dr))
    for (g in domains) out[[g]] <- apply(Xr[dr == g, , drop = FALSE], 2, assoc_auroc, yr[dr == g])
    unlist(lapply(names(out), function(s) setNames(out[[s]], paste(s, MET, sep = "|"))))
  }
  r <- assoc_boot_ci(fn, d$patient_id, d$domain)
  r$group <- sub("\\|.*", "", r$stat)
  r$metric <- sub(".*\\|", "", r$stat)
  r[c("metric", "group", "est", "lo", "hi")]
}

auc_c1 <- auroc_stats(c1, sort(unique(c1$domain)))
auc_c2 <- if (nrow(c2)) auroc_stats(c2, sort(unique(c2$domain))) else NULL
auc_get <- function(tab, grp, m, col = "est") tab[[col]][tab$group == grp & tab$metric == m]

describe <- function(d) {
  do.call(rbind, lapply(MET, function(m) {
    n <- q3(d[[m]][d$difficult == 0])
    h <- q3(d[[m]][d$difficult == 1])
    data.frame(metric = m, n_normal = sum(d$difficult == 0), n_difficult = sum(d$difficult == 1),
               median_normal = n[1], q1_normal = n[2], q3_normal = n[3],
               median_difficult = h[1], q1_difficult = h[2], q3_difficult = h[3])
  }))
}
desc_c1 <- describe(c1)

by_domain <- do.call(rbind, c(
  lapply(sort(unique(c1$domain)), function(g) {
    d <- c1[c1$domain == g, ]
    cbind(domain = g, scope = "center1_oof", describe(d)[-1], metric = MET, patients = n_distinct(d$patient_id))
  }),
  if (nrow(c2)) lapply(sort(unique(c2$domain)), function(g) {
    d <- c2[c2$domain == g, ]
    cbind(domain = g, scope = "center2_external_descriptive", describe(d)[-1], metric = MET, patients = n_distinct(d$patient_id))
  })
))
by_domain$auroc <- NA_real_
by_domain$auroc_lo <- NA_real_
by_domain$auroc_hi <- NA_real_
for (i in seq_len(nrow(by_domain))) {
  tab <- if (by_domain$scope[i] == "center1_oof") auc_c1 else auc_c2
  for (col in c("est", "lo", "hi")) {
    by_domain[[c(est = "auroc", lo = "auroc_lo", hi = "auroc_hi")[[col]]]][i] <- auc_get(tab, by_domain$domain[i], by_domain$metric[i], col)
  }
}
by_domain <- by_domain[c("metric", "domain", "scope", "patients", "n_normal", "n_difficult", "median_normal", "q1_normal",
                         "q3_normal", "median_difficult", "q1_difficult", "q3_difficult", "auroc", "auroc_lo", "auroc_hi")]

# ---- adjusted odds ratios for failure -----------------------------------------------------------

diag_rows <- list()
or_rows <- list()
vif_rows <- list()

fit_model <- function(...) {
  note <- character()
  fit <- withCallingHandlers(assoc_fit_model(...), warning = function(w) {
    note <<- c(note, conditionMessage(w))
    message("WARNING: ", conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  fit$notes <- note
  fit
}

record <- function(fit, scope, model, terms, y, cl, cov_info) {
  tab <- fit$table[match(terms, fit$table$term), ]
  al <- rbind(cov_info$single_level, fit$aliased)
  al_txt <- if (nrow(al)) paste0(al$term, " (", al$reason, ")", collapse = "; ") else ""
  vm <- fit$vif
  base <- data.frame(scope = scope, model = model, n = length(y), events = sum(y), patients = n_distinct(cl),
                     patients_with_events = n_distinct(cl[y == 1]), n_params = fit$epv$n_params, epv = fit$epv$epv,
                     epv_flag = fit$epv$flag, method = fit$method, ci_method = fit$ci_method,
                     vif_max = max(vm), vif_max_term = names(which.max(vm)), aliased_terms = al_txt,
                     separation = fit$separation, note = paste(fit$notes, collapse = " | "))
  diag_rows[[length(diag_rows) + 1L]] <<- cbind(base[1:2], metric = if (length(terms) == 1L) terms else "all", base[-(1:2)])
  or_rows[[length(or_rows) + 1L]] <<- cbind(base[c("scope", "model")], metric = terms, base[c("n", "events", "patients", "patients_with_events",
    "n_params", "epv", "epv_flag", "method", "ci_method")],
    or_sd = tab$or, or_lo = tab$or_lo, or_hi = tab$or_hi, p = tab$p, se_robust = tab$se_robust, se_naive = tab$se_naive,
    se_ratio = tab$se_ratio, or_scale = unname(vapply(terms, or_scale, "")), aliased_terms = al_txt, separation = fit$separation)
  invisible(fit)
}

run_univariable <- function(d, Z, scope, y = d$fail) {
  cov <- assoc_covariates(d)
  for (m in MET) {
    fit <- fit_model(y, cbind(Z[, m, drop = FALSE], cov$X), d$patient_id, m, d$domain, label = paste(scope, m))
    record(fit, scope, "univariable_adjusted", m, y, d$patient_id, cov)
  }
  invisible(cov)
}

Z1 <- assoc_metric_matrix(c1, MET)
cov1 <- run_univariable(c1, Z1, "center1_oof")

joint1 <- fit_model(c1$fail, cbind(Z1, cov1$X), c1$patient_id, MET, c1$domain, label = "center1_oof joint")
record(joint1, "center1_oof", "joint_linear", MET, c1$fail, c1$patient_id, cov1)
vif_rows[[1]] <- data.frame(scope = "center1_oof", model = "joint_linear", term = names(joint1$vif), vif = unname(joint1$vif))

if (nrow(c2) && ext_open) {
  ex <- inner_join(c2, filter(assoc_read_unet("center2_labeled"), unet_source == "final"), by = "frame_id")
  if (nrow(ex) != nrow(c2)) stop("Center 2 frames without a final U-Net row")
  ex$fail <- as.integer(ex$unet_iou < thr)
  ex_events <- sum(ex$fail)
  ex_pw <- n_distinct(ex$patient_id[ex$fail == 1])
  ex_enough <- ex_events >= CFG$endpoints$min_external_failure_events &&
    ex_pw >= CFG$endpoints$min_external_patients_with_events
  message("Center 2: ", ex_events, " failures in ", ex_pw, " patients; ",
          if (ex_enough) "ORs estimated" else "below the minimum, ORs not estimated (descriptive only)")
  if (ex_enough) run_univariable(ex, assoc_metric_matrix(ex, MET, ref = Z1), "center2_external")

  pooled <- bind_rows(c1, ex)
  run_univariable(pooled, assoc_metric_matrix(pooled, MET, ref = Z1), "pooled_c1_c2")
}

diag_df <- do.call(rbind, diag_rows)
or_df <- do.call(rbind, or_rows)
vif_df <- do.call(rbind, vif_rows)

# ---- Table 2 (one row per metric; Center 1 inference) -------------------------------------------

pick_or <- function(scope, model, m, col) or_df[[col]][or_df$scope == scope & or_df$model == model & or_df$metric == m]
ci_txt <- function(e, l, h) ifelse(is.finite(e), fmt_ci(e, l, h, 2), "not estimable")

table2 <- do.call(rbind, lapply(MET, function(m) {
  ds <- desc_c1[desc_c1$metric == m, ]
  a <- vapply(c("est", "lo", "hi"), function(k) auc_get(auc_c1, "pooled", m, k), numeric(1))
  w <- vapply(c("est", "lo", "hi"), function(k) auc_get(auc_c1, "within", m, k), numeric(1))
  u <- vapply(c("or_sd", "or_lo", "or_hi", "p"), function(k) pick_or("center1_oof", "univariable_adjusted", m, k), numeric(1))
  j <- vapply(c("or_sd", "or_lo", "or_hi"), function(k) pick_or("center1_oof", "joint_linear", m, k), numeric(1))
  data.frame(
    metric = m, label = METRIC_LABELS[[m]], quantity = QUANTITY[[m]],
    n_normal = ds$n_normal, n_difficult = ds$n_difficult,
    median_normal = ds$median_normal, q1_normal = ds$q1_normal, q3_normal = ds$q3_normal,
    median_difficult = ds$median_difficult, q1_difficult = ds$q1_difficult, q3_difficult = ds$q3_difficult,
    normal_median_iqr = fmt_iqr(ds$median_normal, ds$q1_normal, ds$q3_normal, m),
    difficult_median_iqr = fmt_iqr(ds$median_difficult, ds$q1_difficult, ds$q3_difficult, m),
    auroc_difficult = a[1], auroc_difficult_lo = a[2], auroc_difficult_hi = a[3],
    auroc_text = ci_txt(a[1], a[2], a[3]),
    auroc_direction = if (a[1] >= 0.5) "higher in Difficult" else "lower in Difficult",
    auroc_within = w[1], auroc_within_lo = w[2], auroc_within_hi = w[3],
    or_sd = u[1], or_lo = u[2], or_hi = u[3], or_p = u[4], or_scale = or_scale(m),
    or_method = pick_or("center1_oof", "univariable_adjusted", m, "method"), or_text = ci_txt(u[1], u[2], u[3]),
    or_joint = j[1], or_joint_lo = j[2], or_joint_hi = j[3], or_joint_text = ci_txt(j[1], j[2], j[3]),
    or_joint_method = pick_or("center1_oof", "joint_linear", m, "method"),
    vif_joint = unname(joint1$vif[m]),
    stringsAsFactors = FALSE
  )
}))
table2 <- table2[c("metric", "label", "quantity", "n_normal", "n_difficult",
                   "median_normal", "q1_normal", "q3_normal", "median_difficult", "q1_difficult", "q3_difficult",
                   "normal_median_iqr", "difficult_median_iqr",
                   "auroc_difficult", "auroc_difficult_lo", "auroc_difficult_hi", "auroc_text", "auroc_direction",
                   "auroc_within", "auroc_within_lo", "auroc_within_hi",
                   "or_sd", "or_lo", "or_hi", "or_p", "or_scale", "or_method", "or_text",
                   "or_joint", "or_joint_lo", "or_joint_hi", "or_joint_text", "or_joint_method", "vif_joint")]

assoc_write_csv(table2, assoc_out("results", "stats", "assoc_table2.csv"))
assoc_write_csv(or_df, assoc_out("results", "stats", "assoc_or_models.csv"))
assoc_write_csv(by_domain, assoc_out("results", "stats", "assoc_by_domain.csv"))
assoc_write_csv(diag_df, assoc_out("results", "stats", "assoc_diagnostics.csv"))
assoc_write_csv(vif_df, assoc_out("results", "stats", "assoc_vif.csv"))

# ---- registry -----------------------------------------------------------------------------------

reg("assoc_failure_iou_threshold", thr, digits = 2)
reg("assoc_n_fov_excluded", as.integer(n_fov_excluded),
    note = "labeled frames dropped because no field of view was found (fov_found == FALSE); they have no optics or U-Net rows")
reg("assoc_n_frames_c1", nrow(c1))
reg("assoc_n_patients_c1", n_distinct(c1$patient_id))
reg("assoc_n_normal_c1", sum(c1$difficult == 0))
reg("assoc_n_difficult_c1", sum(c1$difficult == 1))
reg("assoc_n_failures_c1", sum(c1$fail))
reg("assoc_failure_rate_c1", mean(c1$fail), fmt = "percent")
reg("assoc_n_failure_patients_c1", n_distinct(c1$patient_id[c1$fail == 1]))
uni <- diag_df[diag_df$scope == "center1_oof" & diag_df$model == "univariable_adjusted", ]
jt <- diag_df[diag_df$scope == "center1_oof" & diag_df$model == "joint_linear", ]
reg("assoc_epv_univariable", min(uni$epv), digits = 1)
reg("assoc_epv_joint", jt$epv, digits = 1)
reg("assoc_joint_method", jt$method)
assoc_reg_num("assoc_vif_max_joint", jt$vif_max, digits = 1)
reg("assoc_vif_max_joint_term", jt$vif_max_term)
uni_or <- or_df[or_df$scope == "center1_oof" & or_df$model == "univariable_adjusted", ]
assoc_reg_num("assoc_se_ratio_median", stats::median(uni_or$se_ratio, na.rm = TRUE), digits = 2,
    note = "median over the 11 univariable adjusted models of cluster-robust SE / model-based SE; NA for ridge fits")
reg("assoc_n_or_excl_one", sum(uni_or$or_lo > 1 | uni_or$or_hi < 1, na.rm = TRUE))
reg("assoc_n_auroc_excl_half", sum(table2$auroc_difficult_lo > 0.5 | table2$auroc_difficult_hi < 0.5, na.rm = TRUE))
top <- table2[which.max(abs(table2$auroc_difficult - 0.5)), ]
reg("assoc_top_auroc_metric", top$label)
for (m in MET) {
  r <- table2[table2$metric == m, ]
  assoc_reg_ci(paste0("assoc_or_", m), r$or_sd, r$or_lo, r$or_hi, 2, note = paste("adjusted OR for failure", r$or_scale))
  assoc_reg_ci(paste0("assoc_or_joint_", m), r$or_joint, r$or_joint_lo, r$or_joint_hi, 2, note = paste("joint model OR", r$or_scale))
  assoc_reg_ci(paste0("assoc_auroc_", m), r$auroc_difficult, r$auroc_difficult_lo, r$auroc_difficult_hi, 2, note = "AUROC for Difficult, Center 1 pooled")
  assoc_reg_ci(paste0("assoc_auroc_within_", m), r$auroc_within, r$auroc_within_lo, r$auroc_within_hi, 2, note = "pair-weighted within-domain AUROC for Difficult, Center 1")
  reg(paste0("assoc_median_normal_", m), r$median_normal, digits = 3, fmt = function(v, d) fmt_metric(v, m))
  reg(paste0("assoc_median_difficult_", m), r$median_difficult, digits = 3, fmt = function(v, d) fmt_metric(v, m))
}

if (nrow(c2)) {
  reg("assoc_n_frames_c2", nrow(c2))
  reg("assoc_n_patients_c2", n_distinct(c2$patient_id))
  for (m in MET) {
    assoc_reg_ci(paste0("assoc_ext_auroc_", m), auc_get(auc_c2, "pooled", m), auc_get(auc_c2, "pooled", m, "lo"),
                 auc_get(auc_c2, "pooled", m, "hi"), 2, note = "AUROC for Difficult, Center 2 (descriptive)")
  }
}
if (nrow(c2) && ext_open) {
  reg("assoc_ext_n_failures", ex_events)
  reg("assoc_ext_n_failure_patients", ex_pw)
  reg("assoc_ext_or_estimated", if (ex_enough) "yes" else "no")
  if (ex_enough) {
    for (m in MET) {
      r <- or_df[or_df$scope == "center2_external" & or_df$metric == m, ]
      assoc_reg_ci(paste0("assoc_ext_or_", m), r$or_sd, r$or_lo, r$or_hi, 2, note = "adjusted OR for failure, Center 2 (descriptive)")
    }
  }
}

write_results_part("03_association")
message("03_association done: ", nrow(table2), " metrics")
