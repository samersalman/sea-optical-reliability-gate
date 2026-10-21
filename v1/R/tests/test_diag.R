# Tests of 34_diagnostics.R (post hoc diagnostics) and 21_supplement.R (Tables S1 to S3).
#   Rscript v1/R/tests/test_diag.R        (re-runs itself with OPTICS_DATA=synthetic)
# Everything is built from the synthetic data in work/synthetic/ and written under work/synthetic/out/work/tests/diag
# (the supplement tables land in work/synthetic/out/tables and are removed again). The real results, tables, lock and
# models are only READ, to prove they are untouched: the md5 of every existing results/stats CSV is compared with
# work/logs/stats_md5_before_fixpass.txt when that file exists, and the keys of 34 are checked against the real parts.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
r_dir <- dirname(.here)
for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R", "assoc_lib.R", "30_bundle.R", "04a_gate_fit_lock.R",
            "04b_external_eval.R", "07_stream.R", "08_sensitivity.R", "34_diagnostics.R", "21_supplement.R")) source(file.path(r_dir, f))

failures <- character()
check <- function(name, expr) {
  res <- tryCatch(if (isTRUE(expr)) "ok" else "FALSE", error = function(e) paste("error:", conditionMessage(e)))
  cat(sprintf("%-4s %s\n", if (res == "ok") "ok" else "FAIL", name))
  if (res != "ok") { cat("     ", res, "\n"); failures <<- c(failures, name) }
}
errs <- function(expr, pattern) {
  msg <- tryCatch({ force(expr); "" }, error = function(e) conditionMessage(e))
  nzchar(msg) && grepl(pattern, msg)
}
quiet <- function(expr) suppressMessages(suppressWarnings(expr))
near <- function(a, b, tol = 1e-12) isTRUE(all(abs(a - b) <= tol))

options(warn = 1)
B_TEST <- 200L
CORES <- 2L
base <- file.path("work", "tests", "diag")
unlink(p(base), recursive = TRUE)
mk_paths <- function(tag, derived = p("derived")) {
  list(derived = derived, models_rel = file.path(base, tag, "models"), lock = p(base, tag, "analysis_lock.yaml"),
       stats = p(base, tag, "stats"), results = p(base, tag, "root"))
}
rd <- function(paths, f) utils::read.csv(file.path(paths$stats, f))

real_lock <- file.path(V1, "config", "analysis_lock.yaml")
real_dirs <- c(file.path(V1, "results"), file.path(V1, "figures"), file.path(V1, "tables"), file.path(V1, "work", "models"))
snap_real <- function() list(lock = file.exists(real_lock), files = unlist(lapply(real_dirs, function(d) list.files(d, recursive = TRUE))))
real_before <- snap_real()
real_csv_md5 <- function() unname(tools::md5sum(list.files(file.path(V1, "results", "stats"), pattern = "\\.csv$", full.names = TRUE)))
md5_baseline <- file.path(V1, "work", "logs", "stats_md5_before_fixpass.txt")
real_lock_md5 <- if (file.exists(real_lock)) unname(tools::md5sum(real_lock)) else NA_character_
real_before_md5 <- tools::md5sum(list.files(file.path(V1, "results", "stats"), pattern = "\\.csv$", full.names = TRUE))

# ---- unit checks of the pieces ---------------------------------------------------------------------------------
set.seed(11)
toy_n <- c(500L, 100L)
toy <- data.frame(patient_id = c(paste0("P", rep(1:50, each = 10)), paste0("T", rep(1:10, each = 10))),
                  center = rep(1:2, toy_n), fold = c(rep(0:4, each = 100), rep(NA, 100)), stringsAsFactors = FALSE)
for (m in OPTICAL_METRICS) toy[[m]] <- stats::rnorm(nrow(toy))
toy$sharp_tenengrad <- exp(toy$sharp_tenengrad)
auc0 <- auroc_fail(diag_domain_scores(toy, OPTICAL_METRICS), toy$center == 2L)
check("domain classifier: two centers drawn from the same distribution cannot be told apart (AUROC near one half)", auc0 > 0.35 && auc0 < 0.65)
toy2 <- toy
toy2$lum_mean[toy2$center == 2L] <- toy2$lum_mean[toy2$center == 2L] + 2
auc1 <- auroc_fail(diag_domain_scores(toy2, OPTICAL_METRICS), toy2$center == 2L)
check("domain classifier: a shift of two SD in one metric is found (AUROC well above one half); patients never split across folds",
      auc1 > 0.85 && all(table(toy$patient_id, patient_foldid(toy$patient_id, 5)) %in% c(0L, 10L)))
sm_std <- diag_std_matrix(toy2, toy2[toy2$center == 1L, ])
check("diag_std_matrix: reference frames have mean 0 and SD 1 per metric (sharpness on the log scale)",
      near(colMeans(sm_std[toy2$center == 1L, ]), 0, 1e-9) && near(apply(sm_std[toy2$center == 1L, ], 2, sd), 1, 1e-9))

# ---- lock and models on synthetic data -------------------------------------------------------------------------------
pa <- mk_paths("a")
dir.create(dirname(pa$lock), recursive = TRUE)
writeLines(c("# frozen at G2", "failure_iou_threshold: 0.75", "epochs:", "  unet: 30", "  segformer: 40",
             "noninferiority_margin_aurc: 0.01", "lock_fields_sha256: \"deadbeef\""), pa$lock)
sm <- quiet(fit_and_lock(pa))
lock <- yaml::read_yaml(pa$lock)
models <- quiet(load_gate_models(lock))
thr <- lock_failure_threshold(lock)
invisible(quiet(run_sensitivity(pa, B_TEST, CORES)))                    # writes sensitivity.csv, which the re-fit check compares with
invisible(quiet(run_stream(pa, B_TEST, CORES)))                         # writes stream_summary.csv, compared with the flag shares
invisible(quiet(run_external_eval(pa, B_TEST, CORES)))                  # gate_internal.csv, gate_external.csv for Table S2
t_34 <- system.time(res <- quiet(run_diagnostics(pa, B_TEST, CORES)))[["elapsed"]]

# ---- output files and schemas ------------------------------------------------------------------------------------------
files <- c("diag_covariate_shift.csv", "diag_gate_nonzero.csv", "diag_flag_shares.csv", "diag_within_domain.csv", "diag_or_robustness.csv",
           "diag_calibration.csv")
check("run_diagnostics writes the six diag_ CSVs under the stats folder, and nothing outside diag_ plus the inputs", all(file.exists(file.path(pa$stats, files))) &&
        all(grepl("^diag_", setdiff(list.files(pa$stats), c("sensitivity.csv", "stream_flags.csv", "stream_runs.csv", "stream_summary.csv",
                                                              "gate_internal.csv", "gate_external.csv", "gate_paired_diffs.csv",
                                                              "risk_coverage_curves.csv", "gate_range_check.csv")))))
sh <- rd(pa, "diag_covariate_shift.csv")
SHIFT_COLS <- c("scope", "metric", "n_c1", "n_c2", "median_c1", "median_c2", "sd_c1", "std_median_diff", "p05_c1", "p95_c1", "min_c1", "max_c1",
                "share_c2_below_p05", "share_c2_above_p95", "share_c2_beyond_p05_p95", "share_c2_below_min", "share_c2_above_max",
                "mean_excursion_below_min_sd", "mean_excursion_above_max_sd", "auroc_c2_vs_c1", "value", "lo", "hi", "note")
check("diag_covariate_shift.csv: documented columns; 11 metric rows then 7 multivariate rows", identical(names(sh), SHIFT_COLS) &&
        identical(sh$metric[sh$scope == "metric"], OPTICAL_METRICS) && sum(sh$scope == "multivariate") == 7L)
nz <- rd(pa, "diag_gate_nonzero.csv")
NZ_COLS <- c("scope", "id", "label", "model_family", "eval_set", "n_train_frames", "events_train", "n_candidate_terms", "n_nonzero_terms",
             "n_nonzero_metrics", "nonzero_metrics", "n_unpenalised", "lambda", "events_per_nonzero", "events_per_candidate", "score_sd_eval",
             "score_range_eval", "constant_score", "aurc_refit", "aurc_sensitivity_csv", "aurc_abs_diff", "reproduced", "note")
check("diag_gate_nonzero.csv: documented columns; one final-gate row, one row per fold gate, one per sensitivity variant (11)",
      identical(names(nz), NZ_COLS) && sum(nz$scope == "final_gate") == 1L && sum(nz$scope == "fold_gate") == 5L &&
        identical(nz$id[nz$scope == "variant"], SENS_VARIANTS$variant))
fl <- rd(pa, "diag_flag_shares.csv")
check("diag_flag_shares.csv: documented columns", identical(names(fl), c("scope", "statistic", "coverage", "est", "lo", "hi", "n")))
wd <- rd(pa, "diag_within_domain.csv")
check("diag_within_domain.csv: documented columns; every domain of Center 1 with the seven methods",
      identical(names(wd), c("domain", "method", "n_frames", "n_patients", "n_events", "n_patients_with_events", "failure_rate", "aurc", "aurc_lo",
                             "aurc_hi", "auroc", "auroc_lo", "auroc_hi", "sel_risk_80", "sel_risk_80_lo", "sel_risk_80_hi", "diff_aurc_vs_random",
                             "diff_aurc_vs_random_lo", "diff_aurc_vs_random_hi")) &&
        identical(sort(unique(wd$domain)), c("D1", "D2")) && all(table(wd$domain) == 7L))
orb <- rd(pa, "diag_or_robustness.csv")
check("diag_or_robustness.csv: documented columns, one row per metric",
      identical(names(orb), c("metric", "label", "or_scale", "n", "events", "patients", "or_primary", "or_lo_wald", "or_hi_wald", "or_lo_boot",
                              "or_hi_boot", "boot_valid", "or_gtarea", "or_lo_gtarea", "or_hi_gtarea", "excl1_wald", "excl1_boot", "excl1_gtarea",
                              "ci_width_ratio_boot_over_wald")) && identical(orb$metric, OPTICAL_METRICS))
cal <- rd(pa, "diag_calibration.csv")
check("diag_calibration.csv: documented columns, three scopes",
      identical(names(cal), c("scope", "n_frames", "n_patients", "n_events", "observed_rate", "observed_lo", "observed_hi", "mean_predicted",
                              "mean_predicted_lo", "mean_predicted_hi", "offset", "offset_lo", "offset_hi", "sd_predicted", "iqr_predicted",
                              "p05_predicted", "p95_predicted")) && identical(cal$scope, c("center1_oof", "center1_final_in_sample", "center2_final")))

# ---- 1 covariate shift: recomputed from the frames --------------------------------------------------------------------
ft <- quiet(load_frame_table(with_predictions = FALSE))
c1 <- ft[ft$role == "cv", ]
c2 <- ft[ft$role == "external", ]
pm <- sh[sh$scope == "metric", ]
m1 <- "under_frac"
r1 <- pm[pm$metric == m1, ]
check("shift: medians, standardized difference and the shares beyond the 5th/95th percentile equal an independent computation",
      near(r1$median_c1, median(c1[[m1]])) && near(r1$median_c2, median(c2[[m1]])) &&
        near(r1$std_median_diff, (median(c2[[m1]]) - median(c1[[m1]])) / sd(c1[[m1]])) &&
        near(r1$share_c2_below_p05, mean(c2[[m1]] < quantile(c1[[m1]], 0.05))) && near(r1$share_c2_above_p95, mean(c2[[m1]] > quantile(c1[[m1]], 0.95))) &&
        near(r1$share_c2_beyond_p05_p95, r1$share_c2_below_p05 + r1$share_c2_above_p95))
rc <- range_check(models$gate, c2)
check("shift: the share below / above the Center 1 range equals the gate's own range check (training range = Center 1 analyzed range)",
      near(pm$share_c2_below_min, rc$frac_below[match(pm$metric, rc$feature)]) && near(pm$share_c2_above_max, rc$frac_above[match(pm$metric, rc$feature)], 1e-12))
ex <- pmax(0, min(c1[[m1]]) - c2[[m1]])
check("shift: mean excursion below the Center 1 minimum is in Center 1 SD units among the frames below it",
      if (any(ex > 0)) near(r1$mean_excursion_below_min_sd, mean(ex[ex > 0]) / sd(c1[[m1]])) else is.na(r1$mean_excursion_below_min_sd))
check("shift: univariate AUROC of a metric for Center 2 vs Center 1 equals the rank statistic",
      near(pm$auroc_c2_vs_c1, vapply(OPTICAL_METRICS, function(m) auroc_fail(c(c1[[m]], c2[[m]]), rep(c(FALSE, TRUE), c(nrow(c1), nrow(c2)))), 0)))
mv <- function(n) sh[sh$scope == "multivariate" & sh$metric == n, ]
check("shift: the synthetic Center 2 optics are shifted, so the domain classifier separates the centers and Center 2 lies farther out in Mahalanobis distance",
      mv("domain_auroc")$value > 0.9 && mv("domain_auroc")$lo <= mv("domain_auroc")$value && mv("domain_auroc")$value <= mv("domain_auroc")$hi &&
        mv("mahalanobis_ratio_c2_over_c1")$value > 1 && near(mv("mahalanobis_ratio_c2_over_c1")$value, mv("mahalanobis_median_c2")$value / mv("mahalanobis_median_c1_oof")$value) &&
        mv("domain_auroc_excl_top3")$value <= mv("domain_auroc")$value + 0.02)
check("shift: the 'excluding top 3' note names the three most separating metrics",
      identical(sort(strsplit(sub("^without ", "", mv("domain_auroc_excl_top3")$note), ", ")[[1]]),
                sort(pm$metric[order(-abs(pm$auroc_c2_vs_c1 - 0.5))][1:3])))

# ---- 2 gate non-zero coefficients and the variant re-fits ---------------------------------------------------------------
g <- models$gate
b_fin <- as.matrix(coef(g$fit, s = g$lambda))[-1, 1]
fr <- nz[nz$scope == "final_gate", ]
check("non-zero: the final gate's counts equal the non-zero coefficients of the stored fit at its lambda; metrics are the distinct metrics behind them",
      fr$n_nonzero_terms == sum(b_fin != 0) && fr$n_candidate_terms == length(b_fin) && near(fr$lambda, g$lambda) &&
        fr$n_nonzero_metrics == length(unique(sub("_[0-9]+$", "", names(b_fin)[b_fin != 0]))) &&
        near(fr$events_per_candidate, fr$events_train / fr$n_candidate_terms) && fr$events_train == sum(sm$n_events))
fo_obj <- models$folds$optical_gate
check("non-zero: each fold gate's count equals its stored fit; a fold gate with no non-zero term has a constant held-out score",
      all(vapply(names(fo_obj), function(k) {
        gf <- fo_obj[[k]]
        row <- nz[nz$id == paste0("fold", k), ]
        n <- sum(as.matrix(coef(gf$fit, s = gf$lambda))[-1, 1] != 0)
        row$n_nonzero_terms == n && identical(row$constant_score, n == 0)
      }, NA)))
va <- nz[nz$scope == "variant", ]
check("non-zero: every sensitivity variant re-fitted by 08's own code reproduces sensitivity.csv to 1e-12 (locked, device_norm, background_only, lodo_d2_to_d3 required)",
      all(va$reproduced) && all(va$aurc_abs_diff <= 1e-12) && all(DIAG_REPRO_REQUIRED %in% va$id))
spl <- va$model_family == "glmnet" & va$id != "domain_cov"
check("non-zero: for the penalised gates a constant score is exactly the case of zero non-zero terms; a constant score has the random AURC",
      identical(va$constant_score[va$model_family == "glmnet"], va$n_nonzero_terms[va$model_family == "glmnet"] == 0) &&
        all(va$score_range_eval[va$constant_score] < 1e-12) && all(va$score_range_eval[!va$constant_score] > 1e-12))
se <- rd(pa, "sensitivity.csv")
cst <- va$id[va$constant_score]
check("non-zero: variants flagged constant have a gate AURC equal to the random AURC in sensitivity.csv (and a zero paired difference)",
      all(vapply(cst, function(v) near(se$aurc[se$variant == v & se$method == "gate"], se$aurc[se$variant == v & se$method == "random"], 1e-12) &&
                   near(se$diff_vs_random[se$variant == v & se$method == "gate"], 0, 1e-12), NA)))
bad <- se
bad$aurc[bad$variant == "locked" & bad$method == "gate"] <- bad$aurc[bad$variant == "locked" & bad$method == "gate"] + 1e-6
c1s <- quiet(read_sens_frames(pa$derived, "center1_labeled")); c2s <- quiet(read_sens_frames(pa$derived, "center2_labeled"))
c1s$fail <- c1s$iou < thr; c2s$fail <- c2s$iou < thr
check("non-zero: a sensitivity.csv that the re-fit cannot reproduce stops the run instead of passing silently",
      errs(quiet(diag_gate_nonzero(models, c1s, c2s, bad)), "do not reproduce sensitivity.csv"))
check("non-zero: the unpenalized domain indicator of domain_cov is not counted as a penalized coefficient",
      va$n_unpenalised[va$id == "domain_cov"] == 1L && va$n_candidate_terms[va$id == "domain_cov"] == va$n_candidate_terms[va$id == "locked"])

# ---- 3 flag shares ------------------------------------------------------------------------------------------------------
d1 <- quiet(read_eval_frames(pa$derived, "center1_labeled", "unet"))
d2 <- quiet(read_eval_frames(pa$derived, "center2_labeled", "unet"))
thrs <- g$flag_thresholds
fin1 <- predict_gate(g, d1, "link")
oof1 <- predict_oof(models$folds$optical_gate, d1, d1$fold)
fin2 <- predict_gate(g, d2, "link")
fs <- function(sc, cov) fl$est[fl$scope == sc & fl$statistic == "flagged_share" & fl$coverage %in% cov]
check("flags: the final gate's share of Center 1 frames above each locked cutoff equals the direct count (in-sample)",
      near(fs("center1_final", 90), mean(fin1 > thrs[["0.9"]])) && near(fs("center1_final", 80), mean(fin1 > thrs[["0.8"]])))
check("flags: out-of-fold shares are the nominal 10% and 20% (by construction); Center 2 equals the direct count",
      near(fs("center1_oof", 90), mean(oof1 > thrs[["0.9"]])) && fs("center1_oof", 90) <= 0.1 + 1e-9 && fs("center1_oof", 90) > 0.08 &&
        near(fs("center2_final", 90), mean(fin2 > thrs[["0.9"]])) && near(fs("center2_final", 80), mean(fin2 > thrs[["0.8"]])))
su <- rd(pa, "stream_summary.csv")
sov <- function(s, col = "est") su[[col]][su$scope == "overall" & su$statistic == s]
check("flags: the Center 2 and stream shares (estimates and intervals) equal the locked 07 stream summary at both cutoffs",
      near(fs("stream_final", 90), sov("flagged_share_stream_90")) && near(fs("stream_final", 80), sov("flagged_share_stream_80")) &&
        near(fs("center2_final", 90), sov("flagged_share_center2_90")) && near(fs("center2_final", 80), sov("flagged_share_center2_80")) &&
        near(fl$lo[fl$scope == "stream_final" & fl$statistic == "flagged_share" & fl$coverage == 90], sov("flagged_share_stream_90", "lo"), 1e-9) &&
        near(fl$hi[fl$scope == "center2_final" & fl$statistic == "flagged_share" & fl$coverage == 80], sov("flagged_share_center2_80", "hi"), 1e-9) &&
        near(fs("center1_oof", 80), sov("flagged_share_center1_oof_80")))
check("flags: band width is the distance between the two cutoffs and the band share is the share between them",
      near(fl$est[fl$statistic == "band_width_logodds"], thrs[["0.9"]] - thrs[["0.8"]]) &&
        near(fl$est[fl$scope == "center1_final" & fl$statistic == "band_share"], fs("center1_final", 80) - fs("center1_final", 90)) &&
        near(fl$est[fl$scope == "center1_final" & fl$statistic == "band_share"], mean(fin1 > thrs[["0.8"]] & fin1 <= thrs[["0.9"]])) &&
        near(fl$est[fl$scope == "center1_final" & fl$statistic == "band_width_over_score_sd"], (thrs[["0.9"]] - thrs[["0.8"]]) / sd(fin1)))
check("flags: spread of the final-gate scores and of the pooled out-of-fold scores (SD and IQR of the log-odds), plus one SD row per fold",
      near(fl$est[fl$scope == "center1_final" & fl$statistic == "score_sd"], sd(fin1)) && near(fl$est[fl$scope == "center1_oof" & fl$statistic == "score_sd"], sd(oof1)) &&
        near(fl$est[fl$scope == "center1_oof" & fl$statistic == "score_iqr"], diff(quantile(oof1, c(0.25, 0.75)))) &&
        sum(grepl("^center1_oof_fold", fl$scope) & fl$statistic == "score_sd") == 5L)

# ---- 4 within-domain ---------------------------------------------------------------------------------------------------
risk1 <- 1 - d1$iou
sc1 <- score_methods(models, d1, lock$entropy_baseline_choice, oof = TRUE)
dd <- wd[wd$domain == "D1", ]
i1 <- d1$domain == "D1"
check("within-domain: n, events, random AURC (mean risk) and gate AURC / AUROC equal a direct computation on the D1 subset",
      dd$n_frames[1] == sum(i1) && dd$n_events[1] == sum(d1$iou[i1] < thr) && near(dd$aurc[dd$method == "random"], mean(risk1[i1])) &&
        near(dd$aurc[dd$method == "optical_gate"], aurc(sc1$s_optical_gate[i1], risk1[i1])) &&
        near(dd$auroc[dd$method == "optical_gate"], auroc_fail(sc1$s_optical_gate[i1], d1$iou[i1] < thr)) &&
        near(dd$aurc[dd$method == "oracle"], oracle_aurc(risk1[i1])))
check("within-domain: the paired difference to random is the point difference, zero for random itself, and its interval is ordered",
      near(dd$diff_aurc_vs_random, dd$aurc - dd$aurc[dd$method == "random"]) && dd$diff_aurc_vs_random_lo[dd$method == "random"] == 0 &&
        all(dd$diff_aurc_vs_random_lo <= dd$diff_aurc_vs_random_hi) && all(dd$aurc_lo <= dd$aurc & dd$aurc <= dd$aurc_hi))
check("within-domain: the two domains partition Center 1 (frames add up)", sum(wd$n_frames[wd$method == "random"]) == nrow(d1))

# ---- 5 odds-ratio robustness -------------------------------------------------------------------------------------------
fo <- quiet(diag_read_or_frames(pa$derived))
y <- as.integer(fo$iou < thr)
Z <- assoc_metric_matrix(fo, OPTICAL_METRICS)
covp <- assoc_covariates(fo, area = "pred_area_frac")
covg <- assoc_covariates(fo, area = "gt_area_frac")
m <- "haze_dcp"
fit_m <- function(cov) glm(y ~ ., data = data.frame(y = y, x = Z[, m], cov$X), family = binomial())
check("odds ratios: the primary OR of one metric equals a plain logistic fit with the predicted-area covariate; the ground-truth variant uses gt_area_frac",
      near(orb$or_primary[orb$metric == m], exp(coef(fit_m(covp))[["x"]]), 1e-8) && near(orb$or_gtarea[orb$metric == m], exp(coef(fit_m(covg))[["x"]]), 1e-8) &&
        !isTRUE(all.equal(covp$X[, "pred_area_z"], covg$X[, "pred_area_z"])))
check("odds ratios: Wald intervals ordered around the estimate; bootstrap intervals use every resample and bracket the estimate for most metrics",
      all(orb$or_lo_wald <= orb$or_primary & orb$or_primary <= orb$or_hi_wald) && all(orb$boot_valid == B_TEST) &&
        mean(orb$or_lo_boot <= orb$or_primary & orb$or_primary <= orb$or_hi_boot) >= 0.8)
check("odds ratios: the exclusion flags are the interval-versus-one rule, and the summary counts equal them",
      identical(orb$excl1_wald, orb$or_lo_wald > 1 | orb$or_hi_wald < 1) && identical(orb$excl1_boot, orb$or_lo_boot > 1 | orb$or_hi_boot < 1) &&
        identical(orb$excl1_gtarea, orb$or_lo_gtarea > 1 | orb$or_hi_gtarea < 1))
check("odds ratios: a stale assoc_or_models.csv (different primary ORs) stops the run",
      errs(quiet(diag_or_robustness(fo, thr, 20L, 1L, data.frame(scope = "center1_oof", model = "univariable_adjusted", metric = OPTICAL_METRICS,
                                                                    or_sd = 1.5, or_lo = 1, or_hi = 2, stringsAsFactors = FALSE))), "differ from assoc_or_models.csv"))
check("odds ratios: n, events and patients describe the Center 1 out-of-fold frames",
      all(orb$patients == length(unique(fo$patient_id))) && all(orb$n == nrow(fo)) && all(orb$events == sum(y)))

# ---- 6 calibration -----------------------------------------------------------------------------------------------------
cc <- function(s) cal[cal$scope == s, ]
check("calibration: observed rate and mean predicted probability equal direct computations; offset = predicted - observed",
      near(cc("center2_final")$observed_rate, mean(d2$iou < thr)) && near(cc("center2_final")$mean_predicted, mean(plogis(fin2))) &&
        near(cc("center1_oof")$mean_predicted, mean(plogis(oof1))) && near(cc("center1_final_in_sample")$mean_predicted, mean(plogis(fin1))) &&
        near(cal$offset, cal$mean_predicted - cal$observed_rate) && near(cc("center2_final")$sd_predicted, sd(plogis(fin2))))
check("calibration: the final gate fitted on Center 1 has a near-zero offset on Center 1 (penalised, so not exactly zero)", abs(cc("center1_final_in_sample")$offset) < 0.02)

# ---- registry ----------------------------------------------------------------------------------------------------------
quiet(register_34(res))
guard_write(file.path(pa$results, "results", "parts", "34_diagnostics.json"))
write_results_part("34_diagnostics", root = pa$results)
part <- jsonlite::read_json(file.path(pa$results, "results", "parts", "34_diagnostics.json"))
keys <- names(part$entries)
check("34 registers only diag_ keys, in the six families the task names", length(keys) > 90L && all(grepl("^diag_", keys)) &&
        all(c("diag_shift_under_below_min", "diag_shift_domain_auroc", "diag_nz_final_terms", "diag_nz_n_const_variants", "diag_nz_const_variants",
              "diag_flag_c1_final_share_90", "diag_flag_c2_share_80", "diag_flag_band_width", "diag_dom_d1_aurc_gate", "diag_dom_d2_diff_aurc_gate_random_ci",
              "diag_or_n_excl1_wald", "diag_or_n_excl1_boot", "diag_or_n_excl1_gtarea", "diag_or_sharp_tenengrad_gtarea_ci", "diag_or_contrast_local_gtarea_ci",
              "diag_or_red_dom_gtarea_ci", "diag_calib_c1_mean_pred", "diag_calib_c2_offset_ci") %in% keys))
real_parts <- list.files(file.path(V1, "results", "parts"), pattern = "\\.json$", full.names = TRUE)
real_parts <- real_parts[basename(real_parts) != "34_diagnostics.json"]
real_keys <- unlist(lapply(real_parts, function(f) names(jsonlite::read_json(f)$entries)))
check("34 keys collide with no key of the real parts", !any(keys %in% real_keys))
rj <- part$entries
check("registered displays: shares as percent, counts as integers, intervals as '(95% CI a to b)', the constant-variant list as a string",
      grepl("%$", rj$diag_shift_under_below_min$display) && grepl("^[0-9,]+$", rj$diag_nz_final_terms$display) &&
        grepl("^-?[0-9.]+ \\(95% CI -?[0-9.]+ to -?[0-9.]+\\)$", rj$diag_shift_domain_auroc_ci$display) && is.character(rj$diag_nz_const_variants$display) &&
        grepl("%$", rj$diag_flag_band_share_c2$display))
check("registered raw values equal the CSV values",
      near(unlist(rj$diag_shift_under_below_min$raw), pm$share_c2_below_min[pm$metric == "under_frac"]) &&
        near(unlist(rj$diag_nz_final_terms$raw), fr$n_nonzero_terms) && near(unlist(rj$diag_flag_c1_final_share_90$raw), fs("center1_final", 90)) &&
        near(unlist(rj$diag_nz_n_const_variants$raw), sum(va$constant_score)) && near(unlist(rj$diag_or_n_excl1_boot$raw), sum(orb$excl1_boot)))
role_lab <- function(x) gsub("Center 2", tolower(CENTER_ROLE[["2"]]), gsub("Center 1", tolower(CENTER_ROLE[["1"]]), x, fixed = TRUE), fixed = TRUE)
check("diag_nz_const_variants lists the labels of the constant variants (center names as the exhibits print them), or none",
      identical(rj$diag_nz_const_variants$display, if (any(va$constant_score)) paste(role_lab(va$label[va$constant_score]), collapse = "; ") else "none"))

# ---- lock integrity ----------------------------------------------------------------------------------------------------
gate_file <- p(lock$gate_rds_path)
orig <- readBin(gate_file, "raw", file.size(gate_file))
g_mod <- readRDS(gate_file)
g_mod$flag_thresholds <- g_mod$flag_thresholds - 5
saveRDS(g_mod, gate_file)
pt <- pa
pt$stats <- p(base, "tamper_stats")
pt$derived <- p(base, "no_such_dir")
check("34 stops with a SHA-256 mismatch after the gate RDS is modified, before any data are read, and writes nothing",
      errs(quiet(run_diagnostics(pt, 20L, 1L)), "SHA-256 mismatch") && !dir.exists(pt$stats))
writeBin(orig, gate_file)
lines <- readLines(pa$lock)
pc <- pt
pc$lock <- p(base, "tamper_threshold.yaml")
writeLines(sub("^failure_iou_threshold: .*", "failure_iou_threshold: 0.5", lines), pc$lock)
check("a lock edited after locking (failure threshold) stops 34 on lock_fields_sha256", errs(quiet(run_diagnostics(pc, 20L, 1L)), "lock_fields_sha256 mismatch"))

# ---- supplement tables -------------------------------------------------------------------------------------------------
# the builders read p("results", "stats"); point them at the test folder (own CSVs) and fall back to the synthetic out folder for the
# association CSVs this test does not regenerate
sup_read <- function(name, dir = NULL) {
  for (d in c(pa$stats, p("results", "stats"))) {
    path <- file.path(d, name)
    if (file.exists(path)) return(utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = c("NA", "")))
  }
  stop("missing input ", name, call. = FALSE)
}
tdir <- p("tables")
made <- list()
for (nm in c("TableS1", "TableS2", "TableS3")) {
  tb <- quiet(switch(nm, TableS1 = sup_table_s1(), TableS2 = sup_table_s2(), TableS3 = sup_table_s3()))
  made[[nm]] <- list(tb = tb, paths = quiet(sup_save(nm, tb$panels, tb$general)))
}
n_tables <- function(path) { s <- officer::docx_summary(officer::read_docx(path)); max(s$table_index, na.rm = TRUE) }
doc_text <- function(path) paste(officer::docx_summary(officer::read_docx(path))$text, collapse = " ")
check("Tables S1 to S3 are written as docx and csv (one flextable per panel: 1, 7 and 3)",
      all(vapply(made, function(x) all(file.exists(x$paths)), NA)) && n_tables(made$TableS1$paths[["docx"]]) == 1L &&
        n_tables(made$TableS2$paths[["docx"]]) == 7L && n_tables(made$TableS3$paths[["docx"]]) == 3L)
txt <- lapply(made, function(x) doc_text(x$paths[["docx"]]))
check("every table says it is a post hoc diagnostic (in the footnotes of the docx)", all(vapply(txt, function(t) grepl("Post hoc diagnostic", t, fixed = TRUE), NA)))
notes <- lapply(made, function(x) paste(c(unlist(lapply(x$tb$panels, `[[`, "footnote")), x$tb$general), collapse = " "))
check("footnotes explain the abbreviations each table uses (IQR and AUROC in S1; AURC, AUROC, IoU, CI, SD in S2; AUROC, OR, SD, CI in S3)",
      all(vapply(c("IQR, interquartile", "AUROC, area under", "CI, confidence"), grepl, NA, notes$TableS1, fixed = TRUE)) &&
        all(vapply(c("AURC, area under", "AUROC, area under", "IoU, intersection", "CI, confidence", "SD, standard deviation"), grepl, NA, notes$TableS2, fixed = TRUE)) &&
        all(vapply(c("AUROC", "OR, odds ratio", "SD, standard deviation", "CI, confidence"), grepl, NA, notes$TableS3, fixed = TRUE)))
s1 <- read.csv(made$TableS1$paths[["csv"]], header = FALSE, stringsAsFactors = FALSE)
bd <- sup_read("assoc_by_domain.csv")
r_l <- bd[bd$metric == "lum_mean" & bd$domain == "D1", ]
check("Table S1 csv: a cell equals the formatted source value (median [IQR] and the within-domain AUROC) and every metric appears",
      s1[3, 2] == sup_metric_iqr(r_l$median_normal, r_l$q1_normal, r_l$q3_normal) && s1[3, 4] == sup_est_ci(r_l$auroc, r_l$auroc_lo, r_l$auroc_hi, 2) &&
        all(METRIC_LABELS %in% s1[[1]]) && !any(s1 == "NA"))
s2 <- read.csv(made$TableS2$paths[["csv"]], header = FALSE, stringsAsFactors = FALSE)
check("Table S2 csv: panels stacked with a title row; the constant-score flags equal diag_gate_nonzero.csv; no NA text",
      sum(grepl("^Panel [A-G]\\.", s2[[1]])) == 7L && !any(s2 == "NA") &&
        identical(s2[[4]][3:13], sup_yes(va$constant_score)))
s3 <- read.csv(made$TableS3$paths[["csv"]], header = FALSE, stringsAsFactors = FALSE)
check("Table S3 csv: three panels, 11 metric rows in each of A and C", sum(grepl("^Panel [A-C]\\.", s3[[1]])) == 3L && !any(s3 == "NA") &&
        sum(s3[[1]] %in% METRIC_LABELS) + sum(s3[[1]] == paste0(METRIC_LABELS[["sharp_tenengrad"]], "§")) >= 22L)
for (x in made) invisible(file.remove(x$paths))

# ---- the real files were not touched ------------------------------------------------------------------------------------
check("the real lock, results, figures, tables and models folders were not changed by this test", identical(snap_real(), real_before) &&
        identical(unname(tools::md5sum(real_lock)), real_lock_md5))
check("the md5 of every real results/stats CSV is unchanged by this test", identical(tools::md5sum(list.files(file.path(V1, "results", "stats"), pattern = "\\.csv$", full.names = TRUE)),
                                                                                    real_before_md5))
if (file.exists(md5_baseline)) {
  base_md5 <- readLines(md5_baseline)
  nm <- sub("^MD5 \\((.*)\\) = .*$", "\\1", base_md5)
  val <- sub("^.* = ", "", base_md5)
  cur <- tools::md5sum(file.path(V1, "results", "stats", nm))
  check("every results/stats CSV that existed before the fix pass is byte-identical to its recorded md5 (work/logs/stats_md5_before_fixpass.txt)", all(unname(cur) == val))
}

cat(sprintf("\nruntime (synthetic, B = %d): 34 %.1f s\n", B_TEST, t_34))
cat(sprintf("%d failure(s)\n", length(failures)))
if (length(failures)) quit(status = 1)
