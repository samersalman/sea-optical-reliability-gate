# 34: post hoc DIAGNOSTICS that explain why the locked gate did not transport. Descriptive only: nothing here changes,
# re-estimates or replaces a locked quantity (gate, cutoffs, threshold, entropy choice), and every number is labelled
# post hoc. The lock is verified (lock_fields_sha256 and the RDS hashes) before any Center 2 row is read. Patient-clustered
# intervals use the plan's B and bootstrap seed, so paired quantities line up with 04b and 07.
#   Rscript R/34_diagnostics.R            (OPTICS_DATA=synthetic for development; skipped when the synthetic lock is not sealed)
# Diagnostics (CSV under results/stats/, keys in results/parts/34_diagnostics.json)
#   1 covariate shift          diag_covariate_shift.csv   diag_shift_*   per-metric shift, domain classifier, Mahalanobis
#   2 gate fit                 diag_gate_nonzero.csv      diag_nz_*      non-zero penalised coefficients: final, fold and variant gates
#   3 flag shares              diag_flag_shares.csv       diag_flag_*    final gate at the locked cutoffs, spread of the scores
#   4 within-domain internal   diag_within_domain.csv     diag_dom_*     D1 and D2 separately, Center 1 out-of-fold
#   5 odds-ratio robustness    diag_or_robustness.csv     diag_or_*      bootstrap vs Wald CI, ground-truth area covariate
#   6 calibration              diag_calibration.csv       diag_calib_*   mean predicted vs observed failure rate
# The sensitivity variants of step 2 are re-fitted by calling 08_sensitivity.R's own score_variant() with its fitters wrapped
# to hand back the fitted object, so the fits are the ones 08 makes; their AURC is compared with sensitivity.csv.

if (!exists("score_variant", mode = "function") || !exists("read_stream_frames", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R", "assoc_lib.R", "07_stream.R", "08_sensitivity.R")) {
      source(file.path(r_dir, f), chdir = FALSE)
    }
  }, envir = globalenv())
}

DIAG_STATUS <- "post hoc diagnostic; descriptive; not part of the locked analysis"
DIAG_FOLDS <- CFG$segmenters$cv_folds
DIAG_REPRO_TOL <- 1e-12                                    # AURC of a re-fitted variant vs sensitivity.csv
DIAG_REPRO_REQUIRED <- c("locked", "device_norm", "background_only", "lodo_d2_to_d3")
DIAG_CONST_TOL <- 1e-12                                    # a score whose range is below this is constant
DIAG_GAM_EDF_MIN <- 0.1                                    # a GAM smooth with fewer effective df counts as shrunk to zero

# ---- data -----------------------------------------------------------------------------------------------------

# Labeled frames of both centers with the 11 metrics, patient, domain and Center 1 fold. Optics and manifest only: no
# segmentation outcome is read here. Frames without a detected field of view are dropped, as everywhere else.
diag_read_optics <- function(derived) {
  man <- .read_columns(file.path(derived, "manifest.parquet"), c("frame_id", "partition", "patient_id", "domain"))
  man <- man[man$partition %in% c("center1_labeled", "center2_labeled"), ]
  fov <- .read_columns(file.path(derived, "fov.parquet"), c("frame_id", "fov_found"))
  man$fov_found <- fov$fov_found[match(man$frame_id, fov$frame_id)]
  if (anyNA(man$fov_found)) stop("fov.parquet lacks rows for some labeled frames")
  if (isTRUE(CFG$fov$exclude_not_found)) man <- man[man$fov_found, ]
  op <- .read_columns(file.path(derived, "optics.parquet"), c("frame_id", OPTICAL_METRICS))
  d <- cbind(man[c("frame_id", "partition", "patient_id", "domain")], op[match(man$frame_id, op$frame_id), OPTICAL_METRICS])
  if (anyNA(d[[OPTICAL_METRICS[1]]]) || any(!is.finite(as.matrix(d[OPTICAL_METRICS])))) stop("optics missing or non-finite for analysed labeled frames")
  fo <- utils::read.csv(file.path(derived, "folds.csv"), stringsAsFactors = FALSE)
  d$fold <- fo$fold[match(d$patient_id, fo$patient_id)]
  d$center <- ifelse(d$partition == "center1_labeled", 1L, 2L)
  rownames(d) <- NULL
  d
}

# Center 1 out-of-fold U-Net rows with the covariates of the Table 2 models and the ground-truth instrument area
diag_read_or_frames <- function(derived) {
  man <- .read_columns(file.path(derived, "manifest.parquet"),
                       c("frame_id", "partition", "patient_id", "domain", "channel", "region", "instrument", "file_format"))
  man <- man[man$partition == "center1_labeled", ]
  fov <- .read_columns(file.path(derived, "fov.parquet"), c("frame_id", "fov_found"))
  man$fov_found <- fov$fov_found[match(man$frame_id, fov$frame_id)]
  if (anyNA(man$fov_found)) stop("fov.parquet lacks rows for some Center 1 frames")
  if (isTRUE(CFG$fov$exclude_not_found)) man <- man[man$fov_found, ]
  op <- .read_columns(file.path(derived, "optics.parquet"), c("frame_id", OPTICAL_METRICS))
  pr <- as.data.frame(.gl_collect(arrow::open_dataset(file.path(derived, "predictions.parquet")) |>
    dplyr::filter(partition == "center1_labeled", model == "unet") |>
    dplyr::select(frame_id, source, iou, pred_area_frac, gt_area_frac)))
  pr <- pr[pr$frame_id %in% man$frame_id, ]   # predictions.parquet normally lacks the excluded frames; rows for them, if present, are dropped
  if (!setequal(pr$frame_id, man$frame_id) || anyDuplicated(pr$frame_id) || any(pr$source != "oof")) {
    stop("Center 1 out-of-fold U-Net rows and the manifest disagree")
  }
  d <- cbind(man[c("frame_id", "patient_id", "domain", "channel", "region", "instrument", "file_format")],
             op[match(man$frame_id, op$frame_id), OPTICAL_METRICS],
             pr[match(man$frame_id, pr$frame_id), c("iou", "pred_area_frac", "gt_area_frac")])
  if (anyNA(d$gt_area_frac)) stop("ground-truth area missing for some Center 1 frames")
  rownames(d) <- NULL
  d
}

diag_reg <- function(key, x, digits = NULL, fmt = NULL, unit = NULL, note = NULL) {
  if (is.character(x) || all(is.finite(x))) reg(key, x, digits = digits, fmt = fmt, unit = unit, note = note)
  else reg(key, "not estimable", note = note)
}

diag_reg_ci <- function(key, est, lo, hi, digits, note = NULL) {
  if (all(is.finite(c(est, lo, hi)))) {
    reg(key, est, digits = digits, note = note)
    reg(paste0(key, "_ci"), c(est, lo, hi), digits = digits, fmt = "ci", note = note)
  } else if (is.finite(est)) reg(key, est, digits = digits, note = note)
}

# ---- 1 covariate shift ----------------------------------------------------------------------------------------

# Multivariate shift works on the metrics standardised with the Center 1 mean and SD; the gradient energy is logged first,
# as in the odds-ratio models (assoc_lib.R ASSOC_LOG_METRICS), because it spans orders of magnitude.
diag_std_matrix <- function(d, ref, mets = OPTICAL_METRICS) {
  tf <- function(x) {
    M <- as.matrix(x[mets])
    for (m in intersect(mets, ASSOC_LOG_METRICS)) M[, m] <- log(M[, m])
    M
  }
  R <- tf(ref)
  sweep(sweep(tf(d), 2, colMeans(R)), 2, apply(R, 2, stats::sd), "/")
}

# Out-of-fold linear predictor of an unpenalised logistic regression of Center 2 (1) vs Center 1 (0) on the standardised metrics;
# patients are never split across folds, and folds are balanced inside each center.
diag_domain_scores <- function(d, mets, K = DIAG_FOLDS) {
  Z <- diag_std_matrix(d, d[d$center == 1L, ], mets)
  y <- as.numeric(d$center == 2L)
  fold <- integer(nrow(d))
  for (cn in 1:2) fold[d$center == cn] <- patient_foldid(d$patient_id[d$center == cn], K)
  oof <- numeric(nrow(d))
  for (k in seq_len(K)) {
    tr <- fold != k
    fit <- suppressWarnings(stats::glm.fit(cbind(1, Z[tr, , drop = FALSE]), y[tr], family = stats::binomial(),
                                           control = stats::glm.control(maxit = 50)))
    b <- fit$coefficients
    b[is.na(b)] <- 0
    oof[!tr] <- drop(cbind(1, Z[!tr, , drop = FALSE]) %*% b)
  }
  oof
}

diag_covariate_shift <- function(d, B, seed, cores) {
  c1 <- d[d$center == 1L, ]
  c2 <- d[d$center == 2L, ]
  if (!nrow(c1) || !nrow(c2)) stop("covariate shift needs labeled frames from both centers")
  excursion <- function(x, bound, sd1, below) {
    out <- if (below) bound - x[x < bound] else x[x > bound] - bound
    if (length(out)) mean(out) / sd1 else NA_real_
  }
  per_metric <- do.call(rbind, lapply(OPTICAL_METRICS, function(m) {
    x1 <- c1[[m]]
    x2 <- c2[[m]]
    sd1 <- stats::sd(x1)
    q <- stats::quantile(x1, c(0.05, 0.95), names = FALSE)
    data.frame(scope = "metric", metric = m, n_c1 = length(x1), n_c2 = length(x2),
               median_c1 = stats::median(x1), median_c2 = stats::median(x2), sd_c1 = sd1,
               std_median_diff = (stats::median(x2) - stats::median(x1)) / sd1,
               p05_c1 = q[1], p95_c1 = q[2], min_c1 = min(x1), max_c1 = max(x1),
               share_c2_below_p05 = mean(x2 < q[1]), share_c2_above_p95 = mean(x2 > q[2]),
               share_c2_beyond_p05_p95 = mean(x2 < q[1] | x2 > q[2]),
               share_c2_below_min = mean(x2 < min(x1)), share_c2_above_max = mean(x2 > max(x1)),
               mean_excursion_below_min_sd = excursion(x2, min(x1), sd1, TRUE),
               mean_excursion_above_max_sd = excursion(x2, max(x1), sd1, FALSE),
               auroc_c2_vs_c1 = auroc_fail(c(x1, x2), rep(c(FALSE, TRUE), c(length(x1), length(x2)))),
               value = NA_real_, lo = NA_real_, hi = NA_real_, note = "", stringsAsFactors = FALSE)
  }))

  # domain classifier: all metrics, without the under-exposed fraction, without the three metrics that separate the centers most
  top3 <- per_metric$metric[order(-abs(per_metric$auroc_c2_vs_c1 - 0.5))][1:3]
  sets <- list(domain_auroc = OPTICAL_METRICS, domain_auroc_excl_under_frac = setdiff(OPTICAL_METRICS, "under_frac"),
               domain_auroc_excl_top3 = setdiff(OPTICAL_METRICS, top3))
  sc <- as.data.frame(lapply(sets, function(mets) diag_domain_scores(d, mets)))
  ev <- data.frame(patient_id = d$patient_id, center = d$center, sc)
  stat <- function(x) vapply(names(sets), function(s) auroc_fail(x[[s]], x$center == 2L), 0)
  boot <- cluster_boot(ev, "patient_id", "center", B, seed, stat, cores = cores)

  # Mahalanobis distance in Center 1 metric space: Center 1 frames from a mean and covariance estimated without their
  # fold (out-of-fold), Center 2 frames from the full Center 1 estimate
  Z1 <- diag_std_matrix(c1, c1)
  Z2 <- diag_std_matrix(c2, c1)
  maha <- function(X, ref) sqrt(stats::mahalanobis(X, colMeans(ref), stats::cov(ref)))
  d1 <- numeric(nrow(c1))
  for (k in sort(unique(c1$fold))) d1[c1$fold == k] <- maha(Z1[c1$fold == k, , drop = FALSE], Z1[c1$fold != k, , drop = FALSE])
  d2 <- maha(Z2, Z1)
  mv <- function(name, value, lo = NA_real_, hi = NA_real_, note = "") {
    data.frame(scope = "multivariate", metric = name, n_c1 = nrow(c1), n_c2 = nrow(c2), value = value, lo = lo, hi = hi,
               note = note, stringsAsFactors = FALSE)
  }
  ci <- boot$ci
  dnote <- c(domain_auroc = "all 11 metrics",
             domain_auroc_excl_under_frac = "without under_frac",
             domain_auroc_excl_top3 = paste("without", paste(top3, collapse = ", ")))
  multi <- rbind(
    do.call(rbind, lapply(names(sets), function(s) mv(s, ci$est[ci$stat == s], ci$lo[ci$stat == s], ci$hi[ci$stat == s], dnote[[s]]))),
    mv("mahalanobis_median_c1_oof", stats::median(d1), note = "Center 1 frames, mean and covariance estimated without their fold"),
    mv("mahalanobis_median_c2", stats::median(d2), note = "Center 2 frames, mean and covariance of all Center 1 frames"),
    mv("mahalanobis_ratio_c2_over_c1", stats::median(d2) / stats::median(d1), note = "Center 2 median over Center 1 out-of-fold median"),
    mv("mahalanobis_share_c2_above_c1_p95", mean(d2 > stats::quantile(d1, 0.95, names = FALSE)),
       note = "share of Center 2 frames beyond the 95th percentile of the Center 1 out-of-fold distance"))
  fill <- setdiff(names(per_metric), names(multi))
  multi[fill] <- NA
  tab <- rbind(per_metric, multi[names(per_metric)])
  rownames(tab) <- NULL
  list(table = tab, top3 = top3, mahalanobis = list(c1 = d1, c2 = d2))
}

register_shift <- function(res) {
  tb <- res$table
  pm <- tb[tb$scope == "metric", ]
  mv <- function(name) tb[tb$scope == "multivariate" & tb$metric == name, ]
  one <- function(m) pm[pm$metric == m, ]
  note <- "post hoc covariate-shift diagnostic, Center 2 relative to Center 1 (diag_covariate_shift.csv)"
  reg("diag_shift_n_c1", as.integer(pm$n_c1[1]), note = paste(note, "- Center 1 labeled analysed frames"))
  reg("diag_shift_n_c2", as.integer(pm$n_c2[1]), note = paste(note, "- Center 2 labeled analysed frames"))
  u <- one("under_frac")
  reg("diag_shift_under_median_c1", u$median_c1, fmt = "percent", digits = 3, note = paste(note, "- median under-exposed fraction, Center 1"))
  reg("diag_shift_under_median_c2", u$median_c2, fmt = "percent", digits = 3, note = paste(note, "- median under-exposed fraction, Center 2"))
  reg("diag_shift_under_below_min", u$share_c2_below_min, fmt = "percent", digits = 1,
      note = paste(note, "- share of Center 2 frames below the Center 1 minimum of the under-exposed fraction"))
  diag_reg("diag_shift_under_excursion_sd", u$mean_excursion_below_min_sd, digits = 3,
           note = paste(note, "- mean distance below the Center 1 minimum, in Center 1 SD, among the frames below it"))
  reg("diag_shift_under_std_diff", u$std_median_diff, digits = 2, note = paste(note, "- (Center 2 median minus Center 1 median) / Center 1 SD"))
  for (nm in c("lum", "haze")) {
    k <- c(lum = "lum_mean", haze = "haze_dcp")[[nm]]
    r <- one(k)
    reg(paste0("diag_shift_", nm, "_std_diff"), r$std_median_diff, digits = 2, note = paste(note, "- standardised median difference,", k))
    reg(paste0("diag_shift_", nm, "_above_p95"), r$share_c2_above_p95, fmt = "percent", digits = 1,
        note = paste(note, "- share of Center 2 frames above the Center 1 95th percentile,", k))
  }
  ord <- order(-abs(pm$std_median_diff))
  reg("diag_shift_max_abs_std_diff", abs(pm$std_median_diff[ord[1]]), digits = 2, note = paste(note, "- largest absolute standardised median difference over the 11 metrics"))
  reg("diag_shift_max_abs_std_diff_metric", METRIC_LABELS[[pm$metric[ord[1]]]], note = paste(note, "- the metric with the largest absolute standardised median difference"))
  reg("diag_shift_n_over_half_sd", sum(abs(pm$std_median_diff) >= 0.5), note = paste(note, "- metrics whose median moved by at least half a Center 1 SD"))
  others <- pm[pm$metric != "under_frac", ]
  reg("diag_shift_other_max_outside_range", max(others$share_c2_below_min + others$share_c2_above_max), fmt = "percent", digits = 2,
      note = paste(note, "- largest share of Center 2 frames outside the Center 1 range, over the 10 metrics other than the under-exposed fraction"))
  a <- mv("domain_auroc")
  diag_reg_ci("diag_shift_domain_auroc", a$value, a$lo, a$hi, 3, note = paste(note, "- patient-grouped out-of-fold AUROC of a logistic regression separating the centers on the 11 standardised metrics"))
  diag_reg("diag_shift_domain_auroc_excl_under", mv("domain_auroc_excl_under_frac")$value, digits = 3, note = paste(note, "- same without the under-exposed fraction"))
  diag_reg("diag_shift_domain_auroc_excl_top3", mv("domain_auroc_excl_top3")$value, digits = 3,
           note = paste(note, "- same without the three metrics that separate the centers most:", paste(res$top3, collapse = ", ")))
  diag_reg("diag_shift_maha_c1", mv("mahalanobis_median_c1_oof")$value, digits = 2, note = paste(note, "- median Mahalanobis distance, Center 1 out-of-fold frames"))
  diag_reg("diag_shift_maha_c2", mv("mahalanobis_median_c2")$value, digits = 2, note = paste(note, "- median Mahalanobis distance of Center 2 frames in Center 1 metric space"))
  diag_reg("diag_shift_maha_ratio", mv("mahalanobis_ratio_c2_over_c1")$value, digits = 2, note = paste(note, "- Center 2 median over Center 1 out-of-fold median"))
  diag_reg("diag_shift_maha_above_p95", mv("mahalanobis_share_c2_above_c1_p95")$value, fmt = "percent", digits = 1,
           note = paste(note, "- share of Center 2 frames beyond the 95th percentile of the Center 1 out-of-fold distance"))
}

# ---- 2 gate fit: non-zero penalised coefficients ---------------------------------------------------------------

.diag_metric_of <- function(term) sub("_[0-9]+$", "", term)

# non-zero coefficients (intercept excluded) of a glmnet fit at penalty s
.diag_nonzero <- function(fit, s) {
  b <- as.matrix(stats::coef(fit, s = s))[-1L, 1L]
  b[b != 0]
}

diag_nz_row <- function(scope, id, label, family, eval_set, n_train, events, n_candidate, nz_terms, lambda, score,
                        n_unpenalised = 0L, note = "") {
  nz_terms <- nz_terms[!names(nz_terms) %in% "domain_D2"]   # the unpenalised domain indicator is not a penalised coefficient
  mets <- if (length(nz_terms)) unique(.diag_metric_of(names(nz_terms))) else character()
  data.frame(scope = scope, id = id, label = label, model_family = family, eval_set = eval_set,
             n_train_frames = n_train, events_train = events, n_candidate_terms = n_candidate,
             n_nonzero_terms = length(nz_terms), n_nonzero_metrics = length(mets), nonzero_metrics = paste(mets, collapse = " "),
             n_unpenalised = n_unpenalised, lambda = lambda,
             events_per_nonzero = if (length(nz_terms)) events / length(nz_terms) else NA_real_,
             events_per_candidate = events / n_candidate,
             score_sd_eval = stats::sd(score), score_range_eval = diff(range(score)),
             constant_score = diff(range(score)) < DIAG_CONST_TOL,
             aurc_refit = NA_real_, aurc_sensitivity_csv = NA_real_, aurc_abs_diff = NA_real_, reproduced = NA,
             note = note, stringsAsFactors = FALSE)
}

# Wraps 08_sensitivity.R's fitters so that score_variant() (08's own code) hands back the closure of each fitted object.
diag_capturing_score_variant <- function() {
  cap <- new.env()
  wrap <- function(orig, nm) {
    force(orig)
    force(nm)
    function(...) {
      fn <- orig(...)
      cap[[nm]] <- environment(fn)
      fn
    }
  }
  env <- new.env(parent = globalenv())
  for (nm in c("fit_spline_gate", "fit_domain_gate", "fit_xgb_gate", "fit_gam_gate")) env[[nm]] <- wrap(get(nm, globalenv()), nm)
  sv <- score_variant
  environment(sv) <- env
  list(score_variant = sv, captured = cap)
}

diag_variant_aurc <- function(spec, sc) {
  risk <- 1 - sc$ev$iou
  if (identical(spec$aggregation, "patient_macro")) {
    mean(vapply(split(seq_along(risk), sc$ev$patient_id), function(i) aurc(sc$score[i], risk[i]), 0))
  } else aurc(sc$score, risk)
}

diag_gate_nonzero <- function(models, c1, c2, sens_csv = NULL) {
  g <- models$gate
  events_c1 <- sum(c1$fail)
  rows <- list()
  # final locked gate (scored on Center 2, the external evaluation set)
  fin_link <- predict_gate(g, c2, "link")
  rows$final <- diag_nz_row("final_gate", "final", "Locked gate (final, all Center 1)", "glmnet", "center2_labeled", nrow(c1), events_c1,
                            nrow(g$fit$beta), .diag_nonzero(g$fit, g$lambda), g$lambda, fin_link, note = DIAG_STATUS)
  # the five fold gates, each scored on its held-out Center 1 fold
  fo <- models$folds$optical_gate
  for (k in names(fo)) {
    fg <- fo[[k]]
    te <- c1$fold == as.integer(k)
    rows[[paste0("fold", k)]] <- diag_nz_row("fold_gate", paste0("fold", k), paste("Fold gate, held-out fold", k), "glmnet",
                                             paste0("center1_labeled fold ", k), sum(!te), sum(c1$fail[!te]), nrow(fg$fit$beta),
                                             .diag_nonzero(fg$fit, fg$lambda), fg$lambda, predict_gate(fg, c1[te, ], "link"), note = DIAG_STATUS)
  }
  # the 11 variants of 08_sensitivity.R, re-fitted by 08's own score_variant
  cs <- diag_capturing_score_variant()
  for (v in SENS_VARIANTS$variant) {
    spec <- as.list(SENS_VARIANTS[SENS_VARIANTS$variant == v, ])
    rm(list = ls(cs$captured), envir = cs$captured)
    sc <- cs$score_variant(spec, c1, c2, models)
    eval_set <- paste0(spec$eval_partition, " ", spec$eval_domain)
    cap <- cs$captured
    row <- if (!is.null(cap$fit_spline_gate)) {
      e <- cap$fit_spline_gate
      diag_nz_row("variant", v, spec$label, "glmnet", eval_set, nrow(e$tr), sum(e$fail), nrow(e$g$fit$beta),
                  .diag_nonzero(e$g$fit, e$g$lambda), e$g$lambda, sc$score, note = DIAG_STATUS)
    } else if (!is.null(cap$fit_domain_gate)) {
      e <- cap$fit_domain_gate
      diag_nz_row("variant", v, spec$label, "glmnet", eval_set, nrow(e$tr), sum(e$fail), ncol(e$x) - 1L,
                  .diag_nonzero(e$cv$glmnet.fit, e$cv$lambda.1se), e$cv$lambda.1se, sc$score, n_unpenalised = 1L,
                  note = paste(DIAG_STATUS, "; unpenalised domain indicator not counted"))
    } else if (!is.null(cap$fit_xgb_gate)) {
      e <- cap$fit_xgb_gate
      r <- diag_nz_row("variant", v, spec$label, "xgboost", eval_set, nrow(e$tr), sum(e$fail), length(e$features), character(), NA_real_,
                       sc$score, note = paste0(DIAG_STATUS, "; boosting rounds ", e$cv$early_stop$best_iteration))
      r$n_nonzero_terms <- r$n_nonzero_metrics <- NA_integer_
      r$events_per_nonzero <- NA_real_
      r
    } else if (!is.null(cap$fit_gam_gate)) {
      e <- cap$fit_gam_gate
      edf <- summary(e$fit)$s.table[, "edf"]
      r <- diag_nz_row("variant", v, spec$label, "gam", eval_set, nrow(e$tr), sum(e$fail), length(edf), character(), NA_real_, sc$score,
                       note = sprintf("%s; smooths with effective df of at least %s counted as non-zero; total effective df %.2f",
                                      DIAG_STATUS, DIAG_GAM_EDF_MIN, sum(edf)))
      keep <- unname(edf) >= DIAG_GAM_EDF_MIN
      r$n_nonzero_terms <- sum(keep)
      r$n_nonzero_metrics <- sum(keep)
      r$nonzero_metrics <- paste(sub("^s\\((.*)\\)$", "\\1", names(edf))[keep], collapse = " ")
      r$events_per_nonzero <- if (any(keep)) r$events_train / sum(keep) else NA_real_
      r
    } else {
      # the locked gate itself (locked, patient_macro): no fitting, the stored object
      diag_nz_row("variant", v, spec$label, "glmnet", eval_set, nrow(c1), events_c1, nrow(g$fit$beta), .diag_nonzero(g$fit, g$lambda), g$lambda,
                  sc$score, note = paste(DIAG_STATUS, "; the locked gate, not re-fitted"))
    }
    a <- diag_variant_aurc(spec, sc)
    row$aurc_refit <- a
    if (!is.null(sens_csv)) {
      ref <- sens_csv$aurc[sens_csv$variant == v & sens_csv$method == "gate"]
      if (length(ref) == 1L) {
        row$aurc_sensitivity_csv <- ref
        row$aurc_abs_diff <- abs(a - ref)
        row$reproduced <- abs(a - ref) <= DIAG_REPRO_TOL
      }
    }
    rows[[paste0("variant_", v)]] <- row
  }
  tab <- do.call(rbind, rows)
  rownames(tab) <- NULL
  req <- tab[tab$scope == "variant" & tab$id %in% DIAG_REPRO_REQUIRED, ]
  if (!is.null(sens_csv) && any(!req$reproduced %in% TRUE)) {
    stop("re-fitted variant(s) do not reproduce sensitivity.csv to ", DIAG_REPRO_TOL, ": ", paste(req$id[!req$reproduced %in% TRUE], collapse = ", "))
  }
  tab
}

register_nz <- function(tab) {
  note <- "post hoc gate-fit diagnostic (diag_gate_nonzero.csv)"
  f <- tab[tab$scope == "final_gate", ]
  fo <- tab[tab$scope == "fold_gate", ]
  va <- tab[tab$scope == "variant", ]
  reg("diag_nz_final_terms", as.integer(f$n_nonzero_terms), note = paste(note, "- non-zero penalised coefficients in the locked gate"))
  reg("diag_nz_final_metrics", as.integer(f$n_nonzero_metrics), note = paste(note, "- distinct metrics behind them"))
  reg("diag_nz_final_candidate", as.integer(f$n_candidate_terms), note = paste(note, "- candidate spline terms offered to the penalised fit"))
  reg("diag_nz_final_lambda", f$lambda, digits = 4, note = paste(note, "- penalty of the locked gate (1-SE rule)"))
  reg("diag_nz_final_events", as.integer(f$events_train), note = paste(note, "- Center 1 failures the locked gate was fitted on"))
  diag_reg("diag_nz_final_epc", f$events_per_nonzero, digits = 0, note = paste(note, "- failures per non-zero coefficient"))
  reg("diag_nz_final_epc_candidate", f$events_per_candidate, digits = 0, note = paste(note, "- failures per candidate term"))
  reg("diag_nz_fold_terms_min", as.integer(min(fo$n_nonzero_terms)), note = paste(note, "- fewest non-zero coefficients in a fold gate"))
  reg("diag_nz_fold_terms_max", as.integer(max(fo$n_nonzero_terms)), note = paste(note, "- most non-zero coefficients in a fold gate"))
  reg("diag_nz_fold_terms_median", stats::median(fo$n_nonzero_terms), digits = if (stats::median(fo$n_nonzero_terms) %% 1 == 0) 0 else 1, note = paste(note, "- median over the fold gates"))
  reg("diag_nz_fold_n_constant", as.integer(sum(fo$constant_score)), note = paste(note, "- fold gates whose held-out scores are all equal"))
  const <- va[va$constant_score %in% TRUE, ]
  reg("diag_nz_n_variants", as.integer(nrow(va)), note = paste(note, "- sensitivity variants re-fitted"))
  reg("diag_nz_n_const_variants", as.integer(nrow(const)), note = paste(note, "- variants whose evaluation scores are all equal (a constant score)"))
  # the exhibits name the centers by role (00_setup.R CENTER_ROLE); the 08 labels say Center 1 / Center 2
  us <- function(x) gsub("ormalised", "ormalized", gsub("standardised", "standardized", x, fixed = TRUE), fixed = TRUE)   # American spelling for the 08 labels
  role <- function(x) us(if (exists("CENTER_ROLE")) gsub("Center 2", tolower(CENTER_ROLE[["2"]]), gsub("Center 1", tolower(CENTER_ROLE[["1"]]), x, fixed = TRUE), fixed = TRUE) else x)
  reg("diag_nz_const_variants", if (nrow(const)) paste(role(const$label), collapse = "; ") else "none", note = paste(note, "- labels of those variants"))
  ext <- va[grepl("^center2", va$eval_set), ]
  reg("diag_nz_n_const_ext", as.integer(sum(ext$constant_score %in% TRUE)), note = paste(note, "- of those, variants scored on Center 2"))
  reg("diag_nz_n_reproduced", as.integer(sum(va$reproduced %in% TRUE)),
      note = paste(note, "- variants whose re-fitted AURC equals sensitivity.csv to", DIAG_REPRO_TOL))
}

# ---- 3 flag shares of the final gate ---------------------------------------------------------------------------

diag_share_boot <- function(link, thr, patient, strata, B, cores) {
  r <- assoc_boot_ci(function(rows) {
    s <- link[rows]
    s90 <- mean(s > thr[[1]])
    s80 <- mean(s > thr[[2]])
    c(share_90 = s90, share_80 = s80, band = s80 - s90)
  }, patient, strata, B = B, cores = cores)
  r
}

diag_flag_shares <- function(models, c1, c2, st, B, cores) {
  gate <- models$gate
  cov <- unlist(CFG$endpoints$coverage_levels)
  code <- as.character(round(cov * 100))
  thr <- gate$flag_thresholds[as.character(cov)]
  if (anyNA(thr) || length(thr) != 2L) stop("the locked gate lacks flag thresholds for both coverages")
  links <- list(center1_oof = list(s = predict_oof(models$folds$optical_gate, c1, c1$fold), pat = c1$patient_id, strata = c1$domain),
                center1_final = list(s = predict_gate(gate, c1, "link"), pat = c1$patient_id, strata = c1$domain),
                center2_final = list(s = predict_gate(gate, c2, "link"), pat = c2$patient_id, strata = NULL),
                stream_final = list(s = predict_gate(gate, st, "link"), pat = st$video_patient, strata = NULL))
  rows <- list()
  add <- function(scope, statistic, coverage, est, lo = NA_real_, hi = NA_real_, n = NA_real_) {
    rows[[length(rows) + 1L]] <<- data.frame(scope = scope, statistic = statistic, coverage = coverage, est = est, lo = lo, hi = hi,
                                             n = n, stringsAsFactors = FALSE)
  }
  for (j in 1:2) add("locked_cutoffs", "cutoff_logodds", as.numeric(code[j]), unname(thr[[j]]))
  add("locked_cutoffs", "band_width_logodds", NA_real_, unname(thr[[1]] - thr[[2]]))
  for (sc in names(links)) {
    L <- links[[sc]]
    b <- diag_share_boot(L$s, thr, L$pat, L$strata, B, cores)
    n <- length(L$s)
    add(sc, "flagged_share", as.numeric(code[1]), b$est[1], b$lo[1], b$hi[1], n)
    add(sc, "flagged_share", as.numeric(code[2]), b$est[2], b$lo[2], b$hi[2], n)
    add(sc, "band_share", NA_real_, b$est[3], b$lo[3], b$hi[3], n)
    q <- stats::quantile(L$s, c(0.05, 0.25, 0.5, 0.75, 0.95), names = FALSE)
    add(sc, "score_mean", NA_real_, mean(L$s), n = n)
    add(sc, "score_median", NA_real_, q[3], n = n)
    add(sc, "score_sd", NA_real_, stats::sd(L$s), n = n)
    add(sc, "score_iqr", NA_real_, q[4] - q[2], n = n)
    add(sc, "score_p05", NA_real_, q[1], n = n)
    add(sc, "score_p95", NA_real_, q[5], n = n)
    add(sc, "band_width_over_score_sd", NA_real_, (thr[[1]] - thr[[2]]) / stats::sd(L$s), n = n)
    add(sc, "n_distinct_scores", NA_real_, length(unique(L$s)), n = n)
  }
  oof <- links$center1_oof$s
  for (k in sort(unique(c1$fold))) {
    i <- c1$fold == k
    sc <- paste0("center1_oof_fold", k)
    add(sc, "score_mean", NA_real_, mean(oof[i]), n = sum(i))
    add(sc, "score_sd", NA_real_, stats::sd(oof[i]), n = sum(i))
    add(sc, "n_distinct_scores", NA_real_, length(unique(oof[i])), n = sum(i))
  }
  tab <- do.call(rbind, rows)
  rownames(tab) <- NULL
  list(table = tab, thresholds = thr, code = code, links = lapply(links, `[[`, "s"))
}

FLAG_SCOPES <- c(c1_oof = "center1_oof", c1_final = "center1_final", c2 = "center2_final", stream = "stream_final")

register_flags <- function(res) {
  tb <- res$table
  g <- function(scope, statistic, coverage = NA_real_) {
    r <- tb[tb$scope == scope & tb$statistic == statistic & (if (is.na(coverage)) is.na(tb$coverage) else tb$coverage %in% coverage), ]
    if (nrow(r) != 1L) stop("diag_flag_shares lacks ", scope, " / ", statistic)
    r
  }
  note <- "post hoc flag-share diagnostic (diag_flag_shares.csv); final gate at the locked Center 1 out-of-fold cutoffs"
  for (nm in names(FLAG_SCOPES)) {
    sc <- FLAG_SCOPES[[nm]]
    for (j in 1:2) {
      reg(paste0("diag_flag_", nm, "_share_", res$code[j]), g(sc, "flagged_share", as.numeric(res$code[j]))$est, fmt = "percent", digits = 1,
          note = paste0(note, " - ", sc, " - ", res$code[j], "% coverage cutoff"))
    }
    reg(paste0("diag_flag_band_share_", nm), g(sc, "band_share")$est, fmt = "percent", digits = 1,
        note = paste(note, "-", sc, "- share of frames between the two cutoffs"))
  }
  for (j in 1:2) {
    r <- g("center1_final", "flagged_share", as.numeric(res$code[j]))
    reg(paste0("diag_flag_c1_final_share_", res$code[j], "_lo"), r$lo, fmt = "percent", digits = 1, note = paste(note, "- patient-clustered 95% lower limit"))
    reg(paste0("diag_flag_c1_final_share_", res$code[j], "_hi"), r$hi, fmt = "percent", digits = 1, note = paste(note, "- patient-clustered 95% upper limit"))
  }
  reg("diag_flag_sd_oof", g("center1_oof", "score_sd")$est, digits = 3, note = paste(note, "- SD of the Center 1 out-of-fold gate log-odds (pooled fold gates)"))
  reg("diag_flag_sd_final", g("center1_final", "score_sd")$est, digits = 3, note = paste(note, "- SD of the final gate log-odds on Center 1 (in-sample)"))
  reg("diag_flag_sd_ratio", g("center1_final", "score_sd")$est / g("center1_oof", "score_sd")$est, digits = 2, note = paste(note, "- final over out-of-fold SD"))
  reg("diag_flag_iqr_oof", g("center1_oof", "score_iqr")$est, digits = 3, note = paste(note, "- IQR of the out-of-fold log-odds"))
  reg("diag_flag_iqr_final", g("center1_final", "score_iqr")$est, digits = 3, note = paste(note, "- IQR of the final-gate log-odds on Center 1"))
  reg("diag_flag_sd_final_c2", g("center2_final", "score_sd")$est, digits = 3, note = paste(note, "- SD of the final gate log-odds on Center 2"))
  reg("diag_flag_band_width", g("locked_cutoffs", "band_width_logodds")$est, digits = 3,
      note = paste(note, "- distance in log-odds between the two cutoffs"))
  reg("diag_flag_band_over_sd", g("center1_final", "band_width_over_score_sd")$est, digits = 2,
      note = paste(note, "- band width over the final-gate SD on Center 1"))
}

# ---- 4 within-domain internal performance ----------------------------------------------------------------------

diag_within_domain <- function(models, c1, choice, thr, B, cores) {
  sc <- score_methods(models, c1, choice, oof = TRUE)
  out <- lapply(sort(unique(c1$domain)), function(dm) {
    i <- c1$domain == dm
    ev <- evaluate_scored(c1[i, ], sc[i, ], paste0("internal_", dm), "unet", thr, B, cores = cores)
    tb <- ev$table
    dr <- paired_diff_ci(ev$boot, paste0(tb$method, "|aurc"), rep("random|aurc", nrow(tb)))
    data.frame(domain = dm, method = tb$method, n_frames = tb$n_frames, n_patients = tb$n_patients, n_events = tb$n_events,
               n_patients_with_events = tb$n_patients_with_events, failure_rate = tb$n_events / tb$n_frames,
               aurc = tb$aurc, aurc_lo = tb$aurc_lo, aurc_hi = tb$aurc_hi,
               auroc = tb$auroc, auroc_lo = tb$auroc_lo, auroc_hi = tb$auroc_hi,
               sel_risk_80 = tb$sel_risk_80, sel_risk_80_lo = tb$sel_risk_80_lo, sel_risk_80_hi = tb$sel_risk_80_hi,
               diff_aurc_vs_random = dr$diff, diff_aurc_vs_random_lo = dr$lo, diff_aurc_vs_random_hi = dr$hi,
               stringsAsFactors = FALSE)
  })
  tab <- do.call(rbind, out)
  rownames(tab) <- NULL
  tab
}

register_dom <- function(tab) {
  note <- "post hoc within-domain diagnostic, Center 1 out-of-fold (diag_within_domain.csv)"
  short <- c(optical_gate = "gate", entropy = "entropy", random = "random")
  for (dm in unique(tab$domain)) {
    t <- tab[tab$domain == dm, ]
    pre <- paste0("diag_dom_", tolower(dm), "_")
    reg(paste0(pre, "n_frames"), as.integer(t$n_frames[1]), note = paste(note, "-", dm))
    reg(paste0(pre, "n_patients"), as.integer(t$n_patients[1]), note = paste(note, "-", dm))
    reg(paste0(pre, "n_events"), as.integer(t$n_events[1]), note = paste(note, "-", dm))
    for (m in names(short)) {
      r <- t[t$method == m, ]
      .gl_reg_est_ci(paste0(pre, "aurc_", short[[m]]), r$aurc, r$aurc_lo, r$aurc_hi, 3, note = paste(note, "-", dm, "- AURC"))
      if (m != "random") .gl_reg_est_ci(paste0(pre, "auroc_", short[[m]]), r$auroc, r$auroc_lo, r$auroc_hi, 2, note = paste(note, "-", dm, "- AUROC for failure"))
    }
    r <- t[t$method == "optical_gate", ]
    .gl_reg_est_ci(paste0(pre, "diff_aurc_gate_random"), r$diff_aurc_vs_random, r$diff_aurc_vs_random_lo, r$diff_aurc_vs_random_hi, 3,
                   note = paste(note, "-", dm, "- gate minus random AURC, negative favours the gate"))
  }
}

# ---- 5 odds-ratio robustness -------------------------------------------------------------------------------------

diag_or_robustness <- function(fr, thr, B, cores, or_csv = NULL) {
  MET <- OPTICAL_METRICS
  y <- as.integer(fr$iou < thr)
  Z <- assoc_metric_matrix(fr, MET)
  cov_p <- assoc_covariates(fr, area = "pred_area_frac")
  cov_g <- assoc_covariates(fr, area = "gt_area_frac")
  fit <- function(m, cov) {
    f <- suppressWarnings(assoc_fit_model(y, cbind(Z[, m, drop = FALSE], cov$X), fr$patient_id, m, fr$domain, label = m))
    f$table[f$table$term == m, ]
  }
  wald_p <- lapply(MET, fit, cov = cov_p)
  wald_g <- lapply(MET, fit, cov = cov_g)
  # patient-clustered, domain-stratified percentile bootstrap of the primary specification; one set of resamples for all metrics
  Xs <- lapply(setNames(MET, MET), function(m) cbind(1, Z[, m], cov_p$X))
  est_fn <- function(rows) {
    vapply(MET, function(m) {
      f <- suppressWarnings(stats::glm.fit(Xs[[m]][rows, , drop = FALSE], y[rows], family = stats::binomial(),
                                           control = stats::glm.control(maxit = 50)))
      exp(f$coefficients[2])
    }, 0)
  }
  bt <- assoc_boot_ci(est_fn, fr$patient_id, fr$domain, B = B, cores = cores)
  pick <- function(l, col) vapply(l, function(t) t[[col]], 0)
  tab <- data.frame(metric = MET, label = unname(METRIC_LABELS[MET]),
                    or_scale = ifelse(MET %in% ASSOC_LOG_METRICS, "per SD of log value", "per SD"),
                    n = nrow(fr), events = sum(y), patients = length(unique(fr$patient_id)),
                    or_primary = pick(wald_p, "or"), or_lo_wald = pick(wald_p, "or_lo"), or_hi_wald = pick(wald_p, "or_hi"),
                    or_lo_boot = bt$lo, or_hi_boot = bt$hi, boot_valid = bt$n_valid,
                    or_gtarea = pick(wald_g, "or"), or_lo_gtarea = pick(wald_g, "or_lo"), or_hi_gtarea = pick(wald_g, "or_hi"),
                    stringsAsFactors = FALSE)
  tab$excl1_wald <- tab$or_lo_wald > 1 | tab$or_hi_wald < 1
  tab$excl1_boot <- tab$or_lo_boot > 1 | tab$or_hi_boot < 1
  tab$excl1_gtarea <- tab$or_lo_gtarea > 1 | tab$or_hi_gtarea < 1
  tab$ci_width_ratio_boot_over_wald <- (log(tab$or_hi_boot) - log(tab$or_lo_boot)) / (log(tab$or_hi_wald) - log(tab$or_lo_wald))
  if (!is.null(or_csv)) {
    prev <- or_csv[or_csv$scope == "center1_oof" & or_csv$model == "univariable_adjusted", ]
    prev <- prev[match(MET, prev$metric), ]
    if (anyNA(prev$or_sd) || max(abs(prev$or_sd - tab$or_primary)) > 1e-8 || max(abs(prev$or_lo - tab$or_lo_wald)) > 1e-8) {
      stop("the primary odds ratios re-fitted here differ from assoc_or_models.csv")
    }
  }
  rownames(tab) <- NULL
  tab
}

register_or <- function(tab) {
  note <- "post hoc odds-ratio robustness (diag_or_robustness.csv), Center 1 out-of-fold, one metric at a time"
  reg("diag_or_n_excl1_wald", as.integer(sum(tab$excl1_wald)), note = paste(note, "- metrics whose cluster-robust Wald 95% CI excludes one (primary specification)"))
  reg("diag_or_n_excl1_boot", as.integer(sum(tab$excl1_boot)), note = paste(note, "- metrics whose patient-clustered bootstrap percentile 95% CI excludes one"))
  reg("diag_or_n_excl1_gtarea", as.integer(sum(tab$excl1_gtarea)), note = paste(note, "- metrics whose Wald 95% CI excludes one with the ground-truth area covariate"))
  for (m in c("sharp_tenengrad", "contrast_local", "red_dom")) {
    r <- tab[tab$metric == m, ]
    assoc_reg_ci(paste0("diag_or_", m, "_gtarea_ci"), r$or_gtarea, r$or_lo_gtarea, r$or_hi_gtarea, 2,
                 note = paste(note, "- adjusted OR with the ground-truth instrument area instead of the predicted area,", r$or_scale))
  }
}

# ---- 6 calibration offsets and spread ----------------------------------------------------------------------------

diag_calibration <- function(models, c1, c2, B, cores) {
  gate <- models$gate
  sets <- list(center1_oof = list(d = c1, link = predict_oof(models$folds$optical_gate, c1, c1$fold), strata = c1$domain),
               center1_final_in_sample = list(d = c1, link = predict_gate(gate, c1, "link"), strata = c1$domain),
               center2_final = list(d = c2, link = predict_gate(gate, c2, "link"), strata = NULL))
  rows <- lapply(names(sets), function(sc) {
    S <- sets[[sc]]
    pr <- stats::plogis(S$link)
    y <- as.numeric(S$d$fail)
    b <- assoc_boot_ci(function(rows) c(observed = mean(y[rows]), predicted = mean(pr[rows]), offset = mean(pr[rows]) - mean(y[rows])),
                       S$d$patient_id, S$strata, B = B, cores = cores)
    q <- stats::quantile(pr, c(0.05, 0.25, 0.75, 0.95), names = FALSE)
    data.frame(scope = sc, n_frames = nrow(S$d), n_patients = length(unique(S$d$patient_id)), n_events = sum(y),
               observed_rate = b$est[1], observed_lo = b$lo[1], observed_hi = b$hi[1],
               mean_predicted = b$est[2], mean_predicted_lo = b$lo[2], mean_predicted_hi = b$hi[2],
               offset = b$est[3], offset_lo = b$lo[3], offset_hi = b$hi[3],
               sd_predicted = stats::sd(pr), iqr_predicted = q[3] - q[2], p05_predicted = q[1], p95_predicted = q[4],
               stringsAsFactors = FALSE)
  })
  tab <- do.call(rbind, rows)
  rownames(tab) <- NULL
  tab
}

register_calib <- function(tab) {
  note <- "post hoc calibration diagnostic (diag_calibration.csv), final gate"
  pp <- function(v, d) paste0(fmt_pct(v[1], d), " (95% CI ", fmt_pct(v[2], d), " to ", fmt_pct(v[3], d), ")")
  for (sc in c(c1 = "center1_oof", c2 = "center2_final")) {
    r <- tab[tab$scope == sc, ]
    nm <- names(which(c(c1 = "center1_oof", c2 = "center2_final") == sc))
    lab <- if (sc == "center1_oof") "Center 1, out-of-fold links" else "Center 2"
    reg(paste0("diag_calib_", nm, "_mean_pred"), r$mean_predicted, fmt = "percent", digits = 1, note = paste(note, "- mean predicted failure probability,", lab))
    reg(paste0("diag_calib_", nm, "_observed"), r$observed_rate, fmt = "percent", digits = 1, note = paste(note, "- observed failure rate,", lab))
    if (all(is.finite(c(r$offset, r$offset_lo, r$offset_hi)))) {
      reg(paste0("diag_calib_", nm, "_offset_ci"), c(r$offset, r$offset_lo, r$offset_hi), digits = 1, fmt = pp,
          note = paste(note, "- mean predicted minus observed failure rate, percentage points of frames,", lab))
    }
    reg(paste0("diag_calib_", nm, "_offset"), r$offset, fmt = "percent", digits = 1, note = paste(note, "- mean predicted minus observed,", lab))
    reg(paste0("diag_calib_", nm, "_sd_pred"), r$sd_predicted, fmt = "percent", digits = 1, note = paste(note, "- SD of the predicted probability,", lab))
  }
}

# ---- driver ------------------------------------------------------------------------------------------------------

run_diagnostics <- function(paths = default_paths(), B = CFG$statistics$bootstrap_resamples, cores = boot_cores()) {
  lock <- read_lock(paths$lock)
  verify_lock_hashes(lock)                      # before any Center 2 row is read
  choice <- lock$entropy_baseline_choice
  if (!.gl_nonempty(choice)) stop("lock file has no entropy_baseline_choice", call. = FALSE)
  models <- load_gate_models(lock)
  thr <- lock_failure_threshold(lock)
  seed <- CFG$seeds$bootstrap
  csv <- function(f) {
    path <- file.path(paths$stats, f)
    if (file.exists(path)) utils::read.csv(path, stringsAsFactors = FALSE) else NULL
  }
  out <- list()

  t0 <- Sys.time()
  tick <- function(what) {
    message(sprintf("34: %s (%.0f s)", what, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    t0 <<- Sys.time()
  }

  out$shift <- diag_covariate_shift(diag_read_optics(paths$derived), B, seed, cores)
  write_csv_guarded(out$shift$table, file.path(paths$stats, "diag_covariate_shift.csv"))
  tick("covariate shift")

  c1 <- read_sens_frames(paths$derived, "center1_labeled")
  c2 <- read_sens_frames(paths$derived, "center2_labeled")
  c1$fail <- c1$iou < thr
  c2$fail <- c2$iou < thr

  out$nz <- diag_gate_nonzero(models, c1, c2, csv("sensitivity.csv"))
  write_csv_guarded(out$nz, file.path(paths$stats, "diag_gate_nonzero.csv"))
  tick("gate non-zero coefficients and variant re-fits")

  st <- read_stream_frames(paths$derived)
  out$flags <- diag_flag_shares(models, c1, c2, st, B, cores)
  write_csv_guarded(out$flags$table, file.path(paths$stats, "diag_flag_shares.csv"))
  tick("flag shares")

  out$dom <- diag_within_domain(models, c1, choice, thr, B, cores)
  write_csv_guarded(out$dom, file.path(paths$stats, "diag_within_domain.csv"))
  tick("within-domain performance")

  out$or <- diag_or_robustness(diag_read_or_frames(paths$derived), thr, B, cores, csv("assoc_or_models.csv"))
  write_csv_guarded(out$or, file.path(paths$stats, "diag_or_robustness.csv"))
  tick("odds-ratio robustness")

  out$calib <- diag_calibration(models, c1, c2, B, cores)
  write_csv_guarded(out$calib, file.path(paths$stats, "diag_calibration.csv"))
  tick("calibration")

  out$threshold <- thr
  out$lock <- lock
  out
}

register_34 <- function(res) {
  register_shift(res$shift)
  register_nz(res$nz)
  register_flags(res$flags)
  register_dom(res$dom)
  register_or(res$or)
  register_calib(res$calib)
}

if (sys.nframe() == 0L) {
  paths <- default_paths()
  if (SYNTH && !.gl_nonempty(tryCatch(read_lock(paths$lock)$lock_fields_sha256, error = function(e) NULL))) {
    message("34 diagnostics skipped: the synthetic lock carries no lock_fields_sha256 (run 04a on synthetic data with a sealed lock first)")
    quit(status = 0)
  }
  t0 <- Sys.time()
  B <- CFG$statistics$bootstrap_resamples
  res <- run_diagnostics(paths, B)
  register_34(res)
  guard_write(file.path(paths$results, "results", "parts", "34_diagnostics.json"))
  write_results_part("34_diagnostics", root = paths$results)
  sh <- res$shift$table
  cat("34 diagnostics (post hoc, descriptive)\n")
  cat("1 covariate shift: per-metric standardised median difference, share of Center 2 beyond the Center 1 5th/95th percentile\n")
  print(sh[sh$scope == "metric", c("metric", "std_median_diff", "share_c2_below_p05", "share_c2_above_p95", "share_c2_below_min", "share_c2_above_max")],
        digits = 3, row.names = FALSE)
  print(sh[sh$scope == "multivariate", c("metric", "value", "lo", "hi")], digits = 3, row.names = FALSE)
  cat("2 gate fit: non-zero penalised coefficients\n")
  print(res$nz[, c("scope", "id", "n_candidate_terms", "n_nonzero_terms", "n_nonzero_metrics", "constant_score", "aurc_refit", "aurc_abs_diff")],
        digits = 4, row.names = FALSE)
  cat("3 flag shares of the final gate\n")
  fl <- res$flags$table
  print(fl[fl$statistic %in% c("flagged_share", "band_share", "cutoff_logodds", "band_width_logodds", "score_sd", "score_iqr") & !grepl("fold", fl$scope),
           c("scope", "statistic", "coverage", "est", "lo", "hi")], digits = 3, row.names = FALSE)
  cat("4 within-domain internal performance\n")
  print(res$dom[res$dom$method %in% c("random", "entropy", "optical_gate"),
                c("domain", "method", "n_frames", "n_events", "aurc", "aurc_lo", "aurc_hi", "auroc", "diff_aurc_vs_random", "diff_aurc_vs_random_lo", "diff_aurc_vs_random_hi")],
        digits = 3, row.names = FALSE)
  cat("5 odds-ratio robustness\n")
  print(res$or[, c("metric", "or_primary", "or_lo_wald", "or_hi_wald", "or_lo_boot", "or_hi_boot", "or_gtarea", "excl1_wald", "excl1_boot", "excl1_gtarea")],
        digits = 3, row.names = FALSE)
  cat("6 calibration\n")
  print(res$calib[, c("scope", "observed_rate", "mean_predicted", "offset", "offset_lo", "offset_hi", "sd_predicted")], digits = 3, row.names = FALSE)
  cat(sprintf("(%.1f s, B = %d)\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), B))
}
