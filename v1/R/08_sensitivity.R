# 08: exploratory sensitivity analyses of the external-center comparison (gate vs random, entropy, ensemble
# disagreement; risk = 1 - U-Net IoU; AURC, lower is better). Every variant's gate is fitted on Center 1 data only
# (device_norm also takes its D3 normaliser from Center 2 optics, no outcomes), is NOT part of the locked primary, and
# is labelled exploratory (no multiplicity claims). The lock is verified (lock_fields_sha256 and the RDS hashes) before
# any outcome is read. XGBoost rounds and hyperparameters come from sensitivity.xgboost in analysis_plan.yaml.
#   Rscript R/08_sensitivity.R            (OPTICS_DATA=synthetic for development)
# Variants (id: what changes)
#   locked          the locked gate, re-evaluated here as the reference row
#   xgboost         boosted trees on the 11 raw metrics (rounds from patient-grouped CV on Center 1)
#   gam             mgcv GAM, one shrinkage smooth per metric (k = spline df + 1)
#   domain_cov      the locked gate recipe plus an unpenalised D2 indicator; Center 2 is scored at the reference
#                   level, a constant for every Center 2 frame, so the ranking inside Center 2 is unaffected
#   background_only the recipe on unet_bg_* (metrics on the complement of the predicted mask)
#   device_norm     each metric z-scored within domain (Center 1 statistics for D1 and D2, Center 2 statistics for
#                   D3; optics only, no outcome enters the statistics)
#   patient_macro   the locked gate scores aggregated as the mean of per-patient AURC
#   lodo_*          leave-one-domain-out: Center 1 has two domains, so the gate is fitted on one and scored on the
#                   other (lodo_d1_to_d2, lodo_d2_to_d1) and on Center 2 (lodo_d1_to_d3, lodo_d2_to_d3); D3 is never used in
#                   fitting. The segmenter outcomes in D1 and D2 are still the cross-fitted ones (folds mix both domains), so
#                   this tests the gate's transport between domains, not the segmenter's.
# Outputs results/stats/sensitivity.csv (one row per variant x method; n_excluded counts evaluation frames dropped for
#         non-finite background-only metrics), results/parts/08_sensitivity.json

if (!exists("run_locked_evaluation", mode = "function") || !exists("assoc_boot_ci", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R", "assoc_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

SENS_METHODS <- c("random", "entropy", "ens_disagreement", "gate")
SENS_FOLDS <- CFG$segmenters$cv_folds       # patient-grouped folds for the inner CV, as in 04a
SENS_GAM_MIN_UNIQUE <- 10L                  # a metric with fewer distinct values enters the GAM linearly
SENS_XGB <- CFG$sensitivity$xgboost         # rounds, early-stopping patience and booster parameters (analysis_plan.yaml)
if (is.null(SENS_XGB$rounds) || is.null(SENS_XGB$patience) || is.null(SENS_XGB$params)) {
  stop("analysis_plan.yaml: sensitivity.xgboost needs rounds, patience and params")
}
SENS_XGB_ROUNDS <- as.integer(SENS_XGB$rounds)
SENS_XGB_PATIENCE <- as.integer(SENS_XGB$patience)
SENS_STATUS <- "exploratory; fitted on Center 1 only; not part of the locked primary"
# the D3 normaliser takes its mean and SD from Center 2 optics (no outcomes), so "Center 1 only" would overstate it
SENS_STATUS_DEVICE_NORM <- "exploratory; gate fitted on Center 1 only; the D3 normaliser uses Center 2 optics with no outcomes; not part of the locked primary"

SENS_VARIANTS <- data.frame(
  variant = c("locked", "xgboost", "gam", "domain_cov", "background_only", "device_norm", "patient_macro",
              "lodo_d1_to_d3", "lodo_d2_to_d3", "lodo_d1_to_d2", "lodo_d2_to_d1"),
  label = c("Locked gate (reference)", "XGBoost gate", "GAM gate", "Gate + domain covariate",
            "Background-only features", "Device-normalised features", "Patient-macro aggregation",
            "Fit on D1, score Center 2", "Fit on D2, score Center 2", "Fit on D1, score D2", "Fit on D2, score D1"),
  aggregation = c(rep("pooled", 6), "patient_macro", rep("pooled", 4)),
  train_domains = c(rep("D1+D2", 7), "D1", "D2", "D1", "D2"),
  eval_partition = c(rep("center2_labeled", 9), rep("center1_labeled", 2)),
  eval_domain = c(rep("D3", 9), "D2", "D1"),
  stringsAsFactors = FALSE
)

# ---- gate variants: each fitter returns function(newdata) -> log-odds, fitted on `tr` only -------------------

.sens_foldid <- function(tr) patient_foldid(tr$patient_id, SENS_FOLDS)

fit_spline_gate <- function(tr, fail, settings, features = OPTICAL_METRICS) {
  g <- fit_gate(tr, fail, features, .sens_foldid(tr), df = settings$df, alpha = settings$alpha)
  function(nd) predict_gate(g, nd, "link")
}

# same recipe with an unpenalised D2 indicator (Center 1 holds D1 and D2); any other domain is scored at the reference
fit_domain_gate <- function(tr, fail, settings, features = OPTICAL_METRICS) {
  sp <- .gl_make_spec(tr, features, settings$df)
  design <- function(d) cbind(.gl_design(sp$spec, sp$scale, d), domain_D2 = as.numeric(d$domain == "D2"))
  x <- design(tr)
  cv <- glmnet::cv.glmnet(x, as.numeric(fail), family = "binomial", alpha = settings$alpha, foldid = .sens_foldid(tr),
                          type.measure = "deviance", penalty.factor = c(rep(1, ncol(x) - 1L), 0))
  function(nd) as.numeric(stats::predict(cv$glmnet.fit, newx = design(nd), s = cv$lambda.1se, type = "link"))
}

fit_xgb_gate <- function(tr, fail, settings, features = OPTICAL_METRICS) {
  x <- as.matrix(tr[features])
  dtrain <- xgboost::xgb.DMatrix(x, label = as.numeric(fail))
  fold <- .sens_foldid(tr)
  params <- c(SENS_XGB$params, list(seed = CFG$seeds$global))
  cv <- xgboost::xgb.cv(params, dtrain, nrounds = SENS_XGB_ROUNDS, folds = lapply(sort(unique(fold)), function(k) which(fold == k)),
                        early_stopping_rounds = SENS_XGB_PATIENCE, verbose = 0)
  fit <- xgboost::xgb.train(params, dtrain, nrounds = cv$early_stop$best_iteration, verbose = 0)
  function(nd) {
    bad <- features[!vapply(features, function(f) all(is.finite(nd[[f]])), NA)]
    if (length(bad)) stop("non-finite values in feature '", bad[1], "'")
    as.numeric(predict(fit, as.matrix(nd[features]), outputmargin = TRUE))
  }
}

fit_gam_gate <- function(tr, fail, settings, features = OPTICAL_METRICS) {
  centre <- vapply(features, function(f) mean(tr[[f]]), 0)
  scl <- vapply(features, function(f) stats::sd(tr[[f]]), 0)
  scl[!is.finite(scl) | scl == 0] <- 1
  prep <- function(d) {   # z-scored metrics from the training mean and SD
    for (f in features) if (any(!is.finite(d[[f]]))) stop("non-finite values in feature '", f, "'")
    as.data.frame(lapply(setNames(features, features), function(f) (d[[f]] - centre[[f]]) / scl[[f]]))
  }
  z <- prep(tr)
  z$fail <- as.numeric(fail)
  terms <- vapply(features, function(f) {
    if (length(unique(z[[f]])) >= SENS_GAM_MIN_UNIQUE) sprintf("s(%s, k = %d, bs = \"cs\")", f, settings$df + 1L) else f
  }, "")
  fit <- mgcv::gam(stats::reformulate(terms, "fail"), family = stats::binomial(), data = z, method = "REML")
  function(nd) as.numeric(stats::predict(fit, newdata = prep(nd), type = "link"))
}

# ---- data and per-variant scoring ---------------------------------------------------------------------------

BG_COLUMNS <- paste0("unet_bg_", OPTICAL_METRICS)

# U-Net rows of one partition (read_eval_frames) with the background-only metrics from predictions.parquet
read_sens_frames <- function(derived, partition) {
  d <- read_eval_frames(derived, partition, "unet")
  part <- partition
  bg <- .gl_collect(arrow::open_dataset(file.path(derived, "predictions.parquet")) |>
    dplyr::filter(partition == !!part, model == "unet") |>
    dplyr::select(dplyr::all_of(c("frame_id", BG_COLUMNS))))
  if (anyNA(match(d$frame_id, bg$frame_id))) stop("background-only metrics missing for some ", part, " frames")
  out <- as.data.frame(dplyr::left_join(d, bg, by = "frame_id"))
  attr(out, "n_excluded_fov") <- attr(d, "n_excluded_fov")
  out
}

finite_rows <- function(d, cols) rowSums(!is.finite(as.matrix(d[cols]))) == 0L

# per-domain mean and SD of the metrics; takes optics only, so no outcome can enter the statistics
domain_norm_stats <- function(optics) {
  stopifnot(setequal(names(optics), c("domain", OPTICAL_METRICS)))
  lapply(split(optics[OPTICAL_METRICS], optics$domain), function(x) {
    sdv <- vapply(x, stats::sd, 0)
    sdv[!is.finite(sdv) | sdv == 0] <- 1
    list(centre = vapply(x, mean, 0), scale = sdv)
  })
}

normalise_by_domain <- function(d, stats_by_domain) {
  for (dm in unique(d$domain)) {
    st <- stats_by_domain[[dm]]
    if (is.null(st)) stop("no normalisation statistics for domain ", dm)
    i <- d$domain == dm
    d[i, OPTICAL_METRICS] <- sweep(sweep(as.matrix(d[i, OPTICAL_METRICS]), 2, st$centre[OPTICAL_METRICS]), 2, st$scale[OPTICAL_METRICS], "/")
  }
  d
}

# Rows to evaluate and their gate log-odds for one variant. c1 / c2: read_sens_frames output with `fail`.
# Nothing here passes a training function any row outside Center 1 (the device statistics use optics columns only).
score_variant <- function(spec, c1, c2, models) {
  settings <- list(df = models$gate$spec$df, alpha = models$gate$alpha)
  fit_on <- function(fitter, tr, ev, features = OPTICAL_METRICS) {
    list(ev = ev, score = fitter(tr, tr$fail, settings, features)(ev), n_excluded = 0L)
  }
  switch(spec$variant,
    locked = , patient_macro = list(ev = c2, score = predict_gate(models$gate, c2, "link"), n_excluded = 0L),
    xgboost = fit_on(fit_xgb_gate, c1, c2),
    gam = fit_on(fit_gam_gate, c1, c2),
    domain_cov = fit_on(fit_domain_gate, c1, c2),
    background_only = {
      ok1 <- finite_rows(c1, BG_COLUMNS)
      ok2 <- finite_rows(c2, BG_COLUMNS)
      out <- fit_on(fit_spline_gate, c1[ok1, ], c2[ok2, ], BG_COLUMNS)
      out$n_excluded <- sum(!ok2)
      out
    },
    device_norm = {
      st <- domain_norm_stats(rbind(c1[c("domain", OPTICAL_METRICS)], c2[c("domain", OPTICAL_METRICS)]))
      n1 <- normalise_by_domain(c1, st)
      n2 <- normalise_by_domain(c2, st)
      out <- fit_on(fit_spline_gate, n1, n2)
      out$ev <- c2   # outcomes and baselines of the original frames; only the gate saw normalised metrics
      out
    },
    lodo_d1_to_d3 = , lodo_d2_to_d3 = fit_on(fit_spline_gate, c1[c1$domain == spec$train_domains, ], c2),
    lodo_d1_to_d2 = , lodo_d2_to_d1 =
      fit_on(fit_spline_gate, c1[c1$domain == spec$train_domains, ], c1[c1$domain == spec$eval_domain, ]),
    stop("unknown sensitivity variant: ", spec$variant)
  )
}

# ---- AURC of the four methods, pooled or patient-macro, with paired differences ------------------------------

sens_stat_fn <- function(macro) {
  function(d) {
    risk <- d$risk
    clus <- if (!is.null(d$.boot_cluster)) d$.boot_cluster else d$patient_id
    by_clus <- split(seq_along(risk), clus)
    one <- function(m) {
      if (m == "random") {
        return(if (macro) mean(vapply(by_clus, function(i) mean(risk[i]), 0)) else mean(risk))
      }
      s <- d[[paste0("s_", m)]]
      if (macro) mean(vapply(by_clus, function(i) aurc(s[i], risk[i]), 0)) else aurc(s, risk)
    }
    setNames(vapply(SENS_METHODS, one, 0), paste(SENS_METHODS, "aurc", sep = "|"))
  }
}

evaluate_variant <- function(spec, scored, entropy, B, seed, cores) {
  ev <- scored$ev
  d <- data.frame(patient_id = ev$patient_id, domain = ev$domain, risk = 1 - ev$iou, fail = ev$fail,
                  s_entropy = ev[[entropy]], s_ens_disagreement = ev$ens_dis, s_gate = scored$score)
  boot <- cluster_boot(d, "patient_id", "domain", B, seed, sens_stat_fn(spec$aggregation == "patient_macro"), cores = cores)
  ci <- boot$ci
  key <- function(m) paste(m, "aurc", sep = "|")
  tab <- data.frame(variant = spec$variant, label = spec$label, aggregation = spec$aggregation,
                    train_domains = spec$train_domains, eval_partition = spec$eval_partition, eval_domain = spec$eval_domain,
                    method = SENS_METHODS, n_frames = nrow(d), n_patients = length(unique(d$patient_id)),
                    n_events = sum(d$fail), n_excluded = scored$n_excluded,
                    aurc = ci$est[match(key(SENS_METHODS), ci$stat)], aurc_lo = ci$lo[match(key(SENS_METHODS), ci$stat)],
                    aurc_hi = ci$hi[match(key(SENS_METHODS), ci$stat)], stringsAsFactors = FALSE)
  refs <- c(random = "random", entropy = "entropy", ens = "ens_disagreement")
  for (r in names(refs)) {
    cols <- paste0("diff_vs_", r, c("", "_lo", "_hi"))
    tab[cols] <- NA_real_
    for (i in which(SENS_METHODS != refs[[r]])) {
      pd <- paired_diff_ci(boot, key(SENS_METHODS[i]), key(refs[[r]]))
      tab[i, cols] <- c(pd$diff, pd$lo, pd$hi)
    }
  }
  tab$status <- switch(spec$variant, locked = "locked primary gate re-evaluated here (reference)",
                       device_norm = SENS_STATUS_DEVICE_NORM, SENS_STATUS)
  list(table = tab, boot = boot)
}

run_sensitivity <- function(paths = default_paths(), B = CFG$statistics$bootstrap_resamples, cores = boot_cores(),
                            variants = SENS_VARIANTS$variant) {
  lock <- read_lock(paths$lock)
  verify_lock_hashes(lock)
  choice <- lock$entropy_baseline_choice
  if (!.gl_nonempty(choice)) stop("lock file has no entropy_baseline_choice", call. = FALSE)
  models <- load_gate_models(lock)
  thr <- lock_failure_threshold(lock)
  if (!all(variants %in% SENS_VARIANTS$variant)) stop("unknown variant(s): ", paste(setdiff(variants, SENS_VARIANTS$variant), collapse = ", "))

  c1 <- read_sens_frames(paths$derived, "center1_labeled")
  c2 <- read_sens_frames(paths$derived, "center2_labeled")
  c1$fail <- c1$iou < thr
  c2$fail <- c2$iou < thr

  tabs <- scores <- boots <- list()
  for (v in variants) {
    spec <- as.list(SENS_VARIANTS[SENS_VARIANTS$variant == v, ])
    sc <- score_variant(spec, c1, c2, models)
    ev <- evaluate_variant(spec, sc, choice, B, CFG$seeds$bootstrap, cores)
    tabs[[v]] <- ev$table
    boots[[v]] <- ev$boot
    scores[[v]] <- data.frame(frame_id = sc$ev$frame_id, score = sc$score)
  }
  tab <- do.call(rbind, tabs)
  rownames(tab) <- NULL
  write_csv_guarded(tab, file.path(paths$stats, "sensitivity.csv"))
  list(table = tab, scores = scores, boots = boots, threshold = thr, entropy_choice = choice)
}

register_08 <- function(res) {
  tb <- res$table
  g <- tb[tb$method == "gate", ]
  for (i in seq_len(nrow(g))) {
    r <- g[i, ]
    k <- function(stat) paste("sens", r$variant, stat, sep = "_")
    reg(k("n_frames"), as.integer(r$n_frames))
    reg(k("aurc_ci"), c(r$aurc, r$aurc_lo, r$aurc_hi), digits = 3, fmt = "ci", note = r$status)
    for (ref in c("random", "entropy", "ens")) {
      cols <- paste0("diff_vs_", ref, c("", "_lo", "_hi"))
      reg(k(paste0("diff_", ref, "_ci")), as.numeric(unlist(r[cols])), digits = 3, fmt = "ci",
          note = paste("gate minus", ref, "AURC, negative favours the gate;", r$status))
    }
    if (r$variant == "background_only") {
      reg(k("excluded"), as.integer(r$n_excluded), note = "Center 2 frames without finite background-only metrics")
    }
  }
  ext <- g[g$eval_partition == "center2_labeled" & g$variant != "locked", ]
  reg("sens_n_ext_variants", nrow(ext), note = "exploratory variants scored on Center 2, excluding the locked reference")
  reg("sens_n_ext_beat_random", sum(ext$diff_vs_random_hi < 0),
      note = "of those, variants whose 95% CI for gate minus random AURC lies below zero")
}

if (sys.nframe() == 0L) {
  paths <- default_paths()
  t0 <- Sys.time()
  B <- CFG$statistics$bootstrap_resamples
  res <- run_sensitivity(paths, B)
  register_08(res)
  guard_write(file.path(paths$results, "results", "parts", "08_sensitivity.json"))
  write_results_part("08_sensitivity", root = paths$results)
  g <- res$table[res$table$method == "gate", c("variant", "n_frames", "n_excluded", "aurc", "aurc_lo", "aurc_hi",
                                              "diff_vs_random", "diff_vs_random_hi", "diff_vs_ens", "diff_vs_ens_hi")]
  cat("08 sensitivity (exploratory), gate rows; external = Center 2 unless the variant says otherwise\n")
  print(g, digits = 3, row.names = FALSE)
  cat(sprintf("(%.1f s, B = %d)\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), B))
}
