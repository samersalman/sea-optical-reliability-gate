# Statistics helpers for 03_association.R, 06_perturb.R (and anything else that needs a
# patient-clustered bootstrap, a cluster-robust logistic model or a Spearman CI). Source 00_setup.R
# first, and results_lib.R if you call the assoc_reg_* wrappers. Functions only; nothing is read or
# written at source time. All names start with assoc_ so they cannot clash with stats_lib.R.
#
# Bootstrap (patients are the unit; strata, if given, must be constant within a patient, normally domain)
#   assoc_boot_draws(patient, strata, B, seed)  -> draws matrix (B x patients) + row index per patient
#   assoc_boot(patient, fn, strata, B, seed, cores) -> B x k matrix; fn(rows) returns a named numeric vector
#   assoc_boot_ci(est_fn, patient, strata, ...) -> data.frame(stat, est, lo, hi, n_valid) (percentile CI);
#                                                  the B x k replicate matrix is attr(., "boot")
#   Draws are generated in the parent process, so results do not depend on `cores`.
# Rank statistics
#   assoc_auroc(score, label), assoc_auroc_strat(score, label, strata)   Mann-Whitney, ties count 0.5
#   assoc_spearman(x, y), assoc_spearman_boot(x, y, patient, strata, ...)  x may be a matrix; one output row per column
# Logistic models (y is 0/1, X a numeric design matrix WITHOUT intercept, cluster = patient)
#   assoc_covariates(d)             dummy-coded confounders; single-level factors are reported, not hidden
#   assoc_drop_aliased(X)           drops constant and linearly dependent columns, says why
#   assoc_vif(X), assoc_epv(y, k)   VIF from the correlation matrix; events per variable
#   assoc_fit_robust(y, X, cluster) glm + sandwich::vcovCL (HC1, cadjust), robust and model-based SE
#   assoc_fit_ridge(...)            ridge on the penalised columns, patient-bootstrap CI (EPV fallback)
#   assoc_fit_model(...)            drop aliased -> EPV check -> robust fit, or ridge when EPV < 10
# Perturbation
#   assoc_first_below(sev, med, thr), assoc_perturb_curves(pt, ...) -> list(curves, thresholds), assoc_perturb_rho(pt, metrics)
# Data and registry
#   assoc_metric_matrix(d, metrics, ref), assoc_read_unet(partition), assoc_failure_threshold(), assoc_lock_complete(),
#   assoc_reg_ci(key, est, lo, hi, digits), assoc_reg_num(key, x, digits, fmt, unit)
# Output guard
#   assoc_out(...), assoc_write_csv(df, path): refuse any path outside work/synthetic/out in synthetic mode.

if (!exists("CFG") || !exists("p", mode = "function")) stop("source 00_setup.R before assoc_lib.R")

ASSOC_MIN_EPV <- 10                      # events per variable below which the penalised fit replaces the plain one
ASSOC_LOG_METRICS <- "sharp_tenengrad"   # gradient energy spans orders of magnitude; ORs for it are per SD of log

ASSOC_B <- function() CFG$statistics$bootstrap_resamples
ASSOC_SEED <- function() CFG$seeds$bootstrap
ASSOC_LEVEL <- function() CFG$statistics$ci_level
ASSOC_CORES <- function() {
  env <- suppressWarnings(as.integer(Sys.getenv("OPTICS_CORES", "")))
  if (!is.na(env) && env >= 1L) return(env)
  max(1L, min(4L, parallel::detectCores() - 2L))
}

# ---- output guard ---------------------------------------------------------------------------

.assoc_clean_path <- function(path) {
  parts <- strsplit(normalizePath(path, mustWork = FALSE), "/", fixed = TRUE)[[1]]
  out <- character()
  for (s in parts) {
    if (s == "..") out <- head(out, -1) else if (nzchar(s) && s != ".") out <- c(out, s)
  }
  paste0("/", paste(out, collapse = "/"))
}

assoc_guard_path <- function(path) {
  if (SYNTH) {
    out <- .assoc_clean_path(file.path(SYNTH_DIR, "out"))
    here <- .assoc_clean_path(path)
    if (!startsWith(here, paste0(out, "/"))) {
      stop("synthetic mode: refusing to write outside work/synthetic/out: ", path, call. = FALSE)
    }
  }
  invisible(path)
}

# p() path with the guard applied and the directory created
assoc_out <- function(...) {
  path <- assoc_guard_path(p(...))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  path
}

assoc_write_csv <- function(df, path) {
  assoc_guard_path(path)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(df, path, row.names = FALSE)
  invisible(path)
}

# results_root() is where write_results_part() writes; in synthetic mode it must be work/synthetic/out
assoc_guard_results_root <- function() {
  if (exists("results_root", mode = "function")) assoc_guard_path(file.path(results_root(), "results", "x"))
  invisible(TRUE)
}

# ---- lock and threshold ---------------------------------------------------------------------

.assoc_lock <- function(path = p("config", "analysis_lock.yaml")) {
  if (file.exists(path)) yaml::read_yaml(path) else list()
}

# the failure threshold may be revised only at G2 and then lives in the lock file
assoc_failure_threshold <- function(path = p("config", "analysis_lock.yaml")) {
  thr <- .assoc_lock(path)$failure_iou_threshold
  if (is.numeric(thr) && length(thr) == 1L) thr else CFG$endpoints$failure_iou_threshold
}

# TRUE once the gate hash is recorded, i.e. Center 2 outcomes may be opened
assoc_lock_complete <- function(path = p("config", "analysis_lock.yaml")) {
  h <- .assoc_lock(path)$gate_rds_sha256
  is.character(h) && length(h) == 1L && nzchar(h)
}

# ---- patient bootstrap ----------------------------------------------------------------------

.assoc_with_seed <- function(seed, expr) {
  old <- if (exists(".Random.seed", globalenv(), inherits = FALSE)) get(".Random.seed", globalenv()) else NULL
  on.exit(if (is.null(old)) rm(".Random.seed", envir = globalenv()) else assign(".Random.seed", old, globalenv()))
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  expr
}

assoc_boot_draws <- function(patient, strata = NULL, B = ASSOC_B(), seed = ASSOC_SEED()) {
  patient <- as.character(patient)
  pats <- unique(patient)
  rows <- split(seq_along(patient), factor(patient, levels = pats))
  strata <- if (is.null(strata)) rep("all", length(patient)) else as.character(strata)
  chk <- unique(data.frame(patient = patient, strata = strata))
  if (anyDuplicated(chk$patient)) stop("assoc_boot_draws: a patient belongs to more than one stratum")
  by_stratum <- split(seq_along(pats), factor(chk$strata[match(pats, chk$patient)]))
  draws <- .assoc_with_seed(seed, do.call(cbind, lapply(by_stratum, function(ix) {
    matrix(ix[sample.int(length(ix), B * length(ix), replace = TRUE)], nrow = B)
  })))
  dimnames(draws) <- NULL
  list(rows = unname(rows), draws = draws, patients = pats, stratum = chk$strata[match(pats, chk$patient)])
}

assoc_boot <- function(patient, fn, strata = NULL, B = ASSOC_B(), seed = ASSOC_SEED(),
                       cores = ASSOC_CORES(), draws = NULL) {
  d <- if (is.null(draws)) assoc_boot_draws(patient, strata, B, seed) else draws
  B <- nrow(d$draws)
  one <- function(b) {
    rows <- unlist(d$rows[d$draws[b, ]], use.names = FALSE)
    tryCatch(fn(rows), error = function(e) structure(NA_real_, err = conditionMessage(e)))
  }
  res <- parallel::mclapply(seq_len(B), one, mc.cores = cores)
  errs <- unique(unlist(lapply(res, function(r) attr(r, "err"))))
  if (length(errs)) warning("assoc_boot: ", length(errs), " distinct error(s) in replicates, first: ", errs[1], call. = FALSE)
  k <- max(lengths(res))
  nm <- names(res[[which.max(lengths(res))]])
  out <- t(vapply(res, function(r) if (length(r) == k) as.numeric(r) else rep(NA_real_, k), numeric(k)))
  if (k == 1L) out <- matrix(out, ncol = 1L)
  colnames(out) <- nm
  out
}

assoc_pct_ci <- function(x, level = ASSOC_LEVEL()) {
  x <- x[is.finite(x)]
  if (length(x) < 20L) return(c(NA_real_, NA_real_))
  unname(stats::quantile(x, c((1 - level) / 2, 1 - (1 - level) / 2), names = FALSE))
}

# est_fn(rows) -> named numeric vector; rows = 1..n gives the point estimate
assoc_boot_ci <- function(est_fn, patient, strata = NULL, B = ASSOC_B(), seed = ASSOC_SEED(),
                          cores = ASSOC_CORES(), draws = NULL, level = ASSOC_LEVEL()) {
  est <- est_fn(seq_along(patient))
  bt <- assoc_boot(patient, est_fn, strata, B, seed, cores, draws)
  ci <- vapply(seq_along(est), function(j) assoc_pct_ci(bt[, j], level), numeric(2))
  out <- data.frame(stat = names(est), est = as.numeric(est), lo = ci[1, ], hi = ci[2, ],
                    n_valid = colSums(is.finite(bt)), row.names = NULL, stringsAsFactors = FALSE)
  attr(out, "boot") <- bt
  out
}

# ---- rank statistics ------------------------------------------------------------------------

assoc_auroc <- function(score, label) {
  ok <- !is.na(score) & !is.na(label)
  y <- as.logical(label[ok])
  n1 <- sum(y)
  n0 <- sum(!y)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  (sum(rank(score[ok])[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# AUROC within strata, combined with pair-count weights (both classes must occur in a stratum to count)
assoc_auroc_strat <- function(score, label, strata) {
  num <- 0
  den <- 0
  for (s in unique(strata)) {
    i <- strata == s
    a <- assoc_auroc(score[i], label[i])
    if (is.na(a)) next
    w <- sum(label[i] == 1) * sum(label[i] == 0)
    num <- num + w * a
    den <- den + w
  }
  if (den == 0) NA_real_ else num / den
}

assoc_spearman <- function(x, y) {
  X <- as.matrix(x)
  r <- suppressWarnings(stats::cor(apply(X, 2, rank), rank(y)))
  setNames(as.numeric(r), colnames(X))
}

assoc_spearman_boot <- function(x, y, patient, strata = NULL, ...) {
  X <- as.matrix(x)
  if (is.null(colnames(X))) colnames(X) <- paste0("v", seq_len(ncol(X)))
  out <- assoc_boot_ci(function(rows) assoc_spearman(X[rows, , drop = FALSE], y[rows]), patient, strata, ...)
  out$n <- length(y)
  out$n_patients <- length(unique(patient))
  names(out)[1] <- "variable"
  out
}

# ---- design, aliasing, VIF, EPV -------------------------------------------------------------

# Dummy-codes the confounders (reference = most frequent level) and z-scores the predicted area
# (sqrt scale: proportions that can be exactly 0). Returns the matrix and the factors it could not code.
assoc_covariates <- function(d, factors = c("domain", "channel", "region", "file_format", "instrument"),
                             area = "unet_pred_area_frac") {
  cols <- list()
  single <- data.frame(term = character(), reason = character())
  for (f in factors) {
    v <- as.character(d[[f]])
    if (anyNA(v)) stop("assoc_covariates: missing values in ", f)
    lev <- names(sort(table(v), decreasing = TRUE))
    if (length(lev) < 2L) {
      single <- rbind(single, data.frame(term = f, reason = paste0("single level in this sample (", lev, ")")))
      next
    }
    for (l in setdiff(sort(lev), lev[1])) cols[[paste0(f, l)]] <- as.numeric(v == l)
  }
  if (!is.null(area)) {
    a <- sqrt(d[[area]])
    cols[["pred_area_z"]] <- (a - mean(a)) / stats::sd(a)
  }
  list(X = do.call(cbind, cols), single_level = single)
}

# Drops constant columns and columns that are linear combinations of EARLIER columns (pivoted QR),
# and names the terms each dropped column equals. Column order is therefore the priority order.
assoc_drop_aliased <- function(X, tol = 1e-7) {
  X <- as.matrix(X)
  nm <- colnames(X)
  const <- apply(X, 2, function(v) all(v == v[1]))
  keep <- !const
  dropped <- data.frame(term = nm[const], reason = rep("constant in this sample", sum(const)))
  if (sum(keep) > 0L) {
    q <- qr(cbind(1, X[, keep, drop = FALSE]), tol = tol)
    piv <- q$pivot[seq_len(q$rank)]
    kept_idx <- which(keep)[sort(piv[piv != 1L] - 1L)]
    lost <- setdiff(which(keep), kept_idx)
    for (j in lost) {
      fit <- stats::lm.fit(cbind(1, X[, kept_idx, drop = FALSE]), X[, j])
      co <- fit$coefficients[-1]
      with_terms <- nm[kept_idx][abs(co) > 1e-6 & is.finite(co)]
      dropped <- rbind(dropped, data.frame(term = nm[j],
        reason = paste0("aliased with ", if (length(with_terms)) paste(with_terms, collapse = " + ") else "the intercept")))
    }
    keep <- seq_along(nm) %in% kept_idx
  }
  list(X = X[, keep, drop = FALSE], aliased = dropped)
}

assoc_vif <- function(X) {
  X <- as.matrix(X)
  if (ncol(X) < 2L) return(setNames(rep(1, ncol(X)), colnames(X)))
  v <- tryCatch(diag(solve(stats::cor(X))), error = function(e) rep(Inf, ncol(X)))
  setNames(as.numeric(v), colnames(X))
}

assoc_epv <- function(y, n_params, min_epv = ASSOC_MIN_EPV) {
  events <- sum(y == 1)
  limiting <- min(events, sum(y == 0))
  epv <- limiting / n_params
  list(events = events, n_params = n_params, epv = epv, flag = epv < min_epv)
}

# ---- logistic fits --------------------------------------------------------------------------

assoc_fit_robust <- function(y, X, cluster, level = ASSOC_LEVEL()) {
  X <- as.matrix(X)
  d <- as.data.frame(X)
  names(d) <- paste0("x", seq_len(ncol(X)))
  d$.y <- y
  warns <- character()
  fit <- withCallingHandlers(
    stats::glm(.y ~ ., data = d, family = stats::binomial()),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  V <- sandwich::vcovCL(fit, cluster = as.character(cluster), type = "HC1", cadjust = TRUE)
  se_rob <- sqrt(diag(V))
  se_naive <- sqrt(diag(stats::vcov(fit)))
  est <- stats::coef(fit)
  z <- stats::qnorm(1 - (1 - level) / 2)
  tab <- data.frame(term = c("(Intercept)", colnames(X)), coef = unname(est),
                    se_robust = unname(se_rob), se_naive = unname(se_naive),
                    row.names = NULL, stringsAsFactors = FALSE)
  tab$se_ratio <- tab$se_robust / tab$se_naive
  tab$or <- exp(tab$coef)
  tab$or_lo <- exp(tab$coef - z * tab$se_robust)
  tab$or_hi <- exp(tab$coef + z * tab$se_robust)
  tab$p <- 2 * stats::pnorm(-abs(tab$coef / tab$se_robust))
  separation <- !fit$converged || any(grepl("numerically 0 or 1", warns)) || any(tab$se_naive > 10, na.rm = TRUE)
  list(table = tab, converged = fit$converged, separation = separation, warnings = unique(warns),
       method = "cluster-robust logistic", ci_method = "Wald, patient-clustered sandwich")
}

# Ridge on the columns flagged `penalised` (the metrics); confounders stay unpenalised. Lambda from
# patient-grouped CV (minimum deviance); the CI is a patient-bootstrap percentile at that lambda.
assoc_fit_ridge <- function(y, X, cluster, penalised, strata = NULL, B = ASSOC_B(), seed = ASSOC_SEED(),
                            cores = ASSOC_CORES(), nfolds = 10L, level = ASSOC_LEVEL()) {
  X <- as.matrix(X)
  pf <- as.numeric(penalised)
  pats <- unique(as.character(cluster))
  foldid <- .assoc_with_seed(seed, sample(rep_len(seq_len(min(nfolds, length(pats))), length(pats))))
  foldid <- foldid[match(as.character(cluster), pats)]
  cv <- .assoc_with_seed(seed, glmnet::cv.glmnet(X, y, family = "binomial", alpha = 0, penalty.factor = pf,
                                                 foldid = foldid, standardize = FALSE, type.measure = "deviance"))
  lam <- cv$lambda.min
  path <- exp(seq(log(lam * 10), log(lam), length.out = 6))   # glmnet is unreliable with a single lambda
  coefs <- function(rows) {
    f <- suppressWarnings(glmnet::glmnet(X[rows, , drop = FALSE], y[rows], family = "binomial", alpha = 0,
                                         penalty.factor = pf, standardize = FALSE, lambda = path, thresh = 1e-10))
    b <- as.numeric(stats::coef(f, s = lam))[-1]
    setNames(b, colnames(X))
  }
  ci <- assoc_boot_ci(coefs, cluster, strata, B, seed, cores, level = level)
  tab <- data.frame(term = ci$stat, coef = ci$est, se_robust = NA_real_, se_naive = NA_real_, se_ratio = NA_real_,
                    or = exp(ci$est), or_lo = exp(ci$lo), or_hi = exp(ci$hi), p = NA_real_, stringsAsFactors = FALSE)
  list(table = tab, converged = TRUE, separation = FALSE, warnings = character(),
       method = "ridge logistic", ci_method = "patient-bootstrap percentile at CV lambda", lambda = lam)
}

# One call per model: aliased columns out, EPV counted on what is left, then the plain cluster-robust
# fit, or (EPV < min_epv) the ridge fit with a warning. `penalised` names the columns that get the penalty.
assoc_fit_model <- function(y, X, cluster, penalised, strata = NULL, label = "model",
                            min_epv = ASSOC_MIN_EPV, force_plain = FALSE, ...) {
  kept <- assoc_drop_aliased(X)
  ep <- assoc_epv(y, ncol(kept$X), min_epv)
  vif <- assoc_vif(kept$X)
  use_ridge <- ep$flag && !force_plain
  if (ep$flag) {
    warning(sprintf("%s: events per variable = %.1f (< %d; %d events, %d parameters)%s", label, ep$epv, min_epv,
                    ep$events, ep$n_params, if (use_ridge) "; using the ridge fit with a patient-bootstrap CI" else ""),
            call. = FALSE)
  }
  fit <- if (use_ridge) {
    assoc_fit_ridge(y, kept$X, cluster, colnames(kept$X) %in% penalised, strata, ...)
  } else {
    assoc_fit_robust(y, kept$X, cluster)
  }
  c(fit, list(aliased = kept$aliased, epv = ep, vif = vif, label = label,
              ridge_used = use_ridge, n = length(y), kept_terms = colnames(kept$X)))
}

# z-scores metric columns (log first for ASSOC_LOG_METRICS). `ref` = a previous result: reuses its centre and
# scale, so ORs from another sample stay "per SD of the reference sample".
assoc_metric_matrix <- function(d, metrics, ref = NULL, log_metrics = ASSOC_LOG_METRICS) {
  tf <- function(m) {
    v <- d[[m]]
    if (m %in% log_metrics) {
      if (any(v <= 0, na.rm = TRUE)) stop("assoc_metric_matrix: non-positive values in log-scaled metric ", m)
      v <- log(v)
    }
    v
  }
  raw <- vapply(metrics, tf, numeric(nrow(d)))
  colnames(raw) <- metrics
  centre <- if (is.null(ref)) colMeans(raw) else attr(ref, "centre")[metrics]
  scale <- if (is.null(ref)) apply(raw, 2, stats::sd) else attr(ref, "scale")[metrics]
  Z <- sweep(sweep(raw, 2, centre), 2, scale, "/")
  attr(Z, "centre") <- centre
  attr(Z, "scale") <- scale
  Z
}

# U-Net rows of one partition, filtered inside arrow so other partitions are never loaded into R
assoc_read_unet <- function(part) {
  arrow::open_dataset(p("derived", "predictions.parquet")) |>
    dplyr::filter(partition == !!part, model == "unet") |>
    dplyr::select(frame_id, source, iou, pred_area_frac) |>
    dplyr::collect() |>
    dplyr::rename(unet_source = source, unet_iou = iou, unet_pred_area_frac = pred_area_frac)
}

# ---- perturbation ---------------------------------------------------------------------------

# Median IoU curve -> first severity (in order) with median < thr, and the linearly interpolated crossing.
assoc_first_below <- function(sev, med, thr) {
  o <- order(sev)
  sev <- sev[o]
  med <- med[o]
  i <- which(med < thr)[1]
  if (is.na(i)) return(list(first = NA_real_, interp = NA_real_))
  interp <- if (i == 1L) sev[1] else sev[i - 1] + (med[i - 1] - thr) / (med[i - 1] - med[i]) * (sev[i] - sev[i - 1])
  list(first = sev[i], interp = interp)
}

# Dose-response per model x family x severity with patient-clustered bands (median and mean IoU), and per
# model x family the first severity with median IoU < thr (point estimate, interpolated crossing, bootstrap
# quantiles; "never within the grid" counts as one step past the last severity in the replicates).
# pt: perturb.parquet rows (contract columns), stratified by domain in the bootstrap.
assoc_perturb_curves <- function(pt, thr = CFG$perturbation$rule_first_below_iou, B = ASSOC_B(),
                                 seed = ASSOC_SEED(), cores = ASSOC_CORES(), level = ASSOC_LEVEL()) {
  cell <- paste(pt$model, pt$family, pt$severity, sep = "|")
  frames <- unique(pt[c("frame_id", "patient_id", "domain")])
  if (anyDuplicated(frames$frame_id)) stop("assoc_perturb_curves: a frame has more than one patient or domain")
  cells <- sort(unique(cell))
  W <- matrix(NA_real_, nrow(frames), length(cells))
  W[cbind(match(pt$frame_id, frames$frame_id), match(cell, cells))] <- pt$iou
  est_fn <- function(rows) {
    sub <- W[rows, , drop = FALSE]
    c(setNames(apply(sub, 2, stats::median, na.rm = TRUE), paste0("median@", cells)),
      setNames(colMeans(sub, na.rm = TRUE), paste0("mean@", cells)))
  }
  ci <- assoc_boot_ci(est_fn, frames$patient_id, frames$domain, B, seed, cores, level = level)
  bt <- attr(ci, "boot")

  parts <- strsplit(cells, "|", fixed = TRUE)
  curves <- data.frame(model = vapply(parts, `[`, "", 1), family = vapply(parts, `[`, "", 2),
                       severity = as.integer(vapply(parts, `[`, "", 3)), stringsAsFactors = FALSE)
  curves$n_frames <- colSums(!is.na(W))
  curves$n_patients <- vapply(seq_along(cells), function(j) length(unique(frames$patient_id[!is.na(W[, j])])), integer(1))
  for (s in c("median", "mean")) {
    i <- match(paste0(s, "@", cells), ci$stat)
    curves[[paste0(s, "_iou")]] <- ci$est[i]
    curves[[paste0(s, "_lo")]] <- ci$lo[i]
    curves[[paste0(s, "_hi")]] <- ci$hi[i]
  }
  curves$param <- vapply(cells, function(ce) {
    v <- unique(round(pt$param[cell == ce], 10))
    if (length(v) == 1L) v else NA_real_
  }, numeric(1))
  o <- order(curves$model, curves$family, curves$severity)
  curves <- curves[o, ]
  rownames(curves) <- NULL

  top <- max(curves$severity)
  thresholds <- do.call(rbind, lapply(split(seq_len(nrow(curves)), paste(curves$model, curves$family)), function(ix) {
    cv <- curves[ix, ]
    fb <- assoc_first_below(cv$severity, cv$median_iou, thr)
    cols <- match(paste0("median@", cv$model[1], "|", cv$family[1], "|", cv$severity), colnames(bt))
    rep_first <- apply(bt[, cols, drop = FALSE], 1, function(m) {
      if (anyNA(m)) return(NA_real_)
      f <- assoc_first_below(cv$severity, m, thr)$first
      if (is.na(f)) top + 1 else f
    })
    q <- assoc_pct_ci(rep_first, level)
    data.frame(model = cv$model[1], family = cv$family[1], threshold_iou = thr,
               baseline_median_iou = cv$median_iou[cv$severity == 0],
               first_severity_below = fb$first, severity_interp = fb$interp,
               param_at_first = if (is.na(fb$first)) NA_real_ else cv$param[cv$severity == fb$first],
               first_severity_lo = if (is.na(q[1]) || q[1] > top) NA_real_ else q[1],
               first_severity_hi = if (is.na(q[2]) || q[2] > top) NA_real_ else q[2],
               prob_never = mean(rep_first == top + 1, na.rm = TRUE),
               stringsAsFactors = FALSE)
  }))
  rownames(thresholds) <- NULL
  list(curves = curves, thresholds = thresholds)
}

# Spearman rho between each metric and severity, per family, with patient-clustered CI. `pt` has one row
# per frame x family x severity (metrics do not depend on the segmenter), severity 0 included in every family.
assoc_perturb_rho <- function(pt, metrics, ...) {
  do.call(rbind, lapply(sort(unique(pt$family)), function(f) {
    d <- pt[pt$family == f, ]
    r <- assoc_spearman_boot(as.matrix(d[metrics]), d$severity, d$patient_id, d$domain, ...)
    r$family <- f
    r
  }))
}

# ---- registry wrappers (results_lib.R) -----------------------------------------------------

# est (lo to hi) at fixed digits; a missing estimate is registered as the string "not estimable"
assoc_reg_ci <- function(key, est, lo, hi, digits = 2, note = NULL) {
  if (all(is.finite(c(est, lo, hi)))) reg(key, c(est, lo, hi), digits = digits, fmt = "ci", note = note)
  else reg(key, "not estimable", note = note)
}

assoc_reg_num <- function(key, x, digits = NULL, fmt = NULL, unit = NULL, note = NULL) {
  if (is.finite(x)) reg(key, x, digits = digits, fmt = fmt, unit = unit, note = note)
  else reg(key, "not estimable", note = note)
}
