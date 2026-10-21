# Statistics library for the reliability gate (04a, 04b, 05 and the sensitivity analyses).
# Source 00_setup.R first (and 01_load.R if you use .read_contract); base R + splines + glmnet + parallel + arrow.
#
# Risk-coverage   risk_coverage(score, risk)         data frame k, coverage, selective_risk
#   score = predicted risk, HIGHER = LESS RELIABLE. Frames are retained from the lowest score up;
#   risk = 1 - IoU. Tied scores are retained in random order in expectation (each tied frame
#   counts with the mean risk of its tie group), so a constant score gives the random AURC.
#                 aurc(score, risk)                   mean of the cumulative selective risk over all coverages
#                 selective_risk_at(score, risk, coverage)   coverage in (0, 1], k = max(1, round(coverage * n))
#                 random_aurc(risk) (= mean risk), oracle_aurc(risk) (order by the true risk)
# Discrimination  auroc_fail(score, fail)             rank based, ties get midranks, NA if one class is empty
#                 threshold_at_sens / spec_at_sens / sens_at_spec   operating points; flag = score >= threshold
#                 calibration_fit(p, fail)            c(slope, intercept, intercept_free), Van Calster:
#                   slope = coefficient of logit(p) in glm(fail ~ logit(p)); intercept = calibration-in-the-large
#                   (glm(fail ~ offset(logit(p))), slope fixed at 1); intercept_free = intercept of the slope model
# Bootstrap       boot_plan(data, patient, strata, B, seed)   patient resamples inside strata, drawn once
#                 cluster_boot(data, patient, strata, B, seed, stat_fn, plan = NULL, cores, level)
#                   stat_fn(d) must return a named numeric vector; d carries a `.boot_cluster` column (one id per
#                   DRAWN patient copy). Returns est, reps (B x k), ci (stat, est, lo, hi), plan.
#                   Evaluate every method inside ONE stat_fn (or pass the same `plan`) so differences are paired.
#                 paired_diff_ci(boot, a, b)           percentile CI of reps[, a] - reps[, b]
#                 patient_macro(stat_fn)               wraps a stat_fn: computed per patient, then averaged
# Gate objects    fit_gate(), fit_area(), fit_stack(), predict_gate(), predict_oof(), range_check()
#   gate_obj = list(kind, spec = list(features, df, knots, boundary), fit, lambda, scale = list(centre, scale),
#                   alpha, created, flag_thresholds)
#   kind "optical_gate"/"predicted_area": z-scored features (scale from the TRAINING data), natural splines per
#   feature (knots = training tertiles, boundary = training range, linear beyond; a feature with no usable knot
#   enters linearly), then `fit`: a glmnet fit (optical_gate; `lambda` = 1-SE value) or list(beta) (plain logistic).
#   kind "gate_plus_ensemble": list(kind, spec = list(features = "ens_dis"), fit = list(beta), base = <gate_obj>);
#   link = beta[1] + beta[2] * predict_gate(base, newdata, "link") + beta[3] * ens_dis.
#   predict_gate(gate_obj, newdata, type = c("response", "link")): newdata is a data frame with the columns named in
#   spec$features on the ORIGINAL scale (plus ens_dis for the stack); it returns the failure probability
#   (or the log-odds with type = "link", which has no saturation ties and is what the evaluation ranks on).
#   flag_thresholds (optical gate only): named numeric, names = coverage levels, values = Center 1 OOF quantiles of
#   the link; withhold frames whose link exceeds the value of the chosen coverage.
# Lock            read_lock(), lock_update(), lock_fields_hash(), verify_lock_hashes()
#   verify_lock_hashes(lock) recomputes lock_fields_sha256 (all lock keys but itself and locked_at) and the RDS hashes;
#   every script that uses the lock's threshold, margin, epochs or entropy choice calls it first.
# Output guard    guard_write(path): in synthetic mode refuses any path outside work/synthetic/out
# Evaluation      read_eval_frames(), score_methods(), evaluate_scored(), register_gate_results()

if (!exists("p", mode = "function") || !exists("CFG")) stop("source 00_setup.R before stats_lib.R")
# a glmnet object read back from RDS predicts only once the package's S3 methods are registered
if (!requireNamespace("glmnet", quietly = TRUE)) stop("package glmnet is required")

GATE_METHODS <- c("random", "predicted_area", "entropy", "ens_disagreement", "optical_gate",
                  "gate_plus_ensemble", "oracle")
GATE_PROB_METHODS <- c("predicted_area", "optical_gate", "gate_plus_ensemble")
GATE_SCORE_METHODS <- c("predicted_area", "entropy", "ens_disagreement", "optical_gate", "gate_plus_ensemble")
GATE_LOCK_OBJECTS <- c("gate_rds", "gate_area_rds", "gate_stack_rds", "gate_folds_rds")

# id, a, b, whether the comparison is a non-inferiority question (lower AURC is better)
GATE_COMPARISONS <- data.frame(
  comparison = c("gate_vs_random", "gate_vs_area", "gate_vs_entropy", "gate_vs_ens", "gateens_vs_ens"),
  a = c("optical_gate", "optical_gate", "optical_gate", "optical_gate", "gate_plus_ensemble"),
  b = c("random", "predicted_area", "entropy", "ens_disagreement", "ens_disagreement"),
  noninf = c(FALSE, FALSE, TRUE, TRUE, FALSE),
  stringsAsFactors = FALSE
)

# ---- risk-coverage ---------------------------------------------------------------------------

.gl_check_xy <- function(score, risk) {
  if (length(score) != length(risk)) stop("score and risk differ in length")
  if (!length(risk)) stop("empty input")
  if (anyNA(score) || anyNA(risk)) stop("score and risk must not contain NA")
}

# cumulative selective risk after retaining the k lowest-score frames, k = 1..n
.gl_cum_risk <- function(score, risk) {
  .gl_check_xy(score, risk)
  o <- order(score)
  s <- score[o]
  r <- risk[o]
  grp <- cumsum(c(TRUE, s[-1L] != s[-length(s)]))
  if (grp[length(grp)] < length(grp)) r <- as.vector(rowsum(r, grp, reorder = FALSE) / tabulate(grp))[grp]
  cumsum(r) / seq_along(r)
}

.gl_cov_index <- function(coverage, n) {
  if (any(coverage <= 0 | coverage > 1)) stop("coverage must be in (0, 1]")
  pmin(n, pmax(1L, as.integer(round(coverage * n))))
}

risk_coverage <- function(score, risk) {
  cum <- .gl_cum_risk(score, risk)
  n <- length(cum)
  data.frame(k = seq_len(n), coverage = seq_len(n) / n, selective_risk = cum)
}

aurc <- function(score, risk) mean(.gl_cum_risk(score, risk))

selective_risk_at <- function(score, risk, coverage) {
  cum <- .gl_cum_risk(score, risk)
  cum[.gl_cov_index(coverage, length(cum))]
}

random_aurc <- function(risk) mean(risk)

oracle_aurc <- function(risk) aurc(risk, risk)

# ---- discrimination and calibration ----------------------------------------------------------

auroc_fail <- function(score, fail) {
  fail <- as.logical(fail)
  n1 <- sum(fail)
  n0 <- length(fail) - n1
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  (sum(rank(score)[fail]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# the k-th largest failure score, k = ceiling(sens * n_fail): at least `sens` of failures score >= threshold
threshold_at_sens <- function(score, fail, sens) {
  sf <- sort(score[as.logical(fail)], decreasing = TRUE)
  if (!length(sf)) return(NA_real_)
  sf[max(1L, ceiling(sens * length(sf) - 1e-9))]
}

spec_at_sens <- function(score, fail, sens) {
  fail <- as.logical(fail)
  t <- threshold_at_sens(score, fail, sens)
  if (is.na(t) || all(fail)) return(NA_real_)
  mean(score[!fail] < t)
}

# highest sensitivity with at least `spec` specificity (flag = score strictly above the m-th lowest non-failure score)
sens_at_spec <- function(score, fail, spec) {
  fail <- as.logical(fail)
  if (!any(fail) || all(fail)) return(NA_real_)
  sn <- sort(score[!fail])
  mean(score[fail] > sn[max(1L, ceiling(spec * length(sn) - 1e-9))])
}

calibration_fit <- function(p, fail, eps = 1e-8) {
  fail <- as.numeric(fail)
  out <- c(slope = NA_real_, intercept = NA_real_, intercept_free = NA_real_)
  if (!any(fail == 1) || all(fail == 1)) return(out)
  lp <- qlogis(pmin(pmax(p, eps), 1 - eps))
  ctl <- stats::glm.control(maxit = 50)
  suppressWarnings(tryCatch({
    free <- stats::glm.fit(cbind(1, lp), fail, family = stats::binomial(), control = ctl)
    citl <- stats::glm.fit(matrix(1, length(lp), 1), fail, offset = lp, family = stats::binomial(), control = ctl)
    out[] <- c(free$coefficients[2], citl$coefficients[1], free$coefficients[1])
    out
  }, error = function(e) out))
}

# ---- patient-clustered bootstrap ---------------------------------------------------------------

boot_cores <- function() {
  env <- suppressWarnings(as.integer(Sys.getenv("OPTICS_CORES", "")))
  if (!is.na(env) && env >= 1L) return(env)
  max(1L, min(as.integer(CFG$statistics$mc_cores %||% 8L), parallel::detectCores()))
}

.gl_with_seed <- function(seed, expr) {
  had <- exists(".Random.seed", globalenv(), inherits = FALSE)
  if (had) old <- get(".Random.seed", globalenv())
  on.exit(if (had) assign(".Random.seed", old, globalenv()) else if (exists(".Random.seed", globalenv(), inherits = FALSE)) rm(".Random.seed", envir = globalenv()))
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  expr
}

# Draws B resamples of patients with replacement inside strata. Done serially up front, so the draws do not
# depend on the number of cores and one plan can be shared by several statistics.
boot_plan <- function(data, patient = "patient_id", strata = "domain", B, seed) {
  pid <- as.character(data[[patient]])
  st <- if (is.null(strata)) rep("all", length(pid)) else as.character(data[[strata]])
  rows <- split(seq_along(pid), pid)
  st_of <- vapply(split(st, pid), function(x) {
    if (length(unique(x)) > 1L) stop("a patient falls in more than one stratum")
    x[1]
  }, "")
  groups <- unname(split(seq_along(rows), st_of[names(rows)]))
  draws <- .gl_with_seed(seed, lapply(seq_len(B), function(b) {
    unlist(lapply(groups, function(g) g[sample.int(length(g), length(g), replace = TRUE)]), use.names = FALSE)
  }))
  structure(list(rows = unname(rows), patients = names(rows), draws = draws, B = B, seed = seed), class = "boot_plan")
}

.gl_boot_data <- function(data, plan, b) {
  sel <- plan$rows[plan$draws[[b]]]
  out <- data[unlist(sel, use.names = FALSE), , drop = FALSE]
  out$.boot_cluster <- rep.int(seq_along(sel), lengths(sel))
  out
}

.gl_pct_ci <- function(est, reps, level) {
  a <- (1 - level) / 2
  q <- apply(reps, 2, function(x) {
    x <- x[is.finite(x)]
    if (length(x)) stats::quantile(x, c(a, 1 - a), names = FALSE) else c(NA_real_, NA_real_)
  })
  data.frame(stat = names(est), est = unname(est), lo = q[1, ], hi = q[2, ], row.names = NULL, stringsAsFactors = FALSE)
}

cluster_boot <- function(data, patient = "patient_id", strata = "domain", B, seed, stat_fn,
                         plan = NULL, cores = boot_cores(), level = CFG$statistics$ci_level) {
  if (is.null(plan)) {
    plan <- boot_plan(data, patient, strata, B, seed)
  } else {
    rows <- split(seq_len(nrow(data)), as.character(data[[patient]]))
    if (!identical(names(rows), plan$patients) || !identical(lengths(rows, use.names = FALSE), lengths(plan$rows))) {
      stop("boot plan does not match the patients and frame counts of `data`")
    }
  }
  full <- data
  full$.boot_cluster <- match(as.character(data[[patient]]), plan$patients)
  est <- stat_fn(full)
  if (!is.numeric(est) || is.null(names(est))) stop("stat_fn must return a named numeric vector")
  one <- function(b) {
    r <- stat_fn(.gl_boot_data(data, plan, b))
    if (!identical(names(r), names(est))) stop("stat_fn returned different names for a resample")
    r
  }
  reps <- if (cores > 1L) parallel::mclapply(seq_len(plan$B), one, mc.cores = cores) else lapply(seq_len(plan$B), one)
  bad <- vapply(reps, inherits, NA, "try-error")
  if (any(bad)) stop("bootstrap replicate failed: ", as.character(reps[[which(bad)[1]]]))
  reps <- do.call(rbind, reps)
  colnames(reps) <- names(est)
  structure(list(est = est, reps = reps, ci = .gl_pct_ci(est, reps, level), plan = plan, level = level),
            class = "cboot")
}

paired_diff_ci <- function(boot, a, b) {
  stopifnot(length(a) == length(b))
  alpha <- (1 - boot$level) / 2
  rows <- lapply(seq_along(a), function(i) {
    d <- boot$reps[, a[i]] - boot$reps[, b[i]]
    d <- d[is.finite(d)]
    q <- if (length(d)) stats::quantile(d, c(alpha, 1 - alpha), names = FALSE) else c(NA_real_, NA_real_)
    data.frame(a = a[i], b = b[i], diff = unname(boot$est[a[i]] - boot$est[b[i]]), lo = q[1], hi = q[2])
  })
  do.call(rbind, rows)
}

# stat_fn evaluated within each patient (cluster, inside a bootstrap), then averaged over patients
patient_macro <- function(stat_fn, patient = "patient_id") {
  function(data) {
    g <- if (!is.null(data[[".boot_cluster"]])) data[[".boot_cluster"]] else data[[patient]]
    parts <- lapply(split(seq_len(nrow(data)), g), function(i) stat_fn(data[i, , drop = FALSE]))
    out <- rowMeans(do.call(cbind, parts), na.rm = TRUE)
    out[is.nan(out)] <- NA_real_
    out
  }
}

# ---- gate models --------------------------------------------------------------------------------

# patient-grouped fold ids, balanced on frame counts (greedy, shuffled with a fixed seed)
patient_foldid <- function(patient, K, seed = CFG$seeds$global) {
  pid <- unique(patient)
  if (length(pid) < max(K, 3L)) stop("too few patients (", length(pid), ") for ", K, " grouped folds")
  n <- tabulate(match(patient, pid))
  ord <- .gl_with_seed(seed, sample.int(length(pid)))
  ord <- ord[order(-n[ord])]
  load <- numeric(K)
  fold <- integer(length(pid))
  for (i in ord) {
    k <- which.min(load)
    fold[i] <- k
    load[k] <- load[k] + n[i]
  }
  fold[match(patient, pid)]
}

.gl_basis <- function(z, knots, boundary) {
  if (length(knots)) unclass(splines::ns(z, knots = knots, Boundary.knots = boundary)) else matrix(z, ncol = 1L)
}

# standardisation and spline knots come from the training data only
.gl_make_spec <- function(d, features, df) {
  centre <- vapply(features, function(f) mean(d[[f]]), 0)
  scl <- vapply(features, function(f) stats::sd(d[[f]]), 0)
  scl[!is.finite(scl) | scl == 0] <- 1
  probs <- seq(0, 1, length.out = df + 1L)[-c(1L, df + 1L)]
  knots <- boundary <- setNames(vector("list", length(features)), features)
  for (f in features) {
    z <- (d[[f]] - centre[[f]]) / scl[[f]]
    boundary[[f]] <- range(z)
    kn <- unique(stats::quantile(z, probs, names = FALSE))
    knots[[f]] <- kn[kn > boundary[[f]][1] & kn < boundary[[f]][2]]
  }
  list(spec = list(features = features, df = df, knots = knots, boundary = boundary),
       scale = list(centre = centre, scale = scl))
}

.gl_design <- function(spec, scale, newdata) {
  miss <- setdiff(spec$features, names(newdata))
  if (length(miss)) stop("newdata lacks feature column(s): ", paste(miss, collapse = ", "))
  cols <- lapply(spec$features, function(f) {
    x <- newdata[[f]]
    if (any(!is.finite(x))) stop("non-finite values in feature '", f, "' (", sum(!is.finite(x)), " rows)")
    b <- .gl_basis((x - scale$centre[[f]]) / scale$scale[[f]], spec$knots[[f]], spec$boundary[[f]])
    colnames(b) <- paste0(f, "_", seq_len(ncol(b)))
    b
  })
  do.call(cbind, cols)
}

.gl_logistic <- function(X, y) {
  fit <- suppressWarnings(stats::glm.fit(X, as.numeric(y), family = stats::binomial(), control = stats::glm.control(maxit = 50)))
  if (!fit$converged || anyNA(fit$coefficients)) stop("logistic fit failed (separation or collinearity)")
  setNames(fit$coefficients, colnames(X))
}

fit_gate <- function(d, fail, features = OPTICAL_METRICS, foldid, df = 3, alpha = 0.5, kind = "optical_gate") {
  sp <- .gl_make_spec(d, features, df)
  x <- .gl_design(sp$spec, sp$scale, d)
  cv <- glmnet::cv.glmnet(x, as.numeric(fail), family = "binomial", alpha = alpha, foldid = foldid,
                          type.measure = "deviance")
  list(kind = kind, spec = sp$spec, fit = cv$glmnet.fit, lambda = cv$lambda.1se, scale = sp$scale,
       alpha = alpha, created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
}

# unpenalised logistic regression on a natural spline of one feature (the predicted-area baseline)
fit_area <- function(d, fail, feature = "pred_area_frac", df = 3) {
  sp <- .gl_make_spec(d, feature, df)
  x <- cbind(1, .gl_design(sp$spec, sp$scale, d))
  colnames(x)[1] <- "(Intercept)"
  list(kind = "predicted_area", spec = sp$spec, fit = list(beta = .gl_logistic(x, fail)), lambda = NA_real_,
       scale = sp$scale, alpha = NA_real_, created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
}

# logistic on the gate's log-odds (link, OOF for the training rows) and ens_dis
fit_stack <- function(gate_link, ens_dis, fail, base) {
  x <- cbind("(Intercept)" = 1, gate_logit = gate_link, ens_dis = ens_dis)
  list(kind = "gate_plus_ensemble", spec = list(features = "ens_dis"), fit = list(beta = .gl_logistic(x, fail)),
       base = base, created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
}

predict_gate <- function(gate_obj, newdata, type = c("response", "link")) {
  type <- match.arg(type)
  if (identical(gate_obj$kind, "gate_plus_ensemble")) {
    if (!"ens_dis" %in% names(newdata)) stop("newdata lacks feature column(s): ens_dis")
    if (any(!is.finite(newdata$ens_dis))) stop("non-finite values in feature 'ens_dis'")
    b <- gate_obj$fit$beta
    link <- b[[1]] + b[[2]] * predict_gate(gate_obj$base, newdata, "link") + b[[3]] * newdata$ens_dis
  } else {
    x <- .gl_design(gate_obj$spec, gate_obj$scale, newdata)
    link <- if (inherits(gate_obj$fit, "glmnet")) {
      as.numeric(stats::predict(gate_obj$fit, newx = x, s = gate_obj$lambda, type = "link"))
    } else {
      drop(cbind(1, x) %*% gate_obj$fit$beta)
    }
  }
  link <- as.numeric(link)
  if (type == "link") link else stats::plogis(link)
}

# fold_objs: list of gate objects named by the held-out fold; scores each fold's rows with its own model
predict_oof <- function(fold_objs, newdata, fold, type = "link") {
  out <- rep(NA_real_, nrow(newdata))
  for (f in unique(fold)) {
    i <- fold == f
    out[i] <- predict_gate(fold_objs[[as.character(f)]], newdata[i, , drop = FALSE], type)
  }
  out
}

# Share of rows whose feature lies below / above the TRAINING range (where the splines extrapolate linearly);
# the last row, "any_feature", is the share of rows outside the range on at least one feature.
range_check <- function(gate_obj, newdata) {
  sp <- gate_obj$spec
  below <- above <- numeric()
  any_out <- rep(FALSE, nrow(newdata))
  for (f in sp$features) {
    z <- (newdata[[f]] - gate_obj$scale$centre[[f]]) / gate_obj$scale$scale[[f]]
    lo <- z < sp$boundary[[f]][1]
    hi <- z > sp$boundary[[f]][2]
    below[f] <- mean(lo)
    above[f] <- mean(hi)
    any_out <- any_out | lo | hi
  }
  data.frame(feature = c(sp$features, "any_feature"), frac_below = c(below, NA_real_),
             frac_above = c(above, NA_real_), frac_outside = c(below + above, mean(any_out)), row.names = NULL,
             stringsAsFactors = FALSE)
}

# ---- lock file and output guard -----------------------------------------------------------------

.gl_clean <- function(path) {
  out <- character()
  for (s in strsplit(normalizePath(path, mustWork = FALSE), "/", fixed = TRUE)[[1]]) {
    if (s == "..") out <- head(out, -1L) else if (nzchar(s) && s != ".") out <- c(out, s)
  }
  paste0("/", paste(out, collapse = "/"))
}

guard_write <- function(path) {
  if (isTRUE(SYNTH)) {
    ok <- startsWith(.gl_clean(path), paste0(.gl_clean(file.path(SYNTH_DIR, "out")), "/"))
    if (!ok) stop("synthetic mode: refusing to write outside work/synthetic/out: ", path, call. = FALSE)
  }
  invisible(path)
}

# directory (guarded and created) for an output path built with p()
out_path <- function(...) {
  path <- guard_write(p(...))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  path
}

write_csv_guarded <- function(df, path) {
  guard_write(path)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(df, path, row.names = FALSE, na = "")
  invisible(path)
}

.gl_nonempty <- function(x) is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)

# A missing lock file is an error, except in synthetic mode where it starts from the plan's defaults.
read_lock <- function(path) {
  if (file.exists(path)) return(yaml::read_yaml(path))
  if (!isTRUE(SYNTH)) stop("lock file not found: ", path, " (Gate G2 writes it)", call. = FALSE)
  list(failure_iou_threshold = CFG$endpoints$failure_iou_threshold,
       noninferiority_margin_aurc = CFG$endpoints$noninferiority_margin_aurc)
}

lock_failure_threshold <- function(lock) {
  thr <- lock$failure_iou_threshold
  if (is.numeric(thr) && length(thr) == 1L) thr else CFG$endpoints$failure_iou_threshold
}

.gl_yaml_scalar <- function(x) {
  if (is.null(x) || (length(x) == 1L && is.na(x))) return("null")
  if (is.character(x)) return(paste0("\"", gsub("\"", "\\\\\"", x), "\""))
  if (is.logical(x)) return(tolower(as.character(x)))
  format(x, digits = 15)
}

# Text-level update of top-level scalar keys: comments, order and every other key stay as they were.
lock_update <- function(path, values, init = list()) {
  guard_write(path)
  before <- if (file.exists(path)) yaml::read_yaml(path) else init
  lines <- if (file.exists(path)) readLines(path, warn = FALSE) else
    c("# analysis_lock.yaml: frozen choices and object hashes", paste0(names(init), ": ", vapply(init, .gl_yaml_scalar, "")))
  for (k in names(values)) {
    new <- paste0(k, ": ", .gl_yaml_scalar(values[[k]]))
    hit <- grep(paste0("^", k, ":"), lines)
    if (length(hit)) {
      nxt <- lines[hit[1] + 1L]
      if (!is.na(nxt) && grepl("^[ \t-]", nxt)) stop("lock key '", k, "' is a block, not a scalar")
      lines[hit[1]] <- new
    } else {
      lines <- c(lines, new)
    }
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, path)
  after <- yaml::read_yaml(path)
  kept <- setdiff(names(before), names(values))
  if (!identical(before[kept], after[kept])) stop("lock update changed keys it does not own")
  invisible(after)
}

# Canonical JSON of a parsed lock value: object keys sorted bytewise, no whitespace, numbers as %.15g, null for NULL/NA.
.gl_canon <- function(x) {
  if (is.null(x) || (is.atomic(x) && length(x) == 1L && is.na(x))) return("null")
  if (is.list(x)) {
    if (!length(x)) return(if (is.null(names(x))) "[]" else "{}")
    if (is.null(names(x))) return(paste0("[", paste(vapply(x, .gl_canon, ""), collapse = ","), "]"))
    x <- x[order(names(x), method = "radix")]
    return(paste0("{", paste0(vapply(names(x), .gl_canon, ""), ":", vapply(x, .gl_canon, ""), collapse = ","), "}"))
  }
  if (length(x) != 1L) return(paste0("[", paste(vapply(as.list(x), .gl_canon, ""), collapse = ","), "]"))
  if (is.character(x)) {
    s <- gsub("\\", "\\\\", x, fixed = TRUE)
    s <- gsub("\"", "\\\"", s, fixed = TRUE)
    s <- gsub("\n", "\\n", s, fixed = TRUE)
    s <- gsub("\t", "\\t", s, fixed = TRUE)
    s <- gsub("\r", "\\r", s, fixed = TRUE)
    return(paste0("\"", s, "\""))
  }
  if (is.logical(x)) return(if (x) "true" else "false")
  sprintf("%.15g", as.numeric(x))
}

# SHA-256 of the canonical JSON of every lock key except lock_fields_sha256 itself and locked_at (the definition in
# config/CONTRACTS.md). Covers the failure threshold, the margin, epochs, the entropy choice and the object paths/hashes.
LOCK_HASH_EXCLUDED <- c("lock_fields_sha256", "locked_at")
lock_fields_hash <- function(lock) {
  fields <- lock[setdiff(names(lock), LOCK_HASH_EXCLUDED)]
  if (!length(fields)) fields <- structure(list(), names = character())
  unname(tools::sha256sum(bytes = charToRaw(enc2utf8(.gl_canon(fields)))))
}

# lock_fields_sha256 must equal the hash of the other fields (so no frozen choice changed after 04a), and every
# <name>_sha256 key must equal the file named by <name>_path; stops on the first mismatch
verify_lock_hashes <- function(lock) {
  need <- c(paste0(rep(GATE_LOCK_OBJECTS, each = 2L), c("_path", "_sha256")), "lock_fields_sha256")
  gone <- need[!vapply(need, function(k) .gl_nonempty(lock[[k]]), NA)]
  if (length(gone)) stop("lock file is incomplete, missing: ", paste(gone, collapse = ", "), call. = FALSE)
  if (!identical(lock_fields_hash(lock), lock$lock_fields_sha256)) {
    stop("lock_fields_sha256 mismatch: a locked field (failure threshold, margin, epochs, entropy choice, object paths ",
         "or hashes) differs from what 04a recorded", call. = FALSE)
  }
  for (obj in GATE_LOCK_OBJECTS) {
    file <- p(lock[[paste0(obj, "_path")]])
    if (!file.exists(file)) stop("locked object not found: ", file, call. = FALSE)
    found <- unname(tools::sha256sum(file))
    if (!identical(found, lock[[paste0(obj, "_sha256")]])) {
      stop("SHA-256 mismatch for ", file, ": lock records ", lock[[paste0(obj, "_sha256")]], ", file is ", found,
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

load_gate_models <- function(lock) {
  list(gate = readRDS(p(lock$gate_rds_path)), area = readRDS(p(lock$gate_area_rds_path)),
       stack = readRDS(p(lock$gate_stack_rds_path)), folds = readRDS(p(lock$gate_folds_rds_path)))
}

# ---- data assembly ------------------------------------------------------------------------------

# arrow warns "Invalid metadata$r" when a selection drops columns named in the file's R metadata; harmless
.gl_collect <- function(query) {
  withCallingHandlers(dplyr::collect(query), warning = function(w) {
    if (grepl("Invalid metadata", conditionMessage(w), fixed = TRUE)) invokeRestart("muffleWarning")
  })
}

# One model's rows of ONE partition, with optics and patient domain (frames with fov_found == FALSE dropped). The Arrow filter runs before
# collect(), so rows of other partitions are never materialised in R.
read_eval_frames <- function(derived, partition, model = "unet") {
  part <- partition
  mdl <- model
  want_source <- c(center1_labeled = "oof", center2_labeled = "final")[[part]]
  ds <- arrow::open_dataset(file.path(derived, "predictions.parquet"))
  keep <- c("frame_id", "patient_id", "model", "source", "partition", "fold", "iou", "pred_area_frac",
            "entropy_fov", "entropy_roi", "ens_dis")
  pr <- .gl_collect(ds |>
    dplyr::filter(partition == !!part, model == !!mdl) |>
    dplyr::select(dplyr::all_of(keep)))
  if (!nrow(pr)) stop("no ", mdl, " rows for partition ", part)
  stopifnot(all(pr$partition == part), all(pr$source == want_source), !anyDuplicated(pr$frame_id))

  ids <- pr$frame_id
  op <- .gl_collect(arrow::open_dataset(file.path(derived, "optics.parquet")) |>
    dplyr::filter(frame_id %in% !!ids) |>
    dplyr::select(dplyr::all_of(c("frame_id", OPTICAL_METRICS))))
  fo <- utils::read.csv(file.path(derived, "folds.csv"), stringsAsFactors = FALSE)[c("patient_id", "domain")]
  out <- pr |>
    dplyr::mutate(fold = as.integer(fold)) |>
    dplyr::left_join(op, by = "frame_id") |>
    dplyr::left_join(fo, by = "patient_id")

  # frames without a detected field of view are excluded (fov.exclude_not_found) BEFORE the missing-value check, because
  # they carry NaN optics; the predictions file normally lacks them already, so this is a guard and the count is reported
  n_excluded <- 0L
  if (isTRUE(CFG$fov$exclude_not_found)) {
    fv <- .gl_collect(arrow::open_dataset(file.path(derived, "fov.parquet")) |>
      dplyr::filter(frame_id %in% !!ids) |> dplyr::select(frame_id, fov_found))
    if (!setequal(fv$frame_id, ids)) stop("fov.parquet lacks rows for some frames")
    drop <- out$frame_id %in% fv$frame_id[!fv$fov_found]
    n_excluded <- sum(drop)
    out <- out[!drop, ]
  }
  if (anyNA(out[[OPTICAL_METRICS[1]]]) || anyNA(out$domain)) stop("optics or domain missing for some frames")
  attr(out, "rows_in_file") <- nrow(ds)
  attr(out, "rows_loaded") <- nrow(pr)
  attr(out, "n_excluded_fov") <- n_excluded
  out
}

# Scores of every non-trivial method, higher = less reliable. oof = TRUE scores each row with the model
# fitted without its segmenter fold (Center 1); FALSE uses the final locked objects (external data).
score_methods <- function(models, d, entropy_choice, oof) {
  link <- function(final, name) {
    if (oof) predict_oof(models$folds[[name]], d, d$fold, "link") else predict_gate(final, d, "link")
  }
  data.frame(
    s_predicted_area = link(models$area, "predicted_area"),
    s_entropy = d[[entropy_choice]],
    s_ens_disagreement = d$ens_dis,
    s_optical_gate = link(models$gate, "optical_gate"),
    s_gate_plus_ensemble = link(models$stack, "gate_plus_ensemble")
  )
}

# ---- evaluation of the methods ------------------------------------------------------------------

# stat_fn for cluster_boot. d needs risk, fail, patient_id, s_<method> (link scale for the probabilistic ones).
make_gate_stat_fn <- function(cov_levels, sens_levels, grid) {
  sel_names <- paste0("sel_risk_", round(cov_levels * 100))
  spec_names <- paste0("spec_at_sens", round(sens_levels * 100))
  grid_names <- sprintf("cov%.2f", grid)
  function(d) {
    risk <- d$risk
    fail <- d$fail
    n <- length(risk)
    clus <- if (!is.null(d$.boot_cluster)) d$.boot_cluster else d$patient_id
    by_clus <- split(seq_len(n), clus)
    i_cov <- .gl_cov_index(cov_levels, n)
    i_grid <- .gl_cov_index(grid, n)

    scored <- function(score, prob) {
      cum <- .gl_cum_risk(score, risk)
      cal <- if (prob) calibration_fit(stats::plogis(score), fail) else c(slope = NA_real_, intercept = NA_real_, intercept_free = NA_real_)
      c(aurc = mean(cum),
        aurc_macro = mean(vapply(by_clus, function(i) aurc(score[i], risk[i]), 0)),
        auroc = auroc_fail(score, fail),
        setNames(cum[i_cov], sel_names),
        setNames(vapply(sens_levels, function(s) spec_at_sens(score, fail, s), 0), spec_names),
        cal_slope = cal[["slope"]], cal_intercept = cal[["intercept"]], cal_intercept_free = cal[["intercept_free"]],
        setNames(cum[i_grid], grid_names))
    }
    random <- c(aurc = mean(risk),
                aurc_macro = mean(vapply(by_clus, function(i) mean(risk[i]), 0)),
                auroc = 0.5,
                setNames(rep(mean(risk), length(cov_levels)), sel_names),
                setNames(1 - sens_levels, spec_names),
                cal_slope = NA_real_, cal_intercept = NA_real_, cal_intercept_free = NA_real_,
                setNames(rep(mean(risk), length(grid)), grid_names))
    parts <- c(list(random = random),
               lapply(setNames(GATE_SCORE_METHODS, GATE_SCORE_METHODS), function(m) scored(d[[paste0("s_", m)]], m %in% GATE_PROB_METHODS)),
               list(oracle = scored(risk, FALSE)))
    parts <- parts[GATE_METHODS]
    unlist(lapply(names(parts), function(m) setNames(parts[[m]], paste(m, names(parts[[m]]), sep = "|"))))
  }
}

.gl_split_stat <- function(ci) {
  sp <- do.call(rbind, strsplit(ci$stat, "|", fixed = TRUE))
  cbind(ci, method = sp[, 1], what = sp[, 2], stringsAsFactors = FALSE)
}

TABLE_STATS <- c("aurc", "aurc_macro", "auroc", "sel_risk_90", "sel_risk_80", "spec_at_sens50", "spec_at_sens90",
                 "cal_slope", "cal_intercept", "cal_intercept_free")

# d: frames with patient_id, domain, iou; scores: data frame of s_<method> columns (rows aligned with d).
# Returns the per-method table, the paired differences and the risk-coverage curve points.
evaluate_scored <- function(d, scores, partition, model, thr, B = CFG$statistics$bootstrap_resamples,
                            seed = CFG$seeds$bootstrap, cores = boot_cores(), margin = NULL,
                            cov_levels = unlist(CFG$endpoints$coverage_levels),
                            sens_levels = unlist(CFG$endpoints$sens_levels), grid = seq(0.01, 1, by = 0.01)) {
  ev <- data.frame(patient_id = d$patient_id, domain = d$domain, risk = 1 - d$iou, fail = d$iou < thr, scores)
  boot <- cluster_boot(ev, "patient_id", "domain", B, seed, make_gate_stat_fn(cov_levels, sens_levels, grid),
                       cores = cores)
  ci <- .gl_split_stat(boot$ci)

  n_events <- sum(ev$fail)
  n_pat_events <- length(unique(ev$patient_id[ev$fail]))
  tab <- data.frame(partition = partition, model = model, method = GATE_METHODS, n_frames = nrow(ev),
                    n_patients = length(unique(ev$patient_id)), n_events = n_events,
                    n_patients_with_events = n_pat_events,
                    binary_descriptive = n_events < CFG$endpoints$min_external_failure_events ||
                      n_pat_events < CFG$endpoints$min_external_patients_with_events,
                    stringsAsFactors = FALSE)
  for (s in TABLE_STATS) {
    r <- ci[ci$what == s, ]
    r <- r[match(GATE_METHODS, r$method), ]
    tab[[s]] <- r$est
    tab[[paste0(s, "_lo")]] <- r$lo
    tab[[paste0(s, "_hi")]] <- r$hi
  }
  # random is analytic for these (0.5 and 1 - sensitivity): no interval
  rnd <- tab$method == "random"
  for (s in c("auroc", "spec_at_sens50", "spec_at_sens90")) tab[rnd, paste0(s, c("_lo", "_hi"))] <- NA_real_

  diffs <- do.call(rbind, lapply(c("aurc", "auroc", "sel_risk_90"), function(met) {
    a <- paste(GATE_COMPARISONS$a, met, sep = "|")
    b <- paste(GATE_COMPARISONS$b, met, sep = "|")
    out <- paired_diff_ci(boot, a, b)
    data.frame(partition = partition, model = model, comparison = GATE_COMPARISONS$comparison,
               a = GATE_COMPARISONS$a, b = GATE_COMPARISONS$b, metric = met,
               diff = out$diff, lo = out$lo, hi = out$hi, stringsAsFactors = FALSE)
  }))
  has_margin <- is.numeric(margin) && length(margin) == 1L
  diffs$margin <- if (has_margin) margin else NA_real_
  ni <- diffs$metric == "aurc" & diffs$comparison %in% GATE_COMPARISONS$comparison[GATE_COMPARISONS$noninf]
  diffs$noninferior <- if (has_margin) ifelse(ni, diffs$hi < margin, NA) else NA

  cv <- ci[startsWith(ci$what, "cov"), ]
  curves <- data.frame(partition = partition, model = model, method = cv$method,
                       coverage = as.numeric(sub("cov", "", cv$what)), selective_risk = cv$est, lo = cv$lo, hi = cv$hi,
                       stringsAsFactors = FALSE)
  curves <- curves[order(match(curves$method, GATE_METHODS), curves$coverage), ]
  list(table = tab, diffs = diffs, curves = curves, boot = boot)
}

.gl_reg_est_ci <- function(key, est, lo, hi, digits, note = NULL) {
  if (!is.finite(est)) return(invisible(NULL))
  reg(key, est, digits = digits, note = note)
  if (is.finite(lo) && is.finite(hi)) reg(paste0(key, "_ci"), c(est, lo, hi), digits = digits, fmt = "ci", note = note)
}

# prefix: e.g. "gate_ext" or "transfer_int". Keys: <prefix>_<stat>_<method>[_ci], <prefix>_diff_aurc_<comparison>[_ci].
register_gate_results <- function(res, prefix) {
  tb <- res$table
  for (i in seq_len(nrow(tb))) {
    m <- tb$method[i]
    r <- tb[i, ]
    k <- function(stat) paste(prefix, stat, m, sep = "_")
    .gl_reg_est_ci(k("aurc"), r$aurc, r$aurc_lo, r$aurc_hi, 3)
    .gl_reg_est_ci(k("sel90"), r$sel_risk_90, r$sel_risk_90_lo, r$sel_risk_90_hi, 3)
    .gl_reg_est_ci(k("sel80"), r$sel_risk_80, r$sel_risk_80_lo, r$sel_risk_80_hi, 3)
    if (!m %in% c("random", "oracle")) .gl_reg_est_ci(k("auroc"), r$auroc, r$auroc_lo, r$auroc_hi, 2)
    if (m %in% GATE_PROB_METHODS) {
      .gl_reg_est_ci(k("calslope"), r$cal_slope, r$cal_slope_lo, r$cal_slope_hi, 2)
      .gl_reg_est_ci(k("calint"), r$cal_intercept, r$cal_intercept_lo, r$cal_intercept_hi, 2)
    }
    if (m == "optical_gate") {
      .gl_reg_est_ci(k("spec_sens50"), r$spec_at_sens50, r$spec_at_sens50_lo, r$spec_at_sens50_hi, 2)
      .gl_reg_est_ci(k("spec_sens90"), r$spec_at_sens90, r$spec_at_sens90_lo, r$spec_at_sens90_hi, 2)
    }
  }
  df <- res$diffs[res$diffs$metric == "aurc", ]
  for (i in seq_len(nrow(df))) {
    .gl_reg_est_ci(paste(prefix, "diff_aurc", df$comparison[i], sep = "_"), df$diff[i], df$lo[i], df$hi[i], 3,
                   note = "negative favours the first method")
  }
  invisible(NULL)
}

# Counts of a partition (from an evaluate_scored result) as registry keys: <prefix>_n_frames etc.
register_counts <- function(res, prefix) {
  t1 <- res$table[1, ]
  reg(paste0(prefix, "_n_frames"), as.integer(t1$n_frames))
  reg(paste0(prefix, "_n_patients"), as.integer(t1$n_patients))
  reg(paste0(prefix, "_failure_events"), as.integer(t1$n_events))
  reg(paste0(prefix, "_failure_patients"), as.integer(t1$n_patients_with_events))
  reg(paste0(prefix, "_failure_rate"), t1$n_events / t1$n_frames, fmt = "percent")
  reg(paste0(prefix, "_binary_descriptive"), as.integer(t1$binary_descriptive),
      note = "1 = too few failures or patients with failures: binary metrics are descriptive only")
  invisible(NULL)
}

# ---- paths and the locked evaluation shared by 04b and 05 -----------------------------------------

# derived, lock, stats, results are absolute; models_rel is relative to v1 and is what the lock stores (resolved with p())
default_paths <- function() {
  list(derived = p("derived"), models_rel = file.path("work", "models"), lock = p("config", "analysis_lock.yaml"),
       stats = p("results", "stats"), results = if (exists("results_root", mode = "function")) results_root() else V1)
}

# Hash check first, then Center 1 (OOF scores) and only then Center 2 rows are read.
run_locked_evaluation <- function(paths, model, B = CFG$statistics$bootstrap_resamples, cores = boot_cores()) {
  lock <- read_lock(paths$lock)
  verify_lock_hashes(lock)
  choice <- lock$entropy_baseline_choice
  if (!.gl_nonempty(choice)) stop("lock file has no entropy_baseline_choice", call. = FALSE)
  models <- load_gate_models(lock)
  thr <- lock_failure_threshold(lock)
  margin <- lock$noninferiority_margin_aurc
  out <- list(lock = lock, threshold = thr)
  for (name in c("internal", "external")) {
    part <- c(internal = "center1_labeled", external = "center2_labeled")[[name]]
    d <- read_eval_frames(paths$derived, part, model)
    sc <- score_methods(models, d, choice, oof = name == "internal")
    out[[name]] <- evaluate_scored(d, sc, name, model, thr, B, cores = cores, margin = margin)
    out[[name]]$rows_loaded <- attr(d, "rows_loaded")
    out[[name]]$n_excluded_fov <- attr(d, "n_excluded_fov")
    out[[name]]$range <- cbind(partition = name, range_check(models$gate, d))
  }
  out
}
