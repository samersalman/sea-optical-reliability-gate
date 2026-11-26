# Revision 2 post hoc analyses (Reviewer 4 and Reviewer 5), Tier A: stored per-frame tables only.
# Nothing here changes the locked gate, cutoffs or threshold. Every output is post hoc and exploratory.
#   Rscript r2_posthoc.R
# Reads  v1/derived/*.parquet, v1/work/models/gate_*.rds (hash-verified against the lock)
# Writes REVISION 2/analysis/out/*.csv

HERE <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)))
V1_DIR <- Sys.getenv("OPTICS_V1", unset = normalizePath(file.path(HERE, "..", "v1"), mustWork = FALSE))   # repository layout by default
OUT <- file.path(HERE, "out")
Sys.setenv(OPTICS_V1 = V1_DIR)
stopifnot(file.exists(file.path(V1_DIR, "config", "analysis_plan.yaml")))
for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R")) source(file.path(V1_DIR, "R", f))
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
wcsv <- function(df, name) utils::write.csv(df, file.path(OUT, paste0(name, ".csv")), row.names = FALSE)

B <- CFG$statistics$bootstrap_resamples
SEED <- CFG$seeds$bootstrap
CORES <- boot_cores()
LEVEL <- CFG$statistics$ci_level

lock <- read_lock(p("config", "analysis_lock.yaml"))
verify_lock_hashes(lock)
models <- load_gate_models(lock)
choice <- lock$entropy_baseline_choice
THR <- lock_failure_threshold(lock)
RETAINED <- c("contrast_local", "sharp_tenengrad", "lum_entropy")

man <- as.data.frame(arrow::read_parquet(p("derived", "manifest.parquet"),
  col_select = c("frame_id", "channel", "region", "scenario", "instrument", "file_format")))
add_cov <- function(d) {
  d <- as.data.frame(d)
  cbind(d, man[match(d$frame_id, man$frame_id), c("channel", "region", "scenario", "instrument", "file_format")])
}
dev <- add_cov(read_eval_frames(p("derived"), "center1_labeled", "unet"))
ext <- add_cov(read_eval_frames(p("derived"), "center2_labeled", "unet"))
for (nm in c("dev", "ext")) {
  d <- get(nm)
  d$risk <- 1 - d$iou
  d$s_gate <- if (nm == "dev") predict_oof(models$folds$optical_gate, d, d$fold, "link") else predict_gate(models$gate, d, "link")
  d$s_entropy <- d[[choice]]
  d$s_ens <- d$ens_dis
  d$s_area <- if (nm == "dev") predict_oof(models$folds$predicted_area, d, d$fold, "link") else predict_gate(models$area, d, "link")
  d$fail <- as.integer(d$iou < THR)
  assign(nm, d)
}
stopifnot(nrow(dev) == 9590, nrow(ext) == 1069, sum(dev$fail) == 1293, sum(ext$fail) == 109)
cat("loaded dev", nrow(dev), "ext", nrow(ext), "\n")

ci_tab <- function(bt, extra = list()) {
  out <- bt$ci
  for (k in names(extra)) out[[k]] <- extra[[k]]
  out
}
avg_prec <- function(score, fail) {
  if (!any(fail == 1)) return(NA_real_)
  o <- order(-score)
  f <- fail[o]
  prec <- cumsum(f) / seq_along(f)
  sum(prec[f == 1]) / sum(f)
}
METHODS <- c(gate = "s_gate", entropy = "s_entropy", ens = "s_ens", area = "s_area")

# ---- 1 threshold sensitivity --------------------------------------------------------------------------------
THRS <- c(0.75, 0.80, 0.85, 0.90, 0.95)
thr_stat <- function(d) {
  v <- c()
  for (t in THRS) {
    fl <- as.integer(d$iou < t)
    tag <- sprintf("t%.2f", t)
    v[paste(tag, "events", sep = "|")] <- sum(fl)
    v[paste(tag, "prev", sep = "|")] <- mean(fl)
    for (m in names(METHODS)) {
      v[paste(tag, m, "auroc", sep = "|")] <- auroc_fail(d[[METHODS[[m]]]], fl)
      v[paste(tag, m, "auprc", sep = "|")] <- avg_prec(d[[METHODS[[m]]]], fl)
    }
    v[paste(tag, "gate_minus_entropy", "auroc", sep = "|")] <- v[paste(tag, "gate", "auroc", sep = "|")] - v[paste(tag, "entropy", "auroc", sep = "|")]
  }
  v
}
thr_rows <- list()
for (nm in c("dev", "ext")) {
  d <- get(nm)
  bt <- cluster_boot(d, B = B, seed = SEED, stat_fn = thr_stat, cores = CORES)
  ci <- bt$ci
  ci$partition <- nm
  ci$patients_with_event <- NA
  for (t in THRS) {
    fl <- d$iou < t
    ci$patients_with_event[ci$stat == sprintf("t%.2f|events", t)] <- length(unique(d$patient_id[fl]))
  }
  thr_rows[[nm]] <- ci
}
wcsv(do.call(rbind, thr_rows), "s1_threshold_sensitivity")
cat("1 threshold sensitivity done\n")

# ---- 2 fixed development-derived cutoffs, missed failures -----------------------------------------------------
cut_of <- function(score, cov) stats::quantile(score, cov, names = FALSE, type = 7)
cuts <- do.call(rbind, lapply(names(METHODS), function(m) {
  data.frame(method = m, coverage = c(0.90, 0.80),
             cutoff = c(cut_of(dev[[METHODS[[m]]]], 0.90), cut_of(dev[[METHODS[[m]]]], 0.80)))
}))
gt <- models$gate$flag_thresholds
if (!is.null(gt)) {
  for (cv in c(0.90, 0.80)) {
    k <- which(abs(as.numeric(names(gt)) - cv) < 1e-9 | names(gt) == sprintf("%.2f", cv) | names(gt) == as.character(cv))
    if (length(k)) cuts$cutoff[cuts$method == "gate" & cuts$coverage == cv] <- unname(gt[k[1]])
  }
}
cut_stat <- function(d) {
  v <- c()
  for (i in seq_len(nrow(cuts))) {
    s <- d[[METHODS[[cuts$method[i]]]]]
    fl <- s > cuts$cutoff[i]
    tag <- paste(cuts$method[i], sprintf("cov%.0f", 100 * cuts$coverage[i]), sep = "|")
    v[paste(tag, "flagged_share", sep = "|")] <- mean(fl)
    v[paste(tag, "sensitivity", sep = "|")] <- if (any(d$fail == 1)) mean(fl[d$fail == 1]) else NA
    v[paste(tag, "specificity", sep = "|")] <- mean(!fl[d$fail == 0])
    v[paste(tag, "ppv", sep = "|")] <- if (any(fl)) mean(d$fail[fl]) else NA
    v[paste(tag, "missed_failures", sep = "|")] <- sum(!fl & d$fail == 1)
    v[paste(tag, "retained_risk", sep = "|")] <- if (any(!fl)) mean(d$risk[!fl]) else NA
  }
  v
}
cut_rows <- list()
for (nm in c("dev", "ext")) {
  bt <- cluster_boot(get(nm), B = B, seed = SEED, stat_fn = cut_stat, cores = CORES)
  ci <- bt$ci
  ci$partition <- nm
  cut_rows[[nm]] <- ci
}
wcsv(cuts, "s2_cutoffs_used")
wcsv(do.call(rbind, cut_rows), "s2_fixed_cutoffs")
cat("2 fixed cutoffs done\n")

# ---- 3 calibration by fifths of predicted risk (gate, external) -------------------------------------------------
cal_rows <- list()
for (nm in c("dev", "ext")) {
  d <- get(nm)
  pr <- plogis(d$s_gate)
  br <- stats::quantile(pr, seq(0, 1, 0.2))
  g <- cut(pr, unique(br), include.lowest = TRUE, labels = FALSE)
  cal_rows[[nm]] <- do.call(rbind, lapply(sort(unique(g)), function(k) {
    i <- g == k
    data.frame(partition = nm, fifth = k, n = sum(i), mean_predicted = mean(pr[i]), observed = mean(d$fail[i]),
               mean_risk = mean(d$risk[i]))
  }))
}
wcsv(do.call(rbind, cal_rows), "s3_calibration_fifths")

# ---- 4 patient-balanced (patient-macro) estimates and design effect --------------------------------------------
macro_core <- function(d) {
  c(gate_aurc = aurc(d$s_gate, d$risk), entropy_aurc = aurc(d$s_entropy, d$risk), ens_aurc = aurc(d$s_ens, d$risk),
    random_aurc = mean(d$risk),
    gate_auroc = auroc_fail(d$s_gate, d$fail), entropy_auroc = auroc_fail(d$s_entropy, d$fail),
    ens_auroc = auroc_fail(d$s_ens, d$fail))
}
macro_fn <- function(d) {
  m <- patient_macro(macro_core)(d)
  c(m, gate_minus_random_aurc = m[["gate_aurc"]] - m[["random_aurc"]],
    gate_minus_entropy_aurc = m[["gate_aurc"]] - m[["entropy_aurc"]])
}
mac_rows <- list()
for (nm in c("dev", "ext")) {
  bt <- cluster_boot(get(nm), B = B, seed = SEED, stat_fn = macro_fn, cores = CORES)
  ci <- bt$ci
  ci$partition <- nm
  mac_rows[[nm]] <- ci
}
wcsv(do.call(rbind, mac_rows), "s4_patient_macro")

icc_anova <- function(y, g) {
  g <- factor(g)
  k <- nlevels(g)
  n <- length(y)
  ni <- tabulate(g)
  n0 <- (n - sum(ni^2) / n) / (k - 1)
  a <- stats::anova(stats::lm(y ~ g))
  msb <- a$`Mean Sq`[1]
  msw <- a$`Mean Sq`[2]
  max(0, (msb - msw) / (msb + (n0 - 1) * msw))
}
de_rows <- do.call(rbind, lapply(c("dev", "ext"), function(nm) {
  d <- get(nm)
  ni <- tabulate(factor(d$patient_id))
  mbar <- sum(ni^2) / sum(ni)
  icc_f <- icc_anova(d$fail, d$patient_id)
  icc_r <- icc_anova(d$risk, d$patient_id)
  de_f <- 1 + (mbar - 1) * icc_f
  data.frame(partition = nm, frames = nrow(d), patients = length(ni), events = sum(d$fail),
             patients_with_event = length(unique(d$patient_id[d$fail == 1])),
             mean_frames_per_patient = mean(ni), size_weighted_cluster = mbar,
             icc_failure = icc_f, icc_risk = icc_r, design_effect_failure = de_f,
             effective_frames = nrow(d) / de_f, effective_events = sum(d$fail) / de_f)
}))
wcsv(de_rows, "s4_design_effect")
cat("3-4 calibration, macro, design effect done\n")

# ---- 5 extrapolation and shift of every metric, retained flagged -------------------------------------------------
sh <- do.call(rbind, lapply(OPTICAL_METRICS, function(m) {
  x <- dev[[m]]; y <- ext[[m]]
  rng <- range(x)
  data.frame(metric = m, retained_by_locked_gate = m %in% RETAINED,
             ext_below_dev_min = mean(y < rng[1]), ext_above_dev_max = mean(y > rng[2]),
             ext_outside = mean(y < rng[1] | y > rng[2]),
             dev_median = stats::median(x), ext_median = stats::median(y),
             smd = (mean(y) - mean(x)) / stats::sd(x),
             ext_median_percentile_in_dev = mean(x <= stats::median(y)),
             ext_outside_central95_dev = mean(y < stats::quantile(x, 0.025) | y > stats::quantile(x, 0.975)))
}))
sh <- rbind(sh, data.frame(metric = "retained_any", retained_by_locked_gate = TRUE, ext_below_dev_min = NA,
  ext_above_dev_max = NA,
  ext_outside = mean(Reduce(`|`, lapply(RETAINED, function(m) ext[[m]] < min(dev[[m]]) | ext[[m]] > max(dev[[m]])))),
  dev_median = NA, ext_median = NA, smd = NA, ext_median_percentile_in_dev = NA,
  ext_outside_central95_dev = mean(Reduce(`|`, lapply(RETAINED, function(m) ext[[m]] < stats::quantile(dev[[m]], 0.025) | ext[[m]] > stats::quantile(dev[[m]], 0.975)))))
)
wcsv(sh, "s5_shift_extrapolation")

# ---- 6 domain-specific associations, interactions and score distributions ----------------------------------------
alld <- rbind(dev[, c("frame_id", "patient_id", "domain", "fail", "risk", "s_gate", "s_entropy", "s_ens", "scenario", OPTICAL_METRICS)],
              ext[, c("frame_id", "patient_id", "domain", "fail", "risk", "s_gate", "s_entropy", "s_ens", "scenario", OPTICAL_METRICS)])
dom_stat <- function(d) {
  v <- c(failure_rate = mean(d$fail), gate_auroc = auroc_fail(d$s_gate, d$fail),
         gate_aurc_minus_random = aurc(d$s_gate, d$risk) - mean(d$risk))
  for (m in OPTICAL_METRICS) v[paste0(m, "_auroc_low")] <- auroc_fail(-d[[m]], d$fail)
  v
}
dom_rows <- list()
for (dm in c("D1", "D2", "D3")) {
  d <- alld[alld$domain == dm, ]
  bt <- cluster_boot(d, B = B, seed = SEED, stat_fn = dom_stat, cores = CORES)
  ci <- bt$ci
  ci$domain <- dm
  dom_rows[[dm]] <- ci
}
wcsv(do.call(rbind, dom_rows), "s6_domain_associations")

# metric x center interaction on failure, OR per dev-SD, cluster-robust by patient
alld$external <- as.integer(alld$domain == "D3")
int_rows <- do.call(rbind, lapply(OPTICAL_METRICS, function(m) {
  x <- alld[[m]]
  if (m == "sharp_tenengrad") x <- log(x)
  xd <- x[alld$external == 0]
  alld$z <- (x - mean(xd)) / stats::sd(xd)
  fit <- stats::glm(fail ~ z * external, data = alld, family = stats::binomial())
  V <- sandwich::vcovCL(fit, cluster = alld$patient_id)
  b <- stats::coef(fit)
  se_int <- sqrt(V["z:external", "z:external"])
  g <- c(0, 1, 0, 1)
  ext_lin <- b[["z"]] + b[["z:external"]]
  se_ext <- sqrt(V["z", "z"] + V["z:external", "z:external"] + 2 * V["z", "z:external"])
  data.frame(metric = m, retained = m %in% RETAINED,
             or_dev = exp(b[["z"]]), or_dev_lo = exp(b[["z"]] - 1.96 * sqrt(V["z", "z"])), or_dev_hi = exp(b[["z"]] + 1.96 * sqrt(V["z", "z"])),
             or_ext = exp(ext_lin), or_ext_lo = exp(ext_lin - 1.96 * se_ext), or_ext_hi = exp(ext_lin + 1.96 * se_ext),
             ratio_of_or = exp(b[["z:external"]]), p_interaction = 2 * stats::pnorm(-abs(b[["z:external"]] / se_int)))
}))
wcsv(int_rows, "s6_interactions")

score_dist <- do.call(rbind, lapply(split(alld, list(alld$domain, alld$fail)), function(d) {
  if (!nrow(d)) return(NULL)
  q <- stats::quantile(plogis(d$s_gate), c(0.25, 0.5, 0.75))
  data.frame(domain = d$domain[1], failure = d$fail[1], n = nrow(d), p25 = q[1], median = q[2], p75 = q[3])
}))
wcsv(score_dist, "s6_score_distributions")

# pairwise interaction between retained metrics in the development data (does a product term matter?)
pw <- do.call(rbind, combn(RETAINED, 2, simplify = FALSE, FUN = function(pr) {
  z <- function(v) { v <- if (identical(v, "sharp_tenengrad")) log(dev[[v]]) else dev[[v]]; (v - mean(v)) / stats::sd(v) }
  a <- z(pr[1]); b2 <- z(pr[2])
  fit <- stats::glm(dev$fail ~ a * b2, family = stats::binomial())
  V <- sandwich::vcovCL(fit, cluster = dev$patient_id)
  co <- stats::coef(fit)[["a:b2"]]
  data.frame(pair = paste(pr, collapse = " x "), interaction_or = exp(co),
             lo = exp(co - 1.96 * sqrt(V["a:b2", "a:b2"])), hi = exp(co + 1.96 * sqrt(V["a:b2", "a:b2"])),
             p = 2 * stats::pnorm(-abs(co / sqrt(V["a:b2", "a:b2"]))))
}))
wcsv(pw, "s6_pairwise_interactions_dev")
cat("5-6 shift, domain, interactions done\n")

# ---- 7 matched subgroups and scene strata ------------------------------------------------------------------------
eval_fn <- function(d) c(aurc_gate = aurc(d$s_gate, d$risk), aurc_random = mean(d$risk),
                         diff = aurc(d$s_gate, d$risk) - mean(d$risk), auroc = auroc_fail(d$s_gate, d$fail),
                         events = sum(d$fail))
refit_score <- function(train, test, features = OPTICAL_METRICS, df = 3, tf = NULL) {
  if (!is.null(tf)) { train <- tf(train, "train", train); test <- tf(test, "test", train) }
  fid <- patient_foldid(train$patient_id, CFG$segmenters$cv_folds)
  g <- fit_gate(train, train$fail, features, fid, df = df)
  list(score = predict_gate(g, test, "link"), gate = g)
}
nz_terms <- function(g) {
  co <- as.matrix(stats::coef(g$fit, s = g$lambda))
  nm <- rownames(co)[co[, 1] != 0 & rownames(co) != "(Intercept)"]
  paste(unique(sub("_[0-9]+$", "", nm)), collapse = " ")
}
match_rows <- list()
add_eval <- function(label, d, status, nz = NA) {
  bt <- cluster_boot(d, B = B, seed = SEED, stat_fn = eval_fn, cores = CORES)
  ci <- bt$ci
  ci$analysis <- label
  ci$n_frames <- nrow(d)
  ci$n_patients <- length(unique(d$patient_id))
  ci$status <- status
  ci$nonzero_metrics <- nz
  match_rows[[length(match_rows) + 1]] <<- ci
}
mdev <- dev[dev$channel == "small" & dev$file_format == "png", ]
add_eval("Locked gate OOF, training-center frames matched to external acquisition (small channel, PNG)", mdev, "exploratory")
rf <- refit_score(mdev, ext)
e2 <- ext; e2$s_gate <- rf$score
add_eval("Gate refitted on matched training-center frames (small channel, PNG), scored at external center", e2, "exploratory", nz_terms(rf$gate))
mdev2 <- mdev[mdev$region == "lumbar", ]
rf2 <- refit_score(mdev2, ext[ext$region == "lumbar", ])
e3 <- ext[ext$region == "lumbar", ]; e3$s_gate <- rf2$score
add_eval("Gate refitted on matched lumbar small-channel PNG training frames, scored on external lumbar frames", e3, "exploratory", nz_terms(rf2$gate))
for (sc in c("Normal", "Difficult")) {
  add_eval(paste("Locked gate, external center,", sc, "frames"), ext[ext$scenario == sc, ], "exploratory")
  add_eval(paste("Locked gate OOF, training center,", sc, "frames"), dev[dev$scenario == sc, ], "exploratory")
}
wcsv(do.call(rbind, match_rows), "s7_matched_subgroups")
cat("7 matched subgroups done\n")

# ---- 8 ablation, functional forms, scaling, normalization ----------------------------------------------------------
abl_rows <- list()
add_var <- function(label, family, score, nz) {
  e <- ext; e$s_gate <- score
  bt <- cluster_boot(e, B = B, seed = SEED, stat_fn = eval_fn, cores = CORES)
  ci <- bt$ci
  ci$variant <- label
  ci$family <- family
  ci$nonzero_metrics <- nz
  abl_rows[[length(abl_rows) + 1]] <<- ci
}
add_var("Locked gate (reference)", "reference", ext$s_gate, nz_terms(models$gate))
for (m in OPTICAL_METRICS) {
  r <- refit_score(dev, ext, setdiff(OPTICAL_METRICS, m))
  add_var(paste("Drop", m), "leave-one-metric-out", r$score, nz_terms(r$gate))
}
for (m in OPTICAL_METRICS) {
  e <- ext; dd <- dev
  r <- refit_score(dd, e, m)
  sc <- r$score
  if (stats::sd(sc) < 1e-12) {
    # penalty removed every term: refit unpenalised spline logistic for a single-metric reading
    a <- fit_area(dd, dd$fail, feature = m)
    sc <- predict_gate(a, e, "link")
  }
  add_var(paste("Single metric", m), "single-metric (unpenalised if the penalty removed it)", sc, m)
}
r <- refit_score(dev, ext, RETAINED)
add_var("Retained three metrics only, refitted", "subset", r$score, nz_terms(r$gate))
r <- refit_score(dev, ext, df = 1)
add_var("Linear terms (no splines)", "functional form", r$score, nz_terms(r$gate))
clampf <- function(d, role, ref) { for (m in OPTICAL_METRICS) d[[m]] <- pmin(pmax(d[[m]], min(ref[[m]])), max(ref[[m]])); d }
e <- clampf(ext, "test", dev)
add_var("Locked gate, external inputs clamped to development range (constant extrapolation)", "functional form", predict_gate(models$gate, e, "link"), nz_terms(models$gate))
robustf <- function(d, role, ref) {
  for (m in OPTICAL_METRICS) {
    lo <- stats::quantile(ref[[m]], 0.01); hi <- stats::quantile(ref[[m]], 0.99)
    x <- pmin(pmax(d[[m]], lo), hi)
    d[[m]] <- (x - stats::median(ref[[m]])) / stats::IQR(ref[[m]])
  }
  d
}
r <- refit_score(dev, ext, tf = robustf)
add_var("Robust scaling (winsorized at development 1st/99th percentile, median/IQR)", "scaling", r$score, nz_terms(r$gate))
rankf <- function(d, role, ref) { for (m in OPTICAL_METRICS) d[[m]] <- stats::ecdf(ref[[m]])(d[[m]]); d }
r <- refit_score(dev, ext, tf = rankf)
add_var("Rank transform to development distribution", "scaling", r$score, nz_terms(r$gate))
domrank <- function(d) {
  for (m in OPTICAL_METRICS) d[[m]] <- stats::ave(d[[m]], d$domain, FUN = function(v) rank(v) / length(v))
  d
}
r <- refit_score(domrank(dev), domrank(ext))
add_var("Within-domain rank normalization (external ranks from external optics, label-free)", "domain normalization", r$score, nz_terms(r$gate))
wcsv(do.call(rbind, abl_rows), "s8_ablation_variants")
cat("8 ablation done\n")

# ---- 9 perturbation: failure rates and locked-gate behaviour by severity -----------------------------------------
pt <- as.data.frame(arrow::read_parquet(p("derived", "perturb.parquet")))
pt$fail <- as.integer(pt$iou < THR)
pt$gate_p <- plogis(predict_gate(models$gate, pt, "link"))
pr_rows <- do.call(rbind, lapply(split(pt, list(pt$model, pt$family, pt$severity), drop = TRUE), function(d) {
  data.frame(model = d$model[1], family = d$family[1], severity = d$severity[1], param = d$param[1], n = nrow(d),
             failure_rate = mean(d$fail), median_iou = stats::median(d$iou), median_gate_risk = stats::median(d$gate_p),
             gate_auroc = auroc_fail(d$gate_p, d$fail), entropy_auroc = auroc_fail(d$entropy_fov, d$fail))
}))
pr_rows <- pr_rows[order(pr_rows$model, pr_rows$family, pr_rows$severity), ]
wcsv(pr_rows, "s9_perturbation_severity")
pf_stat <- function(d) c(gate_auroc = auroc_fail(d$gate_p, d$fail), entropy_auroc = auroc_fail(d$entropy_fov, d$fail),
                         failure_rate = mean(d$fail))
pf_rows <- list()
for (fam in unique(pt$family)) {
  d <- pt[pt$model == "unet" & pt$family == fam, ]
  bt <- cluster_boot(d, B = B, seed = SEED, stat_fn = pf_stat, cores = CORES)
  ci <- bt$ci
  ci$family <- fam
  pf_rows[[fam]] <- ci
}
d <- pt[pt$model == "unet", ]
bt <- cluster_boot(d, B = B, seed = SEED, stat_fn = pf_stat, cores = CORES)
ci <- bt$ci; ci$family <- "all"
pf_rows[["all"]] <- ci
wcsv(do.call(rbind, pf_rows), "s9_perturbation_gate_by_family")
cat("9 perturbation done\n")

# ---- 10 locked gate specification -----------------------------------------------------------------------------
g <- models$gate
co <- as.matrix(stats::coef(g$fit, s = g$lambda))
coef_df <- data.frame(term = rownames(co), coefficient = co[, 1], row.names = NULL)
wcsv(coef_df, "s10_gate_coefficients")
spec_df <- do.call(rbind, lapply(g$spec$features, function(f) {
  data.frame(metric = f, centre = g$scale$centre[[f]], scale = g$scale$scale[[f]],
             knot1_z = g$spec$knots[[f]][1], knot2_z = g$spec$knots[[f]][2],
             boundary_lo_z = g$spec$boundary[[f]][1], boundary_hi_z = g$spec$boundary[[f]][2])
}))
wcsv(spec_df, "s10_gate_spline_spec")
wcsv(data.frame(item = c("alpha", "lambda_1se", "spline_df", "inner_folds", "seed_global", "seed_folds", "seed_train",
                         "seed_bootstrap", "seed_perturb", "failure_threshold", "cutoff_90", "cutoff_80"),
                value = c(g$alpha, g$lambda, g$spec$df, CFG$segmenters$cv_folds, CFG$seeds$global, CFG$seeds$folds,
                          CFG$seeds$train, CFG$seeds$bootstrap, CFG$seeds$perturb, THR,
                          cuts$cutoff[cuts$method == "gate" & cuts$coverage == 0.9],
                          cuts$cutoff[cuts$method == "gate" & cuts$coverage == 0.8])), "s10_gate_settings")
cat("all done\n")
