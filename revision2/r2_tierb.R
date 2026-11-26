# Revision 2, Tier B statistics on the pixel reruns written by tierb/tierb.py (post hoc, exploratory).
# Nothing here changes the locked gate, cutoffs or threshold.
#   Rscript r2_tierb.R
# Reads  v1/derived/*.parquet, v1/work/models/gate_*.rds (hash-verified), out/tierb_*.parquet, out/tierb_latency.json
# Writes out/t1_*.csv to out/t4_*.csv

HERE <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]), fixed = TRUE)))
V1_DIR <- Sys.getenv("OPTICS_V1", unset = normalizePath(file.path(HERE, "..", "v1"), mustWork = FALSE))   # repository layout by default
OUT <- file.path(HERE, "out")
Sys.setenv(OPTICS_V1 = V1_DIR)
stopifnot(file.exists(file.path(V1_DIR, "config", "analysis_plan.yaml")))
for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R")) source(file.path(V1_DIR, "R", f))
wcsv <- function(df, name) utils::write.csv(df, file.path(OUT, paste0(name, ".csv")), row.names = FALSE)
rpq <- function(name) as.data.frame(arrow::read_parquet(file.path(OUT, paste0("tierb_", name, ".parquet"))))

B <- CFG$statistics$bootstrap_resamples
SEED <- CFG$seeds$bootstrap
CORES <- boot_cores()
lock <- read_lock(p("config", "analysis_lock.yaml"))
verify_lock_hashes(lock)
models <- load_gate_models(lock)
choice <- lock$entropy_baseline_choice
THR <- lock_failure_threshold(lock)
RETAINED <- c("contrast_local", "sharp_tenengrad", "lum_entropy")

prep <- function(d, nm) {
  d$risk <- 1 - d$iou
  d$s_gate <- if (nm == "dev") predict_oof(models$folds$optical_gate, d, d$fold, "link") else predict_gate(models$gate, d, "link")
  d$s_entropy <- d[[choice]]
  d$s_ens <- d$ens_dis
  d$fail <- as.integer(d$iou < THR)
  d
}
dev <- prep(as.data.frame(read_eval_frames(p("derived"), "center1_labeled", "unet")), "dev")
ext <- prep(as.data.frame(read_eval_frames(p("derived"), "center2_labeled", "unet")), "ext")
stopifnot(nrow(dev) == 9590, nrow(ext) == 1069, sum(dev$fail) == 1293, sum(ext$fail) == 109)
METHODS <- c(gate = "s_gate", entropy = "s_entropy", ens = "s_ens")

# ---- T1 boundary-sensitive error measures ---------------------------------------------------------------------------
bd <- rpq("boundary")
bd <- bd[bd$model == "unet", c("frame_id", "iou", "boundary_iou", "hd95_px", "side_px")]
names(bd)[names(bd) == "iou"] <- "iou_rerun"
for (nm in c("dev", "ext")) {
  d <- merge(get(nm), bd, by = "frame_id")
  stopifnot(nrow(d) == nrow(get(nm)), max(abs(d$iou - d$iou_rerun)) < 1e-9)
  d$risk_biou <- 1 - d$boundary_iou
  d$hd95_frac <- pmin(ifelse(is.na(d$hd95_px), 1, d$hd95_px / d$side_px), 1)   # empty mask = whole field of view
  assign(nm, d)
}
BIOU_CUT <- stats::quantile(dev$boundary_iou, mean(dev$fail), names = FALSE)     # same event rate as IoU < 0.90 in development
HD_CUT <- 0.05                                                                    # contour error above 5% of the field-of-view diameter
t1_stat <- function(d) {
  fb <- as.integer(d$boundary_iou < BIOU_CUT)
  fh <- as.integer(d$hd95_frac > HD_CUT)
  v <- c(biou_events = sum(fb), hd_events = sum(fh), iou_ok_but_hd_fail = sum(d$fail == 0 & fh == 1),
         rho_iou_biou = stats::cor(d$iou, d$boundary_iou, method = "spearman"))
  for (m in names(METHODS)) {
    s <- d[[METHODS[[m]]]]
    v[paste0(m, "_aurc_biou")] <- aurc(s, d$risk_biou)
    v[paste0(m, "_aurc_hd")] <- aurc(s, d$hd95_frac)
    v[paste0(m, "_auroc_biou")] <- auroc_fail(s, fb)
    v[paste0(m, "_auroc_hd")] <- auroc_fail(s, fh)
  }
  v["random_aurc_biou"] <- mean(d$risk_biou)
  v["random_aurc_hd"] <- mean(d$hd95_frac)
  v["gate_minus_random_aurc_biou"] <- v[["gate_aurc_biou"]] - v[["random_aurc_biou"]]
  v["gate_minus_random_aurc_hd"] <- v[["gate_aurc_hd"]] - v[["random_aurc_hd"]]
  v
}
t1 <- list()
for (nm in c("dev", "ext")) {
  bt <- cluster_boot(get(nm), B = B, seed = SEED, stat_fn = t1_stat, cores = CORES)
  ci <- bt$ci
  ci$partition <- nm
  t1[[nm]] <- ci
}
t1 <- do.call(rbind, t1)
t1$biou_cut <- BIOU_CUT
t1$hd_cut_frac_of_fov <- HD_CUT
wcsv(t1, "t1_boundary")
cat("T1 boundary done; Boundary IoU cut", round(BIOU_CUT, 3), "\n")

# ---- T2 resolution harmonization ------------------------------------------------------------------------------------
hm <- rpq("harmonize")
hm <- hm[hm$model == "unet", ]
devh <- dev[, c("frame_id", "patient_id", "domain", "fold")]
devh <- merge(devh, hm[, c("frame_id", "iou", "entropy_fov", "entropy_roi", "ens_dis", "scale", OPTICAL_METRICS)], by = "frame_id")
stopifnot(nrow(devh) == nrow(dev))
devh <- prep(devh, "dev")

t2_eval <- function(d) c(aurc_gate = aurc(d$s_gate, d$risk), aurc_random = mean(d$risk),
                         diff = aurc(d$s_gate, d$risk) - mean(d$risk), auroc_gate = auroc_fail(d$s_gate, d$fail),
                         auroc_entropy = auroc_fail(d$s_entropy, d$fail), failure_rate = mean(d$fail), events = sum(d$fail))
t2 <- list()
add2 <- function(label, d) {
  bt <- cluster_boot(d, B = B, seed = SEED, stat_fn = t2_eval, cores = CORES)
  ci <- bt$ci
  ci$analysis <- label
  t2[[label]] <<- ci
}
add2("Development, native resolution, locked out-of-fold gate", dev)
add2("Development, harmonized to external resolution, locked out-of-fold gate", devh)
fit_on <- function(train) {
  fid <- patient_foldid(train$patient_id, CFG$segmenters$cv_folds)
  fit_gate(train, train$fail, OPTICAL_METRICS, fid, df = 3)
}
g_nat <- fit_on(dev)
g_harm <- fit_on(devh)
nz <- function(g) {
  co <- as.matrix(stats::coef(g$fit, s = g$lambda))
  paste(unique(sub("_[0-9]+$", "", rownames(co)[co[, 1] != 0 & rownames(co) != "(Intercept)"])), collapse = " ")
}
e1 <- ext; e1$s_gate <- predict_gate(g_nat, ext, "link")
e2 <- ext; e2$s_gate <- predict_gate(g_harm, ext, "link")
add2("External, gate refitted on native-resolution development data", e1)
add2("External, gate refitted on harmonized development data", e2)
t2 <- do.call(rbind, t2)
t2$nonzero_metrics <- c(NA, NA, nz(g_nat), nz(g_harm))[match(t2$analysis, unique(t2$analysis))]
wcsv(t2, "t2_harmonization_gate")

# does the direction reversal of the retained metrics survive harmonization? OR per development SD, cluster-robust
or_tab <- function(train, label) do.call(rbind, lapply(OPTICAL_METRICS, function(m) {
  tr <- train[, c("patient_id", "fail", m)]; tr$external <- 0
  ex <- ext[, c("patient_id", "fail", m)]; ex$external <- 1
  a <- rbind(tr, ex)
  x <- a[[m]]
  if (m == "sharp_tenengrad") x <- log(x)
  xd <- x[a$external == 0]
  a$z <- (x - mean(xd)) / stats::sd(xd)
  fit <- stats::glm(fail ~ z * external, data = a, family = stats::binomial())
  V <- sandwich::vcovCL(fit, cluster = a$patient_id)
  b <- stats::coef(fit)
  se_int <- sqrt(V["z:external", "z:external"])
  data.frame(development = label, metric = m, retained = m %in% RETAINED,
             or_dev = exp(b[["z"]]), or_dev_lo = exp(b[["z"]] - 1.96 * sqrt(V["z", "z"])), or_dev_hi = exp(b[["z"]] + 1.96 * sqrt(V["z", "z"])),
             or_ext = exp(b[["z"]] + b[["z:external"]]), p_interaction = 2 * stats::pnorm(-abs(b[["z:external"]] / se_int)),
             dev_median = stats::median(train[[m]]), ext_median = stats::median(ext[[m]]),
             frac_ext_outside_dev_range = mean(ext[[m]] < min(train[[m]]) | ext[[m]] > max(train[[m]])))
}))
wcsv(rbind(or_tab(dev, "native"), or_tab(devh, "harmonized")), "t2_harmonization_associations")

# domain separability from the 11 metrics, patient-grouped 5-fold cross-validated logistic model with 3-df splines
shift_auc <- function(train, label) {
  a <- rbind(cbind(train[, c("patient_id", OPTICAL_METRICS)], y = 0), cbind(ext[, c("patient_id", OPTICAL_METRICS)], y = 1))
  fid <- patient_foldid(a$patient_id, 5)
  fml <- stats::as.formula(paste("y ~", paste(sprintf("splines::ns(%s, 3)", OPTICAL_METRICS), collapse = " + ")))
  pr <- numeric(nrow(a))
  for (k in sort(unique(fid))) {
    fit <- suppressWarnings(stats::glm(fml, data = a[fid != k, ], family = stats::binomial()))
    pr[fid == k] <- suppressWarnings(stats::predict(fit, a[fid == k, ], type = "link"))
  }
  data.frame(development = label, shift_auroc = auroc_fail(pr, a$y))
}
wcsv(rbind(shift_auc(dev, "native"), shift_auc(devh, "harmonized")), "t2_harmonization_shift")
cat("T2 harmonization done\n")

# ---- T3 localized and combined perturbations ------------------------------------------------------------------------
pt <- rpq("perturb2")
pt$fail <- as.integer(pt$iou < THR)
pt$gate_p <- plogis(predict_gate(models$gate, pt, "link"))
base <- pt[pt$kind == "none", c("frame_id", "model", "iou", "gate_p", "entropy_fov")]
names(base)[3:5] <- c("iou0", "gate0", "ent0")
pt <- merge(pt, base, by = c("frame_id", "model"))
t3 <- do.call(rbind, lapply(split(pt, list(pt$model, pt$kind, pt$condition, pt$severity), drop = TRUE), function(d) {
  data.frame(model = d$model[1], kind = d$kind[1], condition = d$condition[1], severity = d$severity[1], n = nrow(d),
             failure_rate = mean(d$fail), median_iou = stats::median(d$iou), median_iou_change = stats::median(d$iou - d$iou0),
             median_gate_risk = stats::median(d$gate_p), median_gate_risk_change = stats::median(d$gate_p - d$gate0),
             median_entropy_change = stats::median(d$entropy_fov - d$ent0),
             gate_auroc = auroc_fail(d$gate_p, d$fail), entropy_auroc = auroc_fail(d$entropy_fov, d$fail))
}))
t3 <- t3[order(t3$model, factor(t3$kind, c("none", "local", "combined")), t3$condition, t3$severity), ]
wcsv(t3, "t3_perturbation_local_combined")
pf_stat <- function(d) c(gate_auroc = auroc_fail(d$gate_p, d$fail), entropy_auroc = auroc_fail(d$entropy_fov, d$fail),
                         failure_rate = mean(d$fail))
t3b <- list()
for (k in c("local", "combined")) {
  d <- pt[pt$model == "unet" & pt$kind == k, ]
  ci <- cluster_boot(d, B = B, seed = SEED, stat_fn = pf_stat, cores = CORES)$ci
  ci$kind <- k
  t3b[[k]] <- ci
}
wcsv(do.call(rbind, t3b), "t3_perturbation_pooled")
cat("T3 perturbation done\n")

# ---- T4 latency: gate scoring in R plus the Python timings ----------------------------------------------------------
reps <- 50
tm <- system.time(for (i in seq_len(reps)) predict_gate(models$gate, ext, "link"))[["elapsed"]]
lat <- jsonlite::fromJSON(file.path(OUT, "tierb_latency.json"))
rows <- list(data.frame(step = "Gate scoring from the 11 metrics (R, vectorized)", device = "cpu",
                        median_ms = 1000 * tm / (reps * nrow(ext)), p10_ms = NA, p90_ms = NA, n = reps * nrow(ext)))
for (k in names(lat)) if (is.list(lat[[k]]) && !is.null(lat[[k]]$median_ms)) {
  rows[[k]] <- data.frame(step = k, device = ifelse(grepl("^mps_", k), "mps", "cpu"), median_ms = lat[[k]]$median_ms,
                          p10_ms = lat[[k]]$p10_ms, p90_ms = lat[[k]]$p90_ms, n = lat[[k]]$n)
}
t4 <- do.call(rbind, rows)
t4$machine <- lat$machine
wcsv(t4, "t4_latency")
cat("T4 latency done\n")
writeLines(capture.output(sessionInfo()), file.path(OUT, "tierb_r_sessioninfo.txt"))
