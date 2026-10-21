# Tests for assoc_lib.R, 03_association.R and 06_perturb.R.
#   Rscript v1/R/tests/test_assoc.R        (re-runs itself with OPTICS_DATA=synthetic)
# Reads work/synthetic (read-only); every write goes under work/synthetic/out. The two scripts are run as
# subprocesses, so their CSVs and result parts are left in out/results/ for downstream development.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
R_DIR <- dirname(.here)
source(file.path(R_DIR, "00_setup.R"))
source(file.path(R_DIR, "01_load.R"))
source(file.path(R_DIR, "results_lib.R"))
source(file.path(R_DIR, "30_bundle.R"))
source(file.path(R_DIR, "assoc_lib.R"))
source(file.path(R_DIR, "stats_lib.R"))

failures <- character()
check <- function(name, expr) {
  res <- tryCatch(if (isTRUE(expr)) "ok" else "FALSE", error = function(e) paste("error:", conditionMessage(e)))
  cat(sprintf("%-4s %s\n", if (res == "ok") "ok" else "FAIL", name))
  if (res != "ok") { cat("     ", res, "\n"); failures <<- c(failures, name) }
}
near <- function(a, b, tol = 1e-8) isTRUE(all(abs(a - b) <= tol))
errs <- function(expr, pattern) {
  msg <- tryCatch({ force(expr); "" }, error = function(e) conditionMessage(e))
  nzchar(msg) && grepl(pattern, msg)
}

TMP <- file.path(SYNTH_DIR, "out", "t14_tmp")
dir.create(TMP, recursive = TRUE, showWarnings = FALSE)
real_before <- lapply(c("results", "figures", "tables"), function(d) list.files(file.path(V1, d), recursive = TRUE))
lock_before <- file.exists(file.path(V1, "config", "analysis_lock.yaml"))

# ---- rank statistics ---------------------------------------------------------------------------
cat("-- rank statistics\n")
check("AUROC hand case 0.75", assoc_auroc(c(0.1, 0.4, 0.35, 0.8), c(0, 0, 1, 1)) == 0.75)
check("AUROC all tied = 0.5", assoc_auroc(rep(1, 4), c(0, 1, 0, 1)) == 0.5)
check("AUROC perfect and reversed", assoc_auroc(1:4, c(0, 0, 1, 1)) == 1 && assoc_auroc(4:1, c(0, 0, 1, 1)) == 0)
check("AUROC NA when a class is absent", is.na(assoc_auroc(1:4, rep(1, 4))))
set.seed(1)
sc <- round(rnorm(200), 1)
lb <- rbinom(200, 1, plogis(sc))
check("AUROC equals pROC (ties included)", near(assoc_auroc(sc, lb), as.numeric(pROC::auc(lb, sc, quiet = TRUE, direction = "<"))))
check("stratified AUROC weights by pair count",
      near(assoc_auroc_strat(c(1, 2, 3, 4, 1, 2), c(0, 0, 1, 1, 1, 0), c("a", "a", "a", "a", "b", "b")), 0.8))
x <- c(1, 2, 2, 5, 9, 3)
y <- c(2, 1, 4, 4, 8, 7)
check("Spearman equals cor(method = 'spearman') with ties", near(unname(assoc_spearman(x, y)), cor(x, y, method = "spearman")))
M <- cbind(a = x, b = rev(x))
check("Spearman on a matrix keeps names", identical(names(assoc_spearman(M, y)), c("a", "b")))

# ---- patient bootstrap -------------------------------------------------------------------------
cat("-- bootstrap\n")
pat <- rep(sprintf("P%02d", 1:30), each = 5)
strat <- ifelse(as.integer(substr(pat, 2, 3)) <= 10, "A", "B")
dr <- assoc_boot_draws(pat, strat, B = 40, seed = 7)
pats <- dr$patients
ok_strata <- all(vapply(seq_len(40), function(b) {
  ids <- pats[dr$draws[b, ]]
  sum(strat[match(ids, pat)] == "A") == 10 && sum(strat[match(ids, pat)] == "B") == 20
}, logical(1)))
check("each resample keeps the stratum sizes (10 + 20 patients)", ok_strata)
check("a resample returns whole patients (150 rows every time)",
      all(assoc_boot(pat, function(r) length(r), strat, B = 40, seed = 7, cores = 1) == 150))
fn_mean <- function(r) c(m = mean(seq_along(pat)[r]))
a1 <- assoc_boot(pat, fn_mean, strat, B = 60, seed = 3, cores = 1)
a2 <- assoc_boot(pat, fn_mean, strat, B = 60, seed = 3, cores = 2)
a3 <- assoc_boot(pat, fn_mean, strat, B = 60, seed = 4, cores = 1)
check("same seed reproduces, and cores do not matter", identical(a1, a2))
check("a different seed gives different resamples", !identical(a1, a3))
set.seed(5); u1 <- runif(1)
set.seed(5); invisible(assoc_boot_draws(pat, strat, B = 10, seed = 99)); u2 <- runif(1)
check("global RNG state is left alone", u1 == u2)
check("a patient in two strata is an error", errs(assoc_boot_draws(c("a", "a", "b"), c("x", "y", "x"), 5, 1), "more than one stratum"))
ci <- assoc_boot_ci(function(r) c(m = mean(seq_along(pat)[r])), pat, strat, B = 200, seed = 3, cores = 1)
check("percentile CI brackets the estimate", ci$lo < ci$est && ci$est < ci$hi && ci$n_valid == 200)
sp <- assoc_spearman_boot(cbind(a = 1:150, b = rev(1:150)), 1:150, pat, strat, B = 100, seed = 1, cores = 1)
check("Spearman boot: monotone variables give rho +1 and -1 with a tight CI",
      near(sp$est, c(1, -1)) && all(sp$hi - sp$lo < 1e-9) && all(sp$n_patients == 30))

# ---- cluster-robust logistic model -------------------------------------------------------------
cat("-- cluster-robust model\n")
sim <- function(G, m, sd_u, seed) {
  set.seed(seed)
  g <- rep(seq_len(G), each = m)
  xc <- rep(rnorm(G), each = m)       # cluster-level covariate
  xi <- rnorm(G * m)                  # within-cluster covariate
  u <- rep(rnorm(G, 0, sd_u), each = m)
  list(y = rbinom(G * m, 1, plogis(-0.5 + 0.5 * xc + 0.3 * xi + u)), X = cbind(xc = xc, xi = xi), g = paste0("c", g))
}
s <- sim(150, 12, 1.5, 11)
f <- assoc_fit_robust(s$y, s$X, s$g)
tab <- f$table
ref <- glm(s$y ~ s$X, family = binomial())
ref_se <- sqrt(diag(sandwich::vcovCL(ref, cluster = s$g, type = "HC1", cadjust = TRUE)))
check("coefficients equal glm", near(tab$coef, unname(coef(ref)), 1e-6))
check("robust SE equals sandwich::vcovCL (HC1, cadjust)", near(tab$se_robust, unname(ref_se), 1e-6))
check("with within-patient correlation the robust SE of a cluster-level covariate is larger (ratio > 1.5)",
      tab$se_ratio[tab$term == "xc"] > 1.5)
s0 <- sim(150, 12, 0, 12)
t0 <- assoc_fit_robust(s0$y, s0$X, s0$g)$table
check("without clustering robust and model-based SE agree (ratio within 0.85 to 1.15)", all(t0$se_ratio > 0.85 & t0$se_ratio < 1.15))
check("OR and CI are exp of the Wald interval", near(tab$or_lo, exp(tab$coef - qnorm(0.975) * tab$se_robust)) && near(tab$or, exp(tab$coef)))

# ---- aliasing, VIF, EPV ------------------------------------------------------------------------
cat("-- aliasing, VIF, EPV\n")
set.seed(2)
n <- 400
x1 <- rnorm(n); x3 <- rnorm(n)
yy <- rbinom(n, 1, plogis(0.8 * x1 - 0.5 * x3))
Xa <- cbind(x1 = x1, x3 = x3, dup = 2 * x1 + 1, const = 1)
al <- assoc_drop_aliased(Xa)
check("aliased and constant columns are dropped, in priority order", identical(colnames(al$X), c("x1", "x3")))
check("the reason names the term it equals", grepl("aliased with x1", al$aliased$reason[al$aliased$term == "dup"]) &&
        grepl("constant", al$aliased$reason[al$aliased$term == "const"]))
fa <- suppressWarnings(assoc_fit_model(yy, Xa, paste0("c", rep(1:100, 4)), "x1", label = "alias"))
fb <- glm(yy ~ x1 + x3, family = binomial())
check("the fit with aliased columns equals the fit without them", near(fa$table$coef, unname(coef(fb)), 1e-6) && nrow(fa$aliased) == 2L)
Xv <- cbind(x1 = x1, x2 = x1 + rnorm(n, 0, 0.3), x3 = x3)
check("VIF equals car::vif", near(unname(assoc_vif(Xv)), unname(car::vif(lm(yy ~ x1 + x2 + x3, data = as.data.frame(Xv)))), 1e-8))
e1 <- assoc_epv(c(rep(1, 20), rep(0, 380)), 5)
check("EPV = limiting class / parameters, flagged below 10", e1$epv == 4 && e1$flag && !assoc_epv(c(rep(1, 100), rep(0, 300)), 5)$flag)
check("EPV uses the smaller class", assoc_epv(c(rep(1, 390), rep(0, 10)), 5)$epv == 2)

few <- sim(60, 10, 0.5, 21)
few$y <- c(rep(1, 25), rep(0, 575))[sample(600)]
Xf <- cbind(few$X, w1 = rnorm(600), w2 = rnorm(600), w3 = rnorm(600))
warns <- character()
rf <- withCallingHandlers(
  assoc_fit_model(few$y, Xf, few$g, c("xc", "xi", "w1", "w2", "w3"), label = "toy", B = 200, cores = 1),
  warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
check("EPV < 10 warns, names the numbers, and switches to the ridge fit",
      any(grepl("events per variable = 5\\.0", warns)) && rf$ridge_used && rf$method == "ridge logistic" && rf$epv$flag)
pl <- suppressWarnings(assoc_fit_model(few$y, Xf, few$g, c("xc", "xi", "w1", "w2", "w3"), label = "toy", force_plain = TRUE))
sh <- function(tab) abs(log(tab$or[match(c("xc", "xi", "w1", "w2", "w3"), tab$term)]))
check("the ridge ORs are finite and shrunk towards 1 relative to the plain fit",
      all(is.finite(c(rf$table$or_lo, rf$table$or_hi))) && sum(sh(rf$table)) < sum(sh(pl$table)))
big <- sim(150, 12, 0.5, 22)
w2 <- character()
rb <- withCallingHandlers(assoc_fit_model(big$y, big$X, big$g, c("xc", "xi"), label = "big"),
                          warning = function(w) { w2 <<- c(w2, conditionMessage(w)); invokeRestart("muffleWarning") })
check("EPV >= 10 uses the plain cluster-robust fit without a warning", !rb$ridge_used && !length(w2) && rb$method == "cluster-robust logistic")

# ---- perturbation helpers ----------------------------------------------------------------------
cat("-- severity threshold and curves\n")
check("first severity below 0.75 on a toy curve (interpolated 2.5)",
      with(assoc_first_below(0:5, c(0.95, 0.9, 0.8, 0.7, 0.6, 0.5), 0.75), first == 3 && near(interp, 2.5)))
check("never below -> NA", with(assoc_first_below(0:5, c(0.95, 0.9, 0.85, 0.8, 0.78, 0.76), 0.75), is.na(first) && is.na(interp)))
check("exactly at the threshold is not below", assoc_first_below(0:3, c(0.9, 0.75, 0.6, 0.5), 0.75)$first == 2)
check("baseline already below -> severity 0", with(assoc_first_below(0:3, c(0.7, 0.6, 0.5, 0.4), 0.75), first == 0 && interp == 0))
check("unsorted input is ordered by severity", assoc_first_below(c(3, 0, 2, 1), c(0.5, 0.9, 0.6, 0.8), 0.75)$first == 2)

toy <- do.call(rbind, lapply(1:16, function(i) {
  data.frame(frame_id = paste0("f", i), patient_id = paste0("p", (i - 1) %/% 4 + 1), domain = ifelse(i <= 8, "D1", "D2"),
             model = "unet", family = "defocus", severity = 0:3, param = c(0, 1, 2, 3),
             iou = 0.95 - 0.12 * (0:3) + (i %% 4) * 0.004, stringsAsFactors = FALSE)
}))
cu <- assoc_perturb_curves(toy, thr = 0.75, B = 200, cores = 1)
check("toy curves: one row per severity with medians and ordered bands",
      nrow(cu$curves) == 4 && all(cu$curves$median_lo <= cu$curves$median_hi) && all(cu$curves$mean_lo <= cu$curves$mean_hi) &&
        all(diff(cu$curves$median_iou) < 0) && all(cu$curves$n_frames == 16) && all(cu$curves$n_patients == 4))
check("toy threshold: median 0.95, 0.83, 0.71, 0.59 -> first severity 2, never-share 0",
      cu$thresholds$first_severity_below == 2 && cu$thresholds$prob_never == 0 && near(cu$thresholds$param_at_first, 2))
cu2 <- assoc_perturb_curves(transform(toy, iou = pmin(iou + 0.3, 0.99)), thr = 0.4, B = 200, cores = 1)
check("toy threshold never reached -> NA with prob_never 1", is.na(cu2$thresholds$first_severity_below) && cu2$thresholds$prob_never == 1)

# ---- output guard and lock helpers -------------------------------------------------------------
cat("-- guards\n")
check("synthetic p() paths pass the guard", is.character(assoc_guard_path(p("results", "stats", "x.csv"))))
check("a real v1 path is refused", errs(assoc_guard_path(file.path(V1, "results", "stats", "x.csv")), "outside work/synthetic/out"))
check("a ../ escape out of work/synthetic/out is refused", errs(assoc_guard_path(file.path(SYNTH_DIR, "out", "..", "..", "x.csv")), "outside"))
check("assoc_write_csv refuses and creates nothing on a real path",
      errs(assoc_write_csv(data.frame(a = 1), file.path(V1, "results", "stats", "t14_guard.csv")), "outside") &&
        !file.exists(file.path(V1, "results", "stats", "t14_guard.csv")))
check("results_root() is inside work/synthetic/out", isTRUE(assoc_guard_results_root()))
lk <- file.path(TMP, "lock_a.yaml")
check("no lock file -> not complete, plan threshold", !assoc_lock_complete(lk) && assoc_failure_threshold(lk) == CFG$endpoints$failure_iou_threshold)
yaml::write_yaml(list(failure_iou_threshold = 0.7, gate_rds_sha256 = ""), lk)
check("empty hash -> not complete; lock threshold wins", !assoc_lock_complete(lk) && assoc_failure_threshold(lk) == 0.7)
yaml::write_yaml(list(failure_iou_threshold = 0.7, gate_rds_sha256 = "abc123"), lk)
check("recorded hash -> complete", assoc_lock_complete(lk))
src <- readLines(file.path(R_DIR, "03_association.R"))
i_gate <- grep("if \\(nrow\\(c2\\) && ext_open\\) \\{", src)[1]
i_read <- grep('assoc_read_unet\\("center2_labeled"\\)', src)
check("03 reads Center 2 outcomes only inside the lock-gated block", length(i_read) == 1L && !is.na(i_gate) && i_read > i_gate)
c1p <- assoc_read_unet("center1_labeled")
check("assoc_read_unet filters inside arrow: Center 1 rows only", nrow(c1p) == 1060L && all(c1p$unet_source == "oof"))

# The checks below assume failure threshold 0.75 (v1's own synthetic lock); a missing or 04a-written synthetic lock gives the
# plan's 0.90. So set 0.75 here, in the synthetic out folder only (guarded), and restore the previous lock (or none) at exit.
lock_syn_file <- p("config", "analysis_lock.yaml")
assoc_guard_path(lock_syn_file)
lock_syn_prev <- if (file.exists(lock_syn_file)) readBin(lock_syn_file, "raw", file.size(lock_syn_file))
invisible(reg.finalizer(globalenv(), function(e) if (is.null(lock_syn_prev)) unlink(lock_syn_file) else writeBin(lock_syn_prev, lock_syn_file),
                        onexit = TRUE))
dir.create(dirname(lock_syn_file), recursive = TRUE, showWarnings = FALSE)
yaml::write_yaml(list(failure_iou_threshold = 0.75, noninferiority_margin_aurc = 0.01), lock_syn_file)

# ---- 03_association.R on the synthetic data ----------------------------------------------------
cat("-- 03_association.R\n")
run_script <- function(script, env) {
  out <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"), file.path(R_DIR, script),
                                  env = c("OPTICS_DATA=synthetic", env), stdout = TRUE, stderr = TRUE))
  list(status = attr(out, "status") %||% 0L, log = out)
}
rd <- function(f) utils::read.csv(file.path(SYNTH_DIR, "out", "results", "stats", f), stringsAsFactors = FALSE)
part <- function(name) jsonlite::fromJSON(file.path(SYNTH_DIR, "out", "results", "parts", paste0(name, ".json")), simplifyVector = FALSE)$entries

t_a <- system.time(r0 <- run_script("03_association.R", "OPTICS_ASSOC_EXTERNAL=0"))
check("03 runs with the external block closed", r0$status == 0L)
o0 <- rd("assoc_or_models.csv")
e0 <- part("03_association")
check("closed lock: only Center 1 models, no Center 2 outcome keys",
      setequal(unique(o0$scope), "center1_oof") && !any(grepl("^assoc_ext_(n_fail|or_)", names(e0))))
check("closed lock: Center 2 scenario descriptives are still produced", "assoc_ext_auroc_under_frac" %in% names(e0) &&
        any(rd("assoc_by_domain.csv")$domain == "D3"))
t_b <- system.time(r1 <- run_script("03_association.R", "OPTICS_ASSOC_EXTERNAL=1"))
check("03 runs with the external block open", r1$status == 0L)
cat(sprintf("     03 runtime: %.1f s (closed), %.1f s (open)\n", t_a[["elapsed"]], t_b[["elapsed"]]))

t2 <- rd("assoc_table2.csv")
T2_COLS <- c("metric", "label", "quantity", "n_normal", "n_difficult", "median_normal", "q1_normal", "q3_normal",
             "median_difficult", "q1_difficult", "q3_difficult", "normal_median_iqr", "difficult_median_iqr",
             "auroc_difficult", "auroc_difficult_lo", "auroc_difficult_hi", "auroc_text", "auroc_direction",
             "auroc_within", "auroc_within_lo", "auroc_within_hi", "or_sd", "or_lo", "or_hi", "or_p", "or_scale",
             "or_method", "or_text", "or_joint", "or_joint_lo", "or_joint_hi", "or_joint_text", "or_joint_method", "vif_joint")
check("assoc_table2.csv: one row per metric in config order, exact columns", identical(t2$metric, OPTICAL_METRICS) && identical(names(t2), T2_COLS))
num <- setdiff(T2_COLS, c("metric", "label", "quantity", "normal_median_iqr", "difficult_median_iqr", "auroc_text",
                          "auroc_direction", "or_scale", "or_method", "or_text", "or_joint_text", "or_joint_method"))
check("no missing numbers in Table 2", !anyNA(t2[num]))
row <- function(m) t2[t2$metric == m, ]
check("planted metrics: univariable adjusted OR away from 1 in the planted direction (CI excludes 1)",
      row("under_frac")$or_lo > 1 && row("sharp_tenengrad")$or_hi < 1 && row("haze_dcp")$or_lo > 1)
check("the sharpness OR is stated per SD of the log value", row("sharp_tenengrad")$or_scale == "per SD of log value" && row("lum_mean")$or_scale == "per SD")
check("AUROC for Difficult: under-exposure and haze above 0.5, sharpness below 0.5 (CIs exclude 0.5)",
      row("under_frac")$auroc_difficult_lo > 0.5 && row("haze_dcp")$auroc_difficult_lo > 0.5 && row("sharp_tenengrad")$auroc_difficult_hi < 0.5)
check("CIs bracket their estimates", all(t2$auroc_difficult_lo <= t2$auroc_difficult & t2$auroc_difficult <= t2$auroc_difficult_hi) &&
        all(t2$or_lo <= t2$or_sd & t2$or_sd <= t2$or_hi))
dg <- rd("assoc_diagnostics.csv")
jt <- dg[dg$scope == "center1_oof" & dg$model == "joint_linear", ]
check("joint model: EPV below 10 flagged and the ridge fit chosen automatically, stated in the output",
      jt$epv_flag && jt$epv < 10 && jt$method == "ridge logistic" && grepl("events per variable", jt$note) &&
        all(t2$or_joint_method == "ridge logistic"))
planted <- c("under_frac", "sharp_tenengrad", "haze_dcp")
check("joint ridge ORs keep the planted direction; the 8 unplanted metrics stay within 0.6 to 1.5",
      row("under_frac")$or_joint > 1 && row("sharp_tenengrad")$or_joint < 1 && row("haze_dcp")$or_joint > 1 &&
        all(t2$or_joint[!t2$metric %in% planted] > 0.6 & t2$or_joint[!t2$metric %in% planted] < 1.5))
uni <- dg[dg$scope == "center1_oof" & dg$model == "univariable_adjusted", ]
check("univariable models: EPV above 10, plain cluster-robust fit", all(uni$epv >= 10) && all(uni$method == "cluster-robust logistic"))
# NOTE on the synthetic data: image format coincides with domain here (png only in D3, Center 1 is all jpg). On the real data
# it does not (Center 1 has 5,442 png frames: D2 is all png, D1 is mixed), so format is estimable inside Center 1. These checks
# exercise the aliasing report on a subset where format and domain do coincide; they are not a statement about the real data.
uni <- dg[dg$scope == "center1_oof" & dg$model == "univariable_adjusted", ]
check("univariable models: EPV above 10, plain cluster-robust fit", all(uni$epv >= 10) && all(uni$method == "cluster-robust logistic"))
check("aliasing is reported, not hidden: in the synthetic Center 1 (all jpg) file format is single-level",
      all(grepl("file_format \\(single level", uni$aliased_terms)))
pooled <- dg[dg$scope == "pooled_c1_c2", ]
check("pooled run (synthetic data, png only in D3): png is aliased with domain D3 and said so",
      nrow(pooled) == 11L && all(grepl("file_formatpng \\(aliased with domainD3\\)", pooled$aliased_terms)))
ext <- dg[dg$scope == "center2_external", ]
check("external run: domain, channel and format are single-level and named", nrow(ext) == 11L &&
        all(grepl("domain \\(single level", ext$aliased_terms) & grepl("channel \\(single", ext$aliased_terms) & grepl("file_format \\(single", ext$aliased_terms)))
pr_s <- load_predictions()
fv_s <- load_fov()
c2_found <- pr_s[pr_s$model == "unet" & pr_s$partition == "center2_labeled" & pr_s$frame_id %in% fv_s$frame_id[fv_s$fov_found], ]
c2_fail <- c2_found$iou < assoc_failure_threshold()
check("external events are counted on the frames with a detected FOV and pass the minimum",
      all(ext$events == sum(c2_fail)) && all(ext$patients_with_events == n_distinct(c2_found$patient_id[c2_fail])) &&
        sum(c2_fail) >= CFG$endpoints$min_external_failure_events && all(ext$n == nrow(c2_found)))
w <- rd("assoc_vif.csv")
check("VIF table: joint model, max matches the diagnostics", nrow(w) == jt$n_params && near(max(w$vif), jt$vif_max))
bd <- rd("assoc_by_domain.csv")
check("by-domain table: D1 and D2 from Center 1, D3 flagged descriptive", setequal(unique(bd$scope), c("center1_oof", "center2_external_descriptive")) &&
        setequal(unique(bd$domain), c("D1", "D2", "D3")) && nrow(bd) == 33L)
e1 <- part("03_association")
check("registered keys: OR, AUROC, medians for all 11 metrics and the check counts",
      all(paste0("assoc_or_", OPTICAL_METRICS) %in% names(e1)) && all(paste0("assoc_auroc_within_", OPTICAL_METRICS) %in% names(e1)) &&
        all(c("assoc_epv_joint", "assoc_joint_method", "assoc_vif_max_joint", "assoc_se_ratio_median", "assoc_ext_or_estimated") %in% names(e1)))
check("registered OR text matches the CSV (equal precision)", e1$assoc_or_under_frac$display ==
        paste0(sprintf("%.2f", row("under_frac")$or_sd), " (95% CI ", sprintf("%.2f", row("under_frac")$or_lo), " to ",
               sprintf("%.2f", row("under_frac")$or_hi), ")"))
check("Tenengrad medians are registered in scientific notation", grepl("e-", e1$assoc_median_normal_sharp_tenengrad$display))

# ---- the real-data situation: labeled frames whose field of view was not found ------------------
# On the real data 3 labeled Center 1 frames have fov_found == FALSE, NaN optics and no prediction rows. A copy of the
# synthetic tables is made here with exactly that (the shared synthetic files are not edited).
cat("-- frames without a detected FOV\n")
quiet <- function(expr) suppressMessages(suppressWarnings(expr))
fx <- file.path(TMP, "fov_fixture")
unlink(fx, recursive = TRUE)
fx_syn <- file.path(fx, "work", "synthetic")
dir.create(file.path(fx, "config"), recursive = TRUE)
dir.create(fx_syn, recursive = TRUE)
invisible(file.copy(file.path(V1, "config", "analysis_plan.yaml"), file.path(fx, "config")))
invisible(file.copy(file.path(SYNTH_DIR, "folds.csv"), fx_syn))
rd_syn <- function(nm) quiet(arrow::read_parquet(file.path(SYNTH_DIR, paste0(nm, ".parquet"))))
man_f <- rd_syn("manifest")
fov_f <- rd_syn("fov")
opt_f <- rd_syn("optics")
prd_f <- rd_syn("predictions")
fov_f$fov_found[fov_f$frame_id %in% man_f$frame_id[man_f$labeled]] <- TRUE   # real labeled frames: all found but 3
c1_ids <- man_f$frame_id[man_f$partition == "center1_labeled"]
nf3 <- c1_ids[c(10, 400, 900)]
fov_f$fov_found[fov_f$frame_id %in% nf3] <- FALSE
fov_f$fov_method[fov_f$frame_id %in% nf3] <- "fallback"
opt_f[opt_f$frame_id %in% nf3, OPTICAL_METRICS] <- NaN
prd_f <- prd_f[!prd_f$frame_id %in% nf3, ]
for (nm in c("manifest", "fov", "optics", "predictions")) {
  quiet(arrow::write_parquet(get(paste0(c(manifest = "man", fov = "fov", optics = "opt", predictions = "prd")[[nm]], "_f")), file.path(fx_syn, paste0(nm, ".parquet"))))
}
with_derived <- function(dir, expr) {
  old <- SYNTH_DIR
  assign("SYNTH_DIR", dir, globalenv())
  on.exit(assign("SYNTH_DIR", old, globalenv()))
  expr
}
check("fixture: 3 labeled Center 1 frames are unfound, have NaN optics and no prediction rows",
      sum(!fov_f$fov_found[fov_f$frame_id %in% man_f$frame_id[man_f$labeled]]) == 3L && all(is.nan(opt_f$lum_mean[opt_f$frame_id %in% nf3])) &&
        !any(prd_f$frame_id %in% nf3) && sum(prd_f$model == "unet" & prd_f$partition == "center1_labeled") == 1060L - 3L)
check("without the exclusion the loader stops on the NaN optics (the pre-fix failure)",
      errs(with_derived(fx_syn, load_frame_table(with_predictions = FALSE, exclude_no_fov = FALSE)), "disagree"))
ft_x <- with_derived(fx_syn, quiet(load_frame_table(with_predictions = FALSE)))
check("with the exclusion (default from the plan) the table loads without them and reports the count",
      nrow(ft_x) == 1500L - 3L && attr(ft_x, "n_excluded_fov") == 3L && !any(nf3 %in% ft_x$frame_id) && !anyNA(ft_x$lum_mean))
ft_xp <- with_derived(fx_syn, quiet(load_frame_table(with_predictions = TRUE)))
check("with predictions: every remaining frame has its U-Net and SegFormer row, count still 3",
      nrow(ft_xp) == 1497L && !anyNA(ft_xp$unet_iou) && !anyNA(ft_xp$segformer_iou) && attr(ft_xp, "n_excluded_fov") == 3L)
fx_out <- file.path(fx_syn, "out")
rd_fx <- function(f) utils::read.csv(file.path(fx_out, "results", "stats", f), stringsAsFactors = FALSE)
part_fx <- function(name) jsonlite::fromJSON(file.path(fx_out, "results", "parts", paste0(name, ".json")), simplifyVector = FALSE)$entries
t_x <- system.time(rx <- run_script("03_association.R", c(paste0("OPTICS_V1=", fx), "OPTICS_ASSOC_EXTERNAL=1")))
check("03_association.R passes on the fixture (9,593 versus 9,590 situation: tables with and without the unfound frames)", rx$status == 0L)
check("03 reports the exclusion in its log", any(grepl("excluded 3 labeled frame", rx$log)))
ex_part <- part_fx("03_association")
check("03 registers the count (3) and uses the 1,057 Center 1 and 440 Center 2 frames that remain",
      ex_part$assoc_n_fov_excluded$display == "3" && ex_part$assoc_n_frames_c1$raw == 1057 && ex_part$assoc_n_frames_c2$raw == 440)
check("the fixture run wrote only under its own tree", file.exists(file.path(fx_out, "results", "parts", "03_association.json")) &&
        identical(real_before, lapply(c("results", "figures", "tables"), function(d) list.files(file.path(V1, d), recursive = TRUE))))

# ---- a sealed lock must verify before 03 uses its threshold ---------------------------------------
mdir <- file.path(fx_out, "work", "models")
dir.create(mdir, recursive = TRUE)
saveRDS(list(1), file.path(mdir, "obj.rds"))
lk_fx <- list(failure_iou_threshold = 0.74, epochs = list(unet = 20L, segformer = 20L), noninferiority_margin_aurc = 0.01,
              entropy_baseline_choice = "entropy_fov", locked_at = "2026-10-20T10:00:00+0000")
for (o in GATE_LOCK_OBJECTS) {
  lk_fx[[paste0(o, "_path")]] <- "work/models/obj.rds"
  lk_fx[[paste0(o, "_sha256")]] <- unname(tools::sha256sum(file.path(mdir, "obj.rds")))
}
lk_fx$lock_fields_sha256 <- lock_fields_hash(lk_fx)
dir.create(file.path(fx_out, "config"), recursive = TRUE)
lk_file <- file.path(fx_out, "config", "analysis_lock.yaml")
yaml::write_yaml(lk_fx, lk_file)
t_l <- system.time(rl <- run_script("03_association.R", paste0("OPTICS_V1=", fx)))
cat(sprintf("     03 runtime on the fixture: %.1f s (forced open), %.1f s (sealed lock)\n", t_x[["elapsed"]], t_l[["elapsed"]]))
pl <- part_fx("03_association")
check("sealed and intact lock: 03 runs, takes the locked threshold and opens the Center 2 block",
      rl$status == 0L && pl$assoc_failure_iou_threshold$display == "0.74" && !is.null(pl$assoc_ext_or_estimated))
yaml::write_yaml(modifyList(lk_fx, list(failure_iou_threshold = 0.5)), lk_file)
rt <- run_script("03_association.R", paste0("OPTICS_V1=", fx))
check("sealed lock with the threshold edited afterwards: 03 stops on lock_fields_sha256", rt$status != 0L && any(grepl("lock_fields_sha256 mismatch", rt$log)))
unlink(lk_file)

# ---- 06_perturb.R on the synthetic data --------------------------------------------------------
cat("-- 06_perturb.R\n")
t_c <- system.time(r6 <- run_script("06_perturb.R", character()))
check("06 runs", r6$status == 0L)
cat(sprintf("     06 runtime: %.1f s\n", t_c[["elapsed"]]))
dr6 <- rd("perturb_doseresponse.csv")
th <- rd("perturb_threshold.csv")
rh <- rd("perturb_rho.csv")
check("dose-response: 2 models x 7 families x 6 severities, exact columns",
      nrow(dr6) == 84L && identical(names(dr6), c("model", "family", "severity", "param", "n_frames", "n_patients", "median_iou",
        "median_lo", "median_hi", "mean_iou", "mean_lo", "mean_hi", "median_rel_baseline")))
check("severity 0 curves are the same in every family (baseline repeated)",
      all(tapply(dr6$median_iou[dr6$severity == 0], dr6$model[dr6$severity == 0], function(v) diff(range(v))) < 1e-12))
check("median IoU falls with severity in every family (planted dose-response)",
      all(unlist(lapply(split(dr6, paste(dr6$model, dr6$family)), function(d) all(diff(d$median_iou[order(d$severity)]) < 0.01)))))
check("bands are ordered and bracket the median for at least 90% of points",
      all(dr6$median_lo <= dr6$median_hi) && mean(dr6$median_lo <= dr6$median_iou & dr6$median_iou <= dr6$median_hi) >= 0.9)
check("threshold table: 14 rows, exact columns",
      nrow(th) == 14L && identical(names(th), c("model", "family", "threshold_iou", "baseline_median_iou", "baseline_below_threshold",
        "first_severity_below", "severity_interp", "param_at_first", "first_severity_lo", "first_severity_hi", "prob_never",
        "never_reached", "median_iou_max_severity")))
tu <- th[th$model == "unet", ]
check("planted ordering: exposure_under drops first, U-Net glare never crosses (NA)",
      tu$first_severity_below[tu$family == "exposure_under"] == min(tu$first_severity_below, na.rm = TRUE) &&
        is.na(tu$first_severity_below[tu$family == "glare"]) && tu$never_reached[tu$family == "glare"])
ts <- th[th$model == "segformer", ]
fs <- function(d) ifelse(is.na(d$first_severity_below[match(tu$family, d$family)]), 99, d$first_severity_below[match(tu$family, d$family)])
check("SegFormer (15% more fragile) crosses no later than U-Net in every family", all(fs(ts) <= fs(tu)))
check("the rule value comes from the plan", all(th$threshold_iou == CFG$perturbation$rule_first_below_iou))
check("rho table: 7 families x 11 metrics, exact columns", nrow(rh) == 77L && identical(names(rh), c("family", "metric", "label", "rho",
        "rho_lo", "rho_hi", "n_obs", "n_frames", "n_patients", "expected_pair", "expected_sign", "known_limitation", "as_expected")))
ex <- rh[rh$expected_pair & !rh$known_limitation, ]
check("all 11 expected metric/family pairs have the expected sign and |rho| >= 0.5", nrow(ex) == 11L && all(ex$as_expected))
nz <- rh[rh$family == "noise" & rh$metric == "sharp_tenengrad", ]
check("noise inflates Tenengrad (rho > 0, CI above 0), flagged as a known limitation", nz$rho > 0 && nz$rho_lo > 0 && nz$known_limitation)
check("expected pairs are stronger than the other pairs on median |rho|", median(abs(ex$rho)) > median(abs(rh$rho[!rh$expected_pair])))
check("rho CIs bracket the estimate for at least 90% of rows", mean(rh$rho_lo <= rh$rho & rh$rho <= rh$rho_hi, na.rm = TRUE) >= 0.9)
e6 <- part("06_perturb")
check("registered: thresholds (numeric or the string 'not reached'), rho and counts",
      e6$perturb_thr_unet_glare$display == "not reached" && is.numeric(e6$perturb_thr_unet_defocus$raw) &&
        all(paste0("perturb_rho_", rh$family, "_", rh$metric) %in% names(e6)) && e6$perturb_n_expected_ok$display == "11")

# ---- the two result parts bundle without key clashes ------------------------------------------
cat("-- bundle\n")
broot <- file.path(TMP, "bundle")
unlink(broot, recursive = TRUE)
dir.create(file.path(broot, "results", "parts"), recursive = TRUE)
for (nm in c("03_association", "06_perturb")) {
  file.copy(file.path(SYNTH_DIR, "out", "results", "parts", paste0(nm, ".json")), file.path(broot, "results", "parts"))
}
bn <- tryCatch(bundle_results(broot), error = function(e) conditionMessage(e))
check("03 and 06 parts bundle together (unique keys)", is.list(bn) && bn$n_keys == length(e1) + length(e6))
check("bundled results.json reads back the OR key", identical(results_get("assoc_or_under_frac", root = broot), e1$assoc_or_under_frac$display))

# ---- nothing leaked into the real tree ---------------------------------------------------------
unlink(TMP, recursive = TRUE)
real_after <- lapply(c("results", "figures", "tables"), function(d) list.files(file.path(V1, d), recursive = TRUE))
check("real results/, figures/, tables/ and the lock file are untouched", identical(real_before, real_after) &&
        identical(lock_before, file.exists(file.path(V1, "config", "analysis_lock.yaml"))))

cat(sprintf("\n%s\n", if (length(failures)) paste0(length(failures), " FAILED: ", paste(failures, collapse = "; ")) else "all checks passed"))
if (length(failures)) quit(status = 1)
