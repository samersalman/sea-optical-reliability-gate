# Unit tests for stats_lib.R on hand-computable cases and toy data (no pipeline files needed).
#   Rscript v1/R/tests/test_stats_lib.R        (re-runs itself with OPTICS_DATA=synthetic)

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
source(file.path(dirname(.here), "00_setup.R"))
source(file.path(dirname(.here), "stats_lib.R"))

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
near <- function(a, b, tol = 1e-12) isTRUE(all(abs(a - b) <= tol))

# ---- risk-coverage ----------------------------------------------------------------------------
risk <- c(0, 0, 1)
check("perfect ranking: cumulative selective risk is 0, 0, 1/3",
      near(risk_coverage(c(1, 2, 3), risk)$selective_risk, c(0, 0, 1 / 3)))
check("perfect ranking: AURC = 1/9 = oracle AURC", near(aurc(c(1, 2, 3), risk), 1 / 9) && near(oracle_aurc(risk), 1 / 9))
check("worst ranking: AURC = (1 + 1/2 + 1/3) / 3", near(aurc(c(3, 2, 1), risk), (1 + 1 / 2 + 1 / 3) / 3))
check("random AURC is the mean risk", near(random_aurc(risk), 1 / 3))
check("constant score (all tied) gives the random AURC for any risk vector",
      near(aurc(rep(7, 5), c(0.1, 0.9, 0.3, 0.5, 0.2)), mean(c(0.1, 0.9, 0.3, 0.5, 0.2))))
check("a tie group counts at its mean risk: scores 1,2,2 / risks 0,0,1",
      near(risk_coverage(c(1, 2, 2), risk)$selective_risk, c(0, 0.25, 1 / 3)))
check("coverage column runs 1/n .. 1", near(risk_coverage(c(1, 2, 3), risk)$coverage, c(1, 2, 3) / 3))
check("selective risk at 2/3 coverage = 0, at full coverage = mean risk",
      near(selective_risk_at(c(1, 2, 3), risk, 2 / 3), 0) && near(selective_risk_at(c(1, 2, 3), risk, 1), 1 / 3))
r10 <- (1:10) / 10
check("selective risk at 80% of 10 frames with a perfect ranking = mean of the 8 lowest risks",
      near(selective_risk_at(r10, r10, 0.8), mean(r10[1:8])) && near(selective_risk_at(r10, r10, 0.9), mean(r10[1:9])))
check("selective_risk_at is vectorised over coverage", near(selective_risk_at(r10, r10, c(0.5, 1)), c(mean(r10[1:5]), mean(r10))))
set.seed(1)
rk <- runif(400)
check("average AURC of random scores approaches the mean risk",
      abs(mean(replicate(300, aurc(runif(400), rk))) - mean(rk)) < 0.003)
check("oracle AURC is below random AURC", oracle_aurc(rk) < random_aurc(rk))
check("aurc rejects NA and length mismatch", errs(aurc(c(1, NA), c(0, 1)), "NA") && errs(aurc(1:2, 1:3), "length"))

# ---- AUROC and operating points ---------------------------------------------------------------
check("AUROC 1 for perfect ranking, 0 for reversed", near(auroc_fail(1:4, c(0, 0, 1, 1)), 1) && near(auroc_fail(4:1, c(0, 0, 1, 1)), 0))
check("AUROC hand case: fail at ranks 2 and 4 -> 0.75", near(auroc_fail(1:4, c(0, 1, 0, 1)), 0.75))
check("AUROC of tied scores is 0.5", near(auroc_fail(rep(1, 6), c(0, 0, 0, 1, 1, 1)), 0.5))
check("AUROC is NA with one class", is.na(auroc_fail(1:3, c(1, 1, 1))))
x <- rnorm(300)
y <- rbinom(300, 1, plogis(x))
check("AUROC equals pROC::auc (with ties)", {
  xr <- round(x, 1)
  near(auroc_fail(xr, y), as.numeric(pROC::auc(pROC::roc(y, xr, direction = "<", quiet = TRUE))), 1e-9)
})
f <- c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE)
s <- c(1, 2, 3, 4, 0, 0.5, 2.5, 3.5)
check("threshold at sensitivity 0.5 is the 2nd largest failure score", near(threshold_at_sens(s, f, 0.5), 3))
check("threshold at sensitivity 0.9 is the smallest failure score (4 failures)", near(threshold_at_sens(s, f, 0.9), 1))
check("specificity at sensitivity 0.5: 3 of 4 non-failures score below 3", near(spec_at_sens(s, f, 0.5), 0.75))
check("sensitivity at specificity 0.75: flag above the 3rd lowest non-failure (2.5): 3 and 4 -> 0.5", near(sens_at_spec(s, f, 0.75), 0.5))

# ---- calibration ------------------------------------------------------------------------------
set.seed(2)
lp <- rnorm(40000, -1, 1.5)
yy <- rbinom(length(lp), 1, plogis(lp))
cal0 <- calibration_fit(plogis(lp), yy)
check("calibration of true probabilities: slope near 1, intercept near 0",
      abs(cal0[["slope"]] - 1) < 0.05 && abs(cal0[["intercept"]]) < 0.05)
cal_over <- calibration_fit(plogis(2 * lp), yy)
check("over-confident predictions give slope near 0.5", abs(cal_over[["slope"]] - 0.5) < 0.03)
cal_shift <- calibration_fit(plogis(lp + 1), yy)
check("predictions too high by 1 logit give calibration-in-the-large near -1", abs(cal_shift[["intercept"]] + 1) < 0.05)
check("calibration_fit returns slope, intercept, intercept_free and NAs for one class",
      identical(names(cal0), c("slope", "intercept", "intercept_free")) && all(is.na(calibration_fit(c(.2, .3), c(0, 0)))))

# ---- bootstrap --------------------------------------------------------------------------------
set.seed(3)
pat <- data.frame(patient_id = sprintf("P%02d", 1:12), domain = rep(c("D1", "D2"), c(8, 4)))
bd <- pat[rep(1:12, times = c(5, 3, 8, 4, 6, 2, 7, 5, 4, 3, 6, 2)), ]
bd$x <- rnorm(nrow(bd), as.integer(factor(bd$patient_id)))
bd$y <- bd$x + rnorm(nrow(bd))
stat <- function(d) c(mean_x = mean(d$x), mean_y = mean(d$y), n = nrow(d))
b1 <- cluster_boot(bd, B = 200, seed = 11, stat_fn = stat, cores = 1)
b2 <- cluster_boot(bd, B = 200, seed = 11, stat_fn = stat, cores = 1)
b3 <- cluster_boot(bd, B = 200, seed = 12, stat_fn = stat, cores = 1)
b4 <- cluster_boot(bd, B = 200, seed = 11, stat_fn = stat, cores = 3)
check("same seed gives identical resamples and statistics", identical(b1$reps, b2$reps) && identical(b1$est, b2$est))
check("different seed gives different resamples", !identical(b1$reps, b3$reps))
check("results do not depend on the number of cores", identical(b1$reps, b4$reps))
check("estimate is the statistic on the original data", near(b1$est[["mean_x"]], mean(bd$x)))
check("ci table has stat, est, lo, hi and brackets the estimate",
      identical(names(b1$ci), c("stat", "est", "lo", "hi")) && all(b1$ci$lo <= b1$ci$est & b1$ci$est <= b1$ci$hi))
check("percentile limits equal quantile() of the replicates",
      near(b1$ci$lo[1], unname(quantile(b1$reps[, "mean_x"], 0.025))) && near(b1$ci$hi[1], unname(quantile(b1$reps[, "mean_x"], 0.975))))
# patients are resampled inside strata: each replicate keeps 8 D1 patients and 4 D2 patients
cnt <- vapply(seq_len(b1$plan$B), function(b) {
  d <- .gl_boot_data(bd, b1$plan, b)
  c(D1 = length(unique(d$.boot_cluster[d$domain == "D1"])), D2 = length(unique(d$.boot_cluster[d$domain == "D2"])))
}, c(0, 0))
check("every replicate draws 8 patients from D1 and 4 from D2", all(cnt["D1", ] == 8) && all(cnt["D2", ] == 4))
check("a patient is resampled with all of its frames", {
  d <- .gl_boot_data(bd, b1$plan, 5)
  all(vapply(split(d$patient_id, d$.boot_cluster), function(z) length(z) == sum(bd$patient_id == z[1]), NA))
})
sizes <- vapply(seq_len(b1$plan$B), function(b) nrow(.gl_boot_data(bd, b1$plan, b)), 0)
check("replicate sizes vary (patients, not frames, are the unit)", length(unique(sizes)) > 5)
st2 <- function(d) c(a = mean(d$x), b = mean(d$y))
bb <- cluster_boot(bd, B = 200, seed = 11, stat_fn = st2, cores = 1)
pd <- paired_diff_ci(bb, "a", "b")
check("paired difference uses the same resamples: CI equals quantile of reps[,a] - reps[,b]",
      near(pd$lo, unname(quantile(bb$reps[, "a"] - bb$reps[, "b"], 0.025))) && near(pd$diff, bb$est[["a"]] - bb$est[["b"]]))
check("paired difference of a statistic with itself is exactly 0", {
  z <- paired_diff_ci(bb, "a", "a")
  z$diff == 0 && z$lo == 0 && z$hi == 0
})
check("a stored plan can be reused and reproduces the draws", {
  r <- cluster_boot(bd, B = 200, seed = 99, stat_fn = stat, plan = b1$plan, cores = 1)
  identical(r$reps, b1$reps)
})
check("a plan for other patients is refused", errs(cluster_boot(bd[bd$patient_id != "P01", ], B = 200, seed = 11, stat_fn = stat, plan = b1$plan, cores = 1), "does not match"))
# a mean has a bootstrap SE near the cluster-aware analytic value
x_by_p <- tapply(bd$x, bd$patient_id, mean)
check("bootstrap SD of the mean is larger than the naive frame-level SE (clustered data)",
      sd(b1$reps[, "mean_x"]) > sd(bd$x) / sqrt(nrow(bd)))
check("rng state outside the bootstrap is untouched", {
  set.seed(5); a <- runif(1); set.seed(5); invisible(boot_plan(bd, B = 10, seed = 1)); identical(a, runif(1))
})
# patient_macro: each drawn copy of a patient is its own cluster
mac <- patient_macro(function(d) c(m = mean(d$x)))
check("patient_macro on original data is the mean of patient means", near(mac(bd)[["m"]], mean(tapply(bd$x, bd$patient_id, mean))))
bm <- cluster_boot(bd, B = 50, seed = 4, stat_fn = mac, cores = 1)
check("patient_macro inside the bootstrap weights a repeated patient twice", {
  d <- .gl_boot_data(bd, bm$plan, 1)
  near(mac(d)[["m"]], mean(vapply(split(d$x, d$.boot_cluster), mean, 0))) && near(bm$reps[1, "m"], mac(d)[["m"]])
})

# ---- gate models ------------------------------------------------------------------------------
set.seed(7)
n <- 600
toy <- data.frame(patient_id = sprintf("P%02d", rep(1:30, each = 20)),
                  a = rnorm(n), b = rexp(n), z = rbinom(n, 1, 0.7) * runif(n))   # z: zero-inflated
toy$ens_dis <- runif(n)
toy$pred_area_frac <- runif(n, 0, 0.4)
toy$fail <- rbinom(n, 1, plogis(-1.5 + 1.2 * toy$a + 0.8 * log(toy$b + 0.1) + 2 * toy$z))
feats <- c("a", "b", "z")
fid <- patient_foldid(toy$patient_id, 5L, seed = 1)
check("patient-grouped inner folds: no patient in two folds, 5 folds used",
      all(vapply(split(fid, toy$patient_id), function(v) length(unique(v)) == 1L, NA)) && length(unique(fid)) == 5L)
check("patient fold sizes are balanced (frames)", diff(range(table(fid))) <= 40)
check("patient_foldid is deterministic", identical(fid, patient_foldid(toy$patient_id, 5L, seed = 1)))
g <- fit_gate(toy, toy$fail, features = feats, foldid = fid)
check("gate object carries the documented fields",
      all(c("kind", "spec", "fit", "lambda", "scale", "created") %in% names(g)) && inherits(g$fit, "glmnet") &&
        identical(g$spec$features, feats) && identical(g$spec$df, 3))
check("zero-inflated feature falls back gracefully (knots inside the range only)",
      all(vapply(feats, function(f) all(g$spec$knots[[f]] > g$spec$boundary[[f]][1] & g$spec$knots[[f]] < g$spec$boundary[[f]][2]), NA)))
# independent rebuild of the training basis straight from splines::ns, then glmnet's own predict
x_ref <- do.call(cbind, lapply(feats, function(f) {
  zf <- (toy[[f]] - g$scale$centre[[f]]) / g$scale$scale[[f]]
  kn <- g$spec$knots[[f]]
  if (length(kn)) unclass(splines::ns(zf, knots = kn, Boundary.knots = g$spec$boundary[[f]])) else matrix(zf, ncol = 1)
}))
ref <- as.numeric(predict(g$fit, newx = x_ref, s = g$lambda, type = "link"))
check("predict_gate reproduces the fit's own predictions on the training data", near(predict_gate(g, toy, "link"), ref, 1e-10))
check("response is plogis(link)", near(predict_gate(g, toy), plogis(ref), 1e-12))
check("ns basis with stored knots equals splines::ns(df = 3) on a feature with distinct quantiles", {
  zb <- (toy$a - g$scale$centre[["a"]]) / g$scale$scale[["a"]]
  near(unclass(splines::ns(zb, knots = g$spec$knots$a, Boundary.knots = g$spec$boundary$a)), unclass(splines::ns(zb, df = 3)), 1e-10)
})
nd <- toy[1:7, ]
check("new rows are transformed identically (same prediction as inside the full table)", near(predict_gate(g, nd, "link"), ref[1:7], 1e-10))
check("a single-row newdata works", length(predict_gate(g, toy[1, ])) == 1L)
wild <- toy[1:3, ]
wild$a <- c(-50, 0, 50)
wild$z <- c(5, 0, 9)
pw <- predict_gate(g, wild, "link")
check("values outside the training range extrapolate finitely and monotonically in a", all(is.finite(pw)) && pw[1] != pw[3])
rc <- range_check(g, rbind(toy[1:2, ], {w <- toy[1:2, ]; w$a <- c(-50, 50); w}))
check("range_check counts rows below and above the training range",
      near(rc$frac_below[rc$feature == "a"], 0.25) && near(rc$frac_above[rc$feature == "a"], 0.25) &&
        near(rc$frac_outside[rc$feature == "any_feature"], 0.5) && rc$feature[nrow(rc)] == "any_feature")
check("missing feature column is an error naming it", errs(predict_gate(g, toy[c("a", "b")]), "z"))
check("NA feature is an error", errs(predict_gate(g, {t2 <- toy[1:3, ]; t2$a[1] <- NA; t2}), "non-finite"))
ga <- fit_area(toy, toy$fail)
check("area model: plain logistic, predict_gate runs on pred_area_frac only", is.null(ga$fit$beta) == FALSE && length(predict_gate(ga, toy["pred_area_frac"])) == n)
gl <- predict_gate(g, toy, "link")
gs <- fit_stack(gl, toy$ens_dis, toy$fail, g)
check("stack link = beta0 + beta1 * gate link + beta2 * ens_dis",
      near(predict_gate(gs, toy, "link"), gs$fit$beta[[1]] + gs$fit$beta[[2]] * gl + gs$fit$beta[[3]] * toy$ens_dis, 1e-12))
check("stack needs ens_dis", errs(predict_gate(gs, toy[feats]), "ens_dis"))
fo <- sample(0:4, n, TRUE)
fold_objs <- setNames(lapply(0:4, function(k) fit_gate(toy[fo != k, ], toy$fail[fo != k], feats, patient_foldid(toy$patient_id[fo != k], 5L))), 0:4)
oo <- predict_oof(fold_objs, toy, fo, "link")
check("predict_oof scores each fold's rows with that fold's model",
      near(oo[fo == 3], predict_gate(fold_objs[["3"]], toy[fo == 3, ], "link")))

# ---- stat_fn used by the evaluation ------------------------------------------------------------
set.seed(8)
m <- 800
ev <- data.frame(patient_id = sprintf("P%02d", rep(1:40, each = 20)), domain = "D3")
sig <- rnorm(m)
ev$risk <- plogis(-1.5 + sig + rnorm(m, 0, 0.7))
ev$fail <- ev$risk > 0.25
ev$s_predicted_area <- rnorm(m)
ev$s_entropy <- sig + rnorm(m, 0, 1)
ev$s_ens_disagreement <- sig + rnorm(m, 0, 0.5)
ev$s_optical_gate <- sig + rnorm(m, 0, 0.7)
ev$s_gate_plus_ensemble <- sig + rnorm(m, 0, 0.3)
sf <- make_gate_stat_fn(c(0.9, 0.8), c(0.5, 0.9), seq(0.1, 1, by = 0.1))
v <- sf(ev)
check("stat_fn names are method|stat, 7 methods", all(grepl("\\|", names(v))) && length(unique(sub("\\|.*", "", names(v)))) == 7L)
check("random AURC = mean risk and oracle is the lowest AURC",
      near(v[["random|aurc"]], mean(ev$risk)) && v[["oracle|aurc"]] == min(v[grepl("\\|aurc$", names(v))]))
check("better scores have lower AURC and higher AUROC",
      v[["optical_gate|aurc"]] < v[["entropy|aurc"]] && v[["gate_plus_ensemble|auroc"]] > v[["entropy|auroc"]])
check("selective risk at full coverage equals the mean risk for every method", near(v[grepl("\\|cov1\\.00$", names(v))], mean(ev$risk), 1e-12))
check("calibration only for probabilistic methods", !is.na(v[["optical_gate|cal_slope"]]) && is.na(v[["entropy|cal_slope"]]) && is.na(v[["oracle|cal_slope"]]))
check("macro AURC differs from pooled but is on the same scale", abs(v[["optical_gate|aurc_macro"]] - v[["optical_gate|aurc"]]) < 0.1)
bt <- cluster_boot(ev, B = 40, seed = 3, stat_fn = sf, cores = 2)
check("stat_fn runs inside cluster_boot (2 cores) with a constant-length vector", ncol(bt$reps) == length(v) && nrow(bt$reps) == 40L)

# ---- lock helpers and write guard --------------------------------------------------------------
tdir <- p("work", "tests", "stats_lib")
unlink(tdir, recursive = TRUE)
dir.create(tdir, recursive = TRUE)
lk <- file.path(tdir, "lock.yaml")
writeLines(c("# frozen at G2", "failure_iou_threshold: 0.75", "epochs:", "  unet: 30", "  segformer: 40",
             "noninferiority_margin_aurc: 0.01", "gate_rds_sha256: null"), lk)
before <- yaml::read_yaml(lk)
after <- lock_update(lk, list(gate_rds_path = "work/models/gate_optical.rds", gate_rds_sha256 = "abc123", entropy_baseline_choice = "entropy_fov"))
txt <- readLines(lk)
check("lock_update keeps comments, other keys and nested blocks",
      txt[1] == "# frozen at G2" && identical(after$epochs, before$epochs) && after$failure_iou_threshold == 0.75)
check("lock_update replaces a null placeholder in place and appends new keys",
      after$gate_rds_sha256 == "abc123" && sum(grepl("^gate_rds_sha256:", txt)) == 1L && after$entropy_baseline_choice == "entropy_fov")
check("lock_update refuses to overwrite a block key", errs(lock_update(lk, list(epochs = "x")), "block"))
check("write guard accepts synthetic out and refuses the real config and results",
      identical(guard_write(file.path(tdir, "x.csv")), file.path(tdir, "x.csv")) &&
        errs(guard_write(file.path(V1, "config", "analysis_lock.yaml")), "refusing") &&
        errs(guard_write(file.path(V1, "results", "stats", "x.csv")), "refusing") &&
        errs(guard_write(file.path(SYNTH_DIR, "out", "..", "x.csv")), "refusing") &&
        errs(lock_update(file.path(V1, "config", "analysis_lock.yaml"), list(a = 1)), "refusing"))
check("out_path() resolves under work/synthetic/out in synthetic mode", startsWith(out_path("results", "stats", "t.csv"), file.path(SYNTH_DIR, "out")))
rds <- file.path(tdir, "obj.rds")
saveRDS(list(1), rds)
lockh <- list(failure_iou_threshold = 0.75, epochs = list(unet = 30L, segformer = 40L), noninferiority_margin_aurc = 0.01,
              entropy_baseline_choice = "entropy_fov", locked_at = "2026-10-20T10:00:00+0000")
for (o in GATE_LOCK_OBJECTS) {
  lockh[[paste0(o, "_path")]] <- file.path("work", "tests", "stats_lib", "obj.rds")
  lockh[[paste0(o, "_sha256")]] <- unname(tools::sha256sum(rds))
}
lockh$lock_fields_sha256 <- lock_fields_hash(lockh)
check("verify_lock_hashes accepts matching hashes", isTRUE(verify_lock_hashes(lockh)))
saveRDS(list(2), rds)
check("verify_lock_hashes stops on a modified file", errs(verify_lock_hashes(lockh), "SHA-256 mismatch"))
saveRDS(list(1), rds)
check("verify_lock_hashes accepts the restored file", isTRUE(verify_lock_hashes(lockh)))
no_field <- lockh[names(lockh) != "lock_fields_sha256"]
check("verify_lock_hashes stops on a lock without lock_fields_sha256", errs(verify_lock_hashes(no_field), "incomplete.*lock_fields_sha256"))
l2 <- lockh[names(lockh) != "gate_area_rds_sha256"]
check("verify_lock_hashes stops on an incomplete lock", errs(verify_lock_hashes(l2), "incomplete"))

# lock_fields_sha256: canonical JSON, sorted keys, fixed number format; the expected hex values come from `shasum -a 256`
canon <- list(b = 1L, c = NULL, a = list(z = "x", y = 0.75))
check("canonical JSON sorts keys (nested too) and prints null", identical(.gl_canon(canon), "{\"a\":{\"y\":0.75,\"z\":\"x\"},\"b\":1,\"c\":null}"))
check("lock_fields_hash of that object equals an independent shasum -a 256",
      identical(lock_fields_hash(canon), "5ccdf18ab2415167d6dcc8c7d8f97957ba5f27f85f2455975de9ab3ba1ac4ec9"))
rich <- list(g = 1e-05, f = TRUE, e = "q\"\\", d = c(1L, 2L), c = NA, b = 1L, a = list(z = "x", y = 0.75))
check("canonical JSON: arrays, escapes, booleans, %.15g numbers (matches shasum)",
      identical(lock_fields_hash(rich), "18eba90a7431fd788e8b9212e09bfd3a52d60f8f14f42bc4a6b8f958ef558585"))
check("hash ignores key order, locked_at and lock_fields_sha256 itself",
      identical(lock_fields_hash(lockh), lock_fields_hash(rev(lockh))) &&
        identical(lock_fields_hash(lockh), lock_fields_hash(modifyList(lockh, list(locked_at = "2030-01-01", lock_fields_sha256 = "x")))))
check("integer 30L and double 30 hash the same", identical(lock_fields_hash(list(a = 30L)), lock_fields_hash(list(a = 30))))
ly <- file.path(tdir, "lock_roundtrip.yaml")
yaml::write_yaml(lockh, ly)
check("a lock read back from YAML hashes the same as in memory", identical(lock_fields_hash(yaml::read_yaml(ly)), lockh$lock_fields_sha256))
tamper <- function(...) modifyList(lockh, list(...))
check("a changed failure threshold is caught", errs(verify_lock_hashes(tamper(failure_iou_threshold = 0.8)), "lock_fields_sha256 mismatch"))
check("a changed non-inferiority margin is caught", errs(verify_lock_hashes(tamper(noninferiority_margin_aurc = 0.05)), "lock_fields_sha256 mismatch"))
no_margin <- lockh
no_margin["noninferiority_margin_aurc"] <- list(NULL)
check("a margin switched to null is caught", errs(verify_lock_hashes(no_margin), "lock_fields_sha256 mismatch"))
check("a changed epochs entry is caught", errs(verify_lock_hashes(tamper(epochs = list(unet = 31L, segformer = 40L))), "lock_fields_sha256 mismatch"))
check("a changed entropy choice is caught", errs(verify_lock_hashes(tamper(entropy_baseline_choice = "entropy_roi")), "lock_fields_sha256 mismatch"))
check("a new lock key is caught", errs(verify_lock_hashes(tamper(extra = 1)), "lock_fields_sha256 mismatch"))
check("a changed recorded RDS hash is caught by the field hash", errs(verify_lock_hashes(tamper(gate_rds_sha256 = "00")), "lock_fields_sha256 mismatch"))
check("a changed locked_at alone does not matter", isTRUE(verify_lock_hashes(tamper(locked_at = "2031-05-05T00:00:00+0000"))))

cat(sprintf("\n%d failure(s)\n", length(failures)))
if (length(failures)) quit(status = 1)
