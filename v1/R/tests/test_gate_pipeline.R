# End-to-end tests of 04a_gate_fit_lock.R, 04b_external_eval.R and 05_transfer.R on the synthetic data.
#   Rscript v1/R/tests/test_gate_pipeline.R        (re-runs itself with OPTICS_DATA=synthetic)
# Everything is written under work/synthetic/out/work/tests/gate_pipeline (the default synthetic lock,
# models and results are not touched). The synthetic inputs are read-only.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
r_dir <- dirname(.here)
for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R", "30_bundle.R",
            "04a_gate_fit_lock.R", "04b_external_eval.R", "05_transfer.R")) source(file.path(r_dir, f))

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
B_TEST <- 300L
CORES <- 2L
base <- file.path("work", "tests", "gate_pipeline")
unlink(p(base), recursive = TRUE)
mk_paths <- function(tag, derived = p("derived")) {
  list(derived = derived, models_rel = file.path(base, tag, "models"), lock = p(base, tag, "analysis_lock.yaml"),
       stats = p(base, tag, "stats"), results = p(base, tag, "root"))
}
file_sha <- function(path) unname(tools::sha256sum(path))

# real files that must not change while OPTICS_DATA=synthetic
real_lock <- file.path(V1, "config", "analysis_lock.yaml")
real_dirs <- c(file.path(V1, "results"), file.path(V1, "figures"), file.path(V1, "tables"), file.path(V1, "work", "models"))
snap_real <- function() list(lock = file.exists(real_lock), files = unlist(lapply(real_dirs, function(d) list.files(d, recursive = TRUE))))
real_before <- snap_real()

# ---- 04a ---------------------------------------------------------------------------------------
pa <- mk_paths("a")
dir.create(dirname(pa$lock), recursive = TRUE)
g2_lock_lines <- c("# frozen at G2", "failure_iou_threshold: 0.75", "epochs:", "  unet: 30", "  segformer: 40",
                   "noninferiority_margin_aurc: 0.01", "lock_fields_sha256: \"deadbeef\"")
writeLines(g2_lock_lines, pa$lock)
lock0 <- yaml::read_yaml(pa$lock)

t_a <- system.time(sm <- quiet(fit_and_lock(pa)))[["elapsed"]]
lock1 <- yaml::read_yaml(pa$lock)
n_c1 <- quiet({
  ds <- arrow::open_dataset(file.path(pa$derived, "predictions.parquet"))
  ds |> dplyr::filter(partition == "center1_labeled", model == "unet") |> dplyr::count() |> dplyr::collect() |> dplyr::pull(n)
})
n_all <- quiet(nrow(arrow::open_dataset(file.path(pa$derived, "predictions.parquet"))))
# synthetic predictions still contain frames the FOV detector missed; independent count of those per partition
nf_ids <- quiet(arrow::read_parquet(file.path(pa$derived, "fov.parquet")))$frame_id[!quiet(arrow::read_parquet(file.path(pa$derived, "fov.parquet")))$fov_found]
pr_all <- quiet(arrow::read_parquet(file.path(pa$derived, "predictions.parquet")))
n_nf_c1 <- sum(pr_all$partition == "center1_labeled" & pr_all$model == "unet" & pr_all$frame_id %in% nf_ids)
n_nf_c2 <- sum(pr_all$partition == "center2_labeled" & pr_all$model == "unet" & pr_all$frame_id %in% nf_ids)

check("04a loads only Center 1 U-Net rows: rows touched == Center 1 rows, fewer than the file",
      sm$rows_loaded == n_c1 && n_c1 < n_all && sm$rows_in_file == n_all && identical(sm$partitions, "center1_labeled"))
check("04a: 61 Center 1 patients; frames without a detected field of view are excluded and counted",
      sm$n_patients == 61L && n_nf_c1 > 0L && sm$n_excluded_fov == n_nf_c1 && sm$n_frames == n_c1 - n_nf_c1 &&
        !any(sm$frame_id %in% nf_ids))
own <- setdiff(names(lock0), "lock_fields_sha256")   # 04a owns lock_fields_sha256 (the placeholder is replaced)
check("lock keeps every key it did not own (comment, nested epochs, margin)",
      identical(lock1[own], lock0[own]) && readLines(pa$lock)[1] == "# frozen at G2")
check("lock_fields_sha256 replaces the placeholder and equals the hash of every other key but locked_at",
      lock1$lock_fields_sha256 != "deadbeef" && .gl_nonempty(lock1$lock_fields_sha256) && sum(grepl("^lock_fields_sha256:", readLines(pa$lock))) == 1L &&
        identical(lock1$lock_fields_sha256, lock_fields_hash(lock1)) &&
        identical(lock1$lock_fields_sha256, lock_fields_hash(modifyList(lock1, list(locked_at = "x")))) &&
        !identical(lock1$lock_fields_sha256, lock_fields_hash(modifyList(lock1, list(entropy_baseline_choice = "other")))))
check("lock records path and sha256 for the gate and the three comparator objects",
      all(vapply(GATE_LOCK_OBJECTS, function(o) .gl_nonempty(lock1[[paste0(o, "_path")]]) && .gl_nonempty(lock1[[paste0(o, "_sha256")]]), NA)) &&
        .gl_nonempty(lock1$locked_at) && lock1$entropy_baseline_choice %in% c("entropy_fov", "entropy_roi"))
check("recorded hashes equal the files' SHA-256", all(vapply(GATE_LOCK_OBJECTS, function(o)
  identical(file_sha(p(lock1[[paste0(o, "_path")]])), lock1[[paste0(o, "_sha256")]]), NA)) && isTRUE(verify_lock_hashes(lock1)))
check("objects and lock were written under work/synthetic/out, not the real config or models",
      startsWith(p(lock1$gate_rds_path), file.path(SYNTH_DIR, "out")) && startsWith(pa$lock, file.path(SYNTH_DIR, "out")))
check("entropy choice is the one with the lower Center 1 OOF AURC", lock1$entropy_baseline_choice == names(which.min(sm$entropy_aurc)))

gate <- readRDS(p(lock1$gate_rds_path))
d1 <- read_eval_frames(pa$derived, "center1_labeled", "unet")
d1$fail <- d1$iou < 0.75
check("gate object follows the documented schema",
      all(c("kind", "spec", "fit", "lambda", "scale", "created") %in% names(gate)) && gate$kind == "optical_gate" &&
        inherits(gate$fit, "glmnet") && identical(gate$spec$features, OPTICAL_METRICS) && gate$alpha == 0.5 && gate$spec$df == 3)
check("predict_gate returns probabilities for Center 1 and equals plogis(link)",
      { pr <- predict_gate(gate, d1); all(pr > 0 & pr < 1) && near(pr, plogis(predict_gate(gate, d1, "link")), 1e-12) })
beta <- as.matrix(coef(gate$fit, s = gate$lambda))[, 1]
check("planted metrics (sharpness, under-exposure, haze) are selected by the penalised fit",
      all(vapply(c("sharp_tenengrad", "under_frac", "haze_dcp"), function(m) any(beta[startsWith(names(beta), paste0(m, "_"))] != 0), NA)))
check("flag thresholds are Center 1 OOF quantiles at the coverage levels",
      identical(names(gate$flag_thresholds), c("0.9", "0.8")) &&
        near(unname(gate$flag_thresholds), unname(stats::quantile(sm$oof_link, c(0.9, 0.8), type = 1))))
check("OOF gate beats random ordering on Center 1 (AURC)",
      aurc(sm$oof_link, 1 - d1$iou[match(sm$frame_id, d1$frame_id)]) < mean(1 - d1$iou) - 0.02)

folds_obj <- readRDS(p(lock1$gate_folds_rds_path))
check("fold objects: five gates, five area models, five stacks, each stack embeds its fold's gate",
      identical(sort(names(folds_obj)), c("gate_plus_ensemble", "optical_gate", "predicted_area")) &&
        all(lengths(folds_obj) == 5L) && identical(folds_obj$gate_plus_ensemble[["2"]]$base$created, folds_obj$optical_gate[["2"]]$created))

# a held-out fold's gate never saw that fold: refitting on the other folds reproduces it, the final gate differs
ref2 <- .gate_on(d1[d1$fold != 2L, ])
check("fold-2 gate equals an independent refit on the other four folds and differs from the final gate",
      identical(predict_gate(ref2, d1, "link"), predict_gate(folds_obj$optical_gate[["2"]], d1, "link")) &&
        !near(predict_gate(folds_obj$optical_gate[["2"]], d1, "link"), predict_gate(gate, d1, "link"), 1e-6))

# ---- frames without a detected FOV carry NaN optics: exclusion runs before the missing-value check ----------------
nan_dir <- p(base, "nan_derived")
dir.create(nan_dir, recursive = TRUE)
op_nan <- quiet(arrow::read_parquet(file.path(pa$derived, "optics.parquet")))
op_nan[op_nan$frame_id %in% nf_ids, OPTICAL_METRICS] <- NaN
quiet(arrow::write_parquet(op_nan, file.path(nan_dir, "optics.parquet")))
invisible(file.copy(file.path(pa$derived, c("predictions.parquet", "folds.csv", "fov.parquet")), nan_dir))
nan_c1 <- quiet(read_eval_frames(nan_dir, "center1_labeled", "unet"))
check("unfound frames with NaN optics and a predictions row are dropped and counted instead of stopping the read",
      n_nf_c1 > 0L && attr(nan_c1, "n_excluded_fov") == n_nf_c1 && nrow(nan_c1) == n_c1 - n_nf_c1 && !anyNA(nan_c1[[OPTICAL_METRICS[1]]]))
nan_found <- op_nan
c1_found <- pr_all$frame_id[pr_all$partition == "center1_labeled" & pr_all$model == "unet" & !pr_all$frame_id %in% nf_ids]
nan_found[nan_found$frame_id == c1_found[1], OPTICAL_METRICS] <- NaN
quiet(arrow::write_parquet(nan_found, file.path(nan_dir, "optics.parquet")))
check("NaN optics on a frame whose FOV WAS found still stops the read",
      errs(quiet(read_eval_frames(nan_dir, "center1_labeled", "unet")), "optics or domain missing"))

# ---- Center 2 never influences 04a -------------------------------------------------------------
pois_dir <- p(base, "poison_derived")
dir.create(pois_dir, recursive = TRUE)
pred <- quiet(arrow::read_parquet(file.path(pa$derived, "predictions.parquet")))
set.seed(99)
c2 <- pred$partition == "center2_labeled"
pred$iou[c2] <- runif(sum(c2))
pred$ens_dis[c2] <- runif(sum(c2))
pred$entropy_fov[c2] <- runif(sum(c2))
pred$pred_area_frac[c2] <- runif(sum(c2))
quiet(arrow::write_parquet(pred, file.path(pois_dir, "predictions.parquet")))
invisible(file.copy(file.path(pa$derived, c("optics.parquet", "folds.csv", "fov.parquet")), pois_dir))
pp <- mk_paths("poison", pois_dir)
# same lock header as run a: without a lock the run falls back to the plan (threshold 0.90, no margin) and fits a different gate
dir.create(dirname(pp$lock), recursive = TRUE)
writeLines(g2_lock_lines, pp$lock)
invisible(quiet(fit_and_lock(pp)))
g_pois <- readRDS(p(yaml::read_yaml(pp$lock)$gate_rds_path))
s_pois <- readRDS(p(yaml::read_yaml(pp$lock)$gate_stack_rds_path))
check("scrambling every Center 2 outcome and baseline signal leaves the gate and stack predictions bit-identical",
      identical(predict_gate(g_pois, d1, "link"), predict_gate(gate, d1, "link")) &&
        identical(predict_gate(s_pois, d1, "link"), predict_gate(readRDS(p(lock1$gate_stack_rds_path)), d1, "link")))

# ---- re-lock protection -------------------------------------------------------------------------
before_txt <- readLines(pa$lock)
check("a second run is refused while gate_rds_sha256 is recorded, and the lock is untouched",
      errs(quiet(fit_and_lock(pa)), "OPTICS_RELOCK") && identical(readLines(pa$lock), before_txt))
invisible(quiet(fit_and_lock(pa, relock = TRUE)))
lock2 <- yaml::read_yaml(pa$lock)
check("OPTICS_RELOCK permits a re-lock; it keeps the other keys, reseals the lock and gives the same fit",
      identical(lock2[own], lock0[own]) && isTRUE(verify_lock_hashes(lock2)) &&
        identical(predict_gate(readRDS(p(lock2$gate_rds_path)), d1, "link"), predict_gate(gate, d1, "link")))
check("the real lock file and result folders were not touched", identical(snap_real(), real_before))

# ---- 04b ---------------------------------------------------------------------------------------
t_b <- system.time(res <- quiet(run_external_eval(pa, B_TEST, CORES)))[["elapsed"]]
tb_int <- res$internal$table
tb_ext <- res$external$table
rd <- function(f) utils::read.csv(file.path(pa$stats, f))
cols <- c("partition", "model", "method", "n_frames", "n_patients", "n_events", "n_patients_with_events", "binary_descriptive",
          unlist(lapply(TABLE_STATS, function(s) c(s, paste0(s, c("_lo", "_hi"))))))
check("gate_internal.csv and gate_external.csv have the documented columns, 7 methods",
      identical(names(rd("gate_internal.csv")), cols) && identical(names(rd("gate_external.csv")), cols) &&
        identical(rd("gate_external.csv")$method, GATE_METHODS) && all(rd("gate_external.csv")$partition == "external"))
check("gate_range_check.csv: one row per feature plus any_feature, per partition",
      identical(names(rd("gate_range_check.csv")), c("partition", "feature", "frac_below", "frac_above", "frac_outside")) &&
        nrow(rd("gate_range_check.csv")) == 24L)
check("gate_paired_diffs.csv columns and rows (2 partitions x 5 comparisons x 3 metrics)",
      identical(names(rd("gate_paired_diffs.csv")), c("partition", "model", "comparison", "a", "b", "metric", "diff", "lo", "hi", "margin", "noninferior")) &&
        nrow(rd("gate_paired_diffs.csv")) == 30L)
cv <- rd("risk_coverage_curves.csv")
check("risk_coverage_curves.csv: 7 methods x 100 coverage points x 2 partitions with bands",
      identical(names(cv), c("partition", "model", "method", "coverage", "selective_risk", "lo", "hi")) && nrow(cv) == 1400L &&
        all(cv$lo <= cv$selective_risk + 1e-12 & cv$selective_risk <= cv$hi + 1e-12))
ext <- read_eval_frames(pa$derived, "center2_labeled", "unet")
int <- d1
check("external rows are read only by 04b, after the hash check: 440 Center 2 rows read, fov exclusions dropped, 30+ failures in 10+ patients",
      res$external$rows_loaded == 440L && res$external$n_excluded_fov == n_nf_c2 && tb_ext$n_frames[1] == 440L - n_nf_c2 && tb_ext$n_events[1] >= 30 && tb_ext$n_patients_with_events[1] >= 10 && !tb_ext$binary_descriptive[1])
check("external failures exist (planted: about 40%)", tb_ext$n_events[1] / tb_ext$n_frames[1] > 0.25)
check("random AURC = mean risk and oracle AURC = analytic value, both partitions",
      near(tb_ext$aurc[tb_ext$method == "random"], mean(1 - ext$iou)) && near(tb_ext$aurc[tb_ext$method == "oracle"], oracle_aurc(1 - ext$iou)) &&
        near(tb_int$aurc[tb_int$method == "random"], mean(1 - int$iou)) && near(tb_int$aurc[tb_int$method == "oracle"], oracle_aurc(1 - int$iou)))
a <- setNames(tb_ext$aurc, tb_ext$method)
check("planted effect recovered externally: gate AURC below random, above oracle",
      a[["optical_gate"]] < a[["random"]] - 0.05 && a[["optical_gate"]] > a[["oracle"]])
ai <- setNames(tb_int$aurc, tb_int$method)
check("internal: gate AURC below random; gate is comparable with entropy and below predicted area",
      ai[["optical_gate"]] < ai[["random"]] - 0.03 && abs(ai[["optical_gate"]] - ai[["entropy"]]) < 0.02 && ai[["optical_gate"]] < ai[["predicted_area"]] + 0.005)
check("gate + ensemble improves on the ensemble alone, internal and external (point estimates)",
      ai[["gate_plus_ensemble"]] < ai[["ens_disagreement"]] && a[["gate_plus_ensemble"]] < a[["ens_disagreement"]])
check("AUROC of the gate external is clearly above 0.5 with a CI above 0.5",
      tb_ext$auroc[tb_ext$method == "optical_gate"] > 0.7 && tb_ext$auroc_lo[tb_ext$method == "optical_gate"] > 0.5)
check("random: AUROC 0.5, no interval; oracle AUROC is 1",
      tb_ext$auroc[1] == 0.5 && is.na(tb_ext$auroc_lo[1]) && near(tb_ext$auroc[7], 1))
check("calibration only for probabilistic methods", all(!is.na(tb_ext$cal_slope[tb_ext$method %in% GATE_PROB_METHODS])) &&
        all(is.na(tb_ext$cal_slope[!tb_ext$method %in% GATE_PROB_METHODS])))
check("all intervals bracket their estimates", {
  ok <- function(tb) all(unlist(lapply(setdiff(TABLE_STATS, character()), function(s) {
    e <- tb[[s]]; lo <- tb[[paste0(s, "_lo")]]; hi <- tb[[paste0(s, "_hi")]]
    is.na(lo) | is.na(e) | (lo <= e + 1e-9 & e <= hi + 1e-9)
  })))
  ok(tb_int) && ok(tb_ext)
})
dx <- res$external$diffs
dg <- dx[dx$comparison == "gate_vs_random" & dx$metric == "aurc", ]
dgr <- dx[dx$comparison == "gate_vs_entropy" & dx$metric == "aurc", ]
check("paired difference gate - random: estimate = difference of the table AURCs, CI below zero",
      near(dg$diff, a[["optical_gate"]] - a[["random"]]) && dg$hi < 0)
b <- res$external$boot
check("paired CI is computed on the shared resamples (equals quantiles of reps[, gate] - reps[, entropy])",
      near(dgr$lo, unname(quantile(b$reps[, "optical_gate|aurc"] - b$reps[, "entropy|aurc"], 0.025))) &&
        near(dgr$hi, unname(quantile(b$reps[, "optical_gate|aurc"] - b$reps[, "entropy|aurc"], 0.975))))
check("margin from the lock enters the non-inferiority flag only for gate vs entropy / vs ensemble AURC",
      all(dx$margin == 0.01) && all(!is.na(dx$noninferior[dx$metric == "aurc" & dx$comparison %in% c("gate_vs_entropy", "gate_vs_ens")])) &&
        all(is.na(dx$noninferior[!(dx$metric == "aurc" & dx$comparison %in% c("gate_vs_entropy", "gate_vs_ens"))])) &&
        identical(dgr$noninferior, dgr$hi < 0.01))

# repeating the evaluation with another core count gives identical numbers (plan is drawn serially)
res_b <- quiet(evaluate_scored(ext, score_methods(load_gate_models(lock2), ext, lock2$entropy_baseline_choice, FALSE),
                               "external", "unet", 0.75, B_TEST, cores = 1, margin = 0.01))
check("same seed gives the same table whatever the core count", isTRUE(all.equal(res_b$table, tb_ext, tolerance = 0)))

# few events -> binary metrics are descriptive only; a null margin switches the non-inferiority flag off
few <- ext[ext$patient_id %in% unique(ext$patient_id)[1:6], ]
res_few <- quiet(evaluate_scored(few, score_methods(load_gate_models(lock2), few, lock2$entropy_baseline_choice, FALSE),
                                 "external", "unet", 0.75, 100L, cores = 1, margin = NULL))
check("descriptive-only flag is raised below the event or patient minimum; null margin leaves noninferior NA",
      res_few$table$binary_descriptive[1] && all(is.na(res_few$diffs$noninferior)) && all(is.na(res_few$diffs$margin)))

# ---- tamper test -------------------------------------------------------------------------------
gate_file <- p(lock2$gate_rds_path)
orig_bytes <- readBin(gate_file, "raw", file.size(gate_file))
g_mod <- readRDS(gate_file)
g_mod$lambda <- g_mod$lambda * 1.5
saveRDS(g_mod, gate_file)
pt <- pa
pt$stats <- p(base, "tamper_stats")
pt$derived <- p(base, "no_such_dir")   # a hash check that runs first never reaches the missing data
e1 <- tryCatch(quiet(run_external_eval(pt, 50L, 1L)), error = function(e) conditionMessage(e))
e2 <- tryCatch(quiet(run_transfer(pt, 50L, 1L)), error = function(e) conditionMessage(e))
check("04b stops with a SHA-256 mismatch after the gate RDS is modified, before any data are read",
      is.character(e1) && grepl("SHA-256 mismatch", e1) && grepl("gate_optical.rds", e1))
check("05 stops the same way", is.character(e2) && grepl("SHA-256 mismatch", e2))
check("nothing was written by the refused runs", !dir.exists(pt$stats))
check("04a's re-lock guard also sees the modified file is not the recorded one", !identical(file_sha(gate_file), lock2$gate_rds_sha256))
writeBin(orig_bytes, gate_file)
fold_file <- p(lock2$gate_folds_rds_path)
fold_bytes <- readBin(fold_file, "raw", file.size(fold_file))
saveRDS(list(), fold_file)
check("modifying the fold-model RDS is caught as well", errs(quiet(run_external_eval(pt, 50L, 1L)), "SHA-256 mismatch.*gate_folds"))
writeBin(fold_bytes, fold_file)
check("restored files verify again", isTRUE(verify_lock_hashes(yaml::read_yaml(pa$lock))))
pi <- pa
pi$lock <- p(base, "incomplete.yaml")
writeLines(grep("^gate_stack_rds_sha256:", readLines(pa$lock), value = TRUE, invert = TRUE), pi$lock)
check("a lock without all hashes is refused", errs(run_external_eval(pi, 50L, 1L), "incomplete"))

# changing a frozen field after locking: 04b and 05 stop on lock_fields_sha256 before any data are read
lock_lines <- readLines(pa$lock)
tampered <- function(tag, pattern, replacement) {
  pt2 <- pt
  pt2$lock <- p(base, paste0("tamper_", tag, ".yaml"))
  edited <- sub(pattern, replacement, lock_lines)
  stopifnot(!identical(edited, lock_lines))
  writeLines(edited, pt2$lock)
  pt2
}
for (case in list(list("threshold", "^failure_iou_threshold: .*", "failure_iou_threshold: 0.5"),
                  list("margin", "^noninferiority_margin_aurc: .*", "noninferiority_margin_aurc: 0.5"),
                  list("margin_null", "^noninferiority_margin_aurc: .*", "noninferiority_margin_aurc: null"),
                  list("epochs", "^  unet: 30", "  unet: 31"),
                  list("entropy", "^entropy_baseline_choice: .*", "entropy_baseline_choice: \"entropy_fov_x\""))) {
  pc <- do.call(tampered, case)
  e_b <- tryCatch(quiet(run_external_eval(pc, 50L, 1L)), error = function(e) conditionMessage(e))
  e_t <- tryCatch(quiet(run_transfer(pc, 50L, 1L)), error = function(e) conditionMessage(e))
  check(paste0("a lock edited after locking (", case[[1]], ") stops 04b and 05 before any data are read"),
        is.character(e_b) && grepl("lock_fields_sha256 mismatch", e_b) && is.character(e_t) && grepl("lock_fields_sha256 mismatch", e_t))
}
check("the refused tampered runs wrote nothing", !dir.exists(pt$stats))
e_ok <- tryCatch(quiet(run_external_eval(pt, 50L, 1L)), error = function(e) conditionMessage(e))
check("the untouched lock passes verification: the same run then fails only on the missing data directory",
      isTRUE(verify_lock_hashes(yaml::read_yaml(pa$lock))) && is.character(e_ok) && !grepl("lock_fields_sha256|SHA-256|incomplete", e_ok))

# ---- 05 transfer -------------------------------------------------------------------------------
t_c <- system.time(tr <- quiet(run_transfer(pa, B_TEST, CORES)))[["elapsed"]]
tt <- utils::read.csv(file.path(pa$stats, "gate_transfer.csv"))
sf_int <- read_eval_frames(pa$derived, "center1_labeled", "segformer")
sf_ext <- read_eval_frames(pa$derived, "center2_labeled", "segformer")
check("gate_transfer.csv: same columns as 04b, model segformer, internal and external rows",
      identical(names(tt), cols) && all(tt$model == "segformer") && identical(unique(tt$partition), c("internal", "external")) && nrow(tt) == 14L)
check("transfer risk is 1 - SegFormer IoU: random AURC = mean SegFormer risk (internal and external)",
      near(tt$aurc[tt$partition == "internal" & tt$method == "random"], mean(1 - sf_int$iou)) &&
        near(tt$aurc[tt$partition == "external" & tt$method == "random"], mean(1 - sf_ext$iou)))
check("SegFormer fails more often than the U-Net (planted offset)", tr$external$table$n_events[1] > tb_ext$n_events[1])
tg <- setNames(tr$external$table$aurc, tr$external$table$method)
check("the U-Net-trained gate transfers: external AURC below random, AUROC above 0.7",
      tg[["optical_gate"]] < tg[["random"]] - 0.05 && tr$external$table$auroc[tr$external$table$method == "optical_gate"] > 0.7)
check("transfer uses the locked gate unchanged: its scores equal predict_gate on the SegFormer frames' optics",
      near(score_methods(load_gate_models(lock2), sf_ext, lock2$entropy_baseline_choice, FALSE)$s_optical_gate,
           predict_gate(readRDS(p(lock2$gate_rds_path)), sf_ext, "link")))
check("transfer paired diffs and curves have the 04b layout",
      identical(names(rd("gate_transfer_paired_diffs.csv")), names(rd("gate_paired_diffs.csv"))) &&
        identical(names(rd("risk_coverage_curves_transfer.csv")), names(cv)) && nrow(rd("risk_coverage_curves_transfer.csv")) == 1400L)

# ---- registry and bundle ------------------------------------------------------------------------
reg_reset()
register_04a(sm)
write_results_part("04a_gate_fit_lock", root = pa$results)
register_04b(res, B_TEST)
write_results_part("04b_external_eval", root = pa$results)
register_05(tr)
write_results_part("05_transfer", root = pa$results)
bund <- bundle_results(pa$results)
rj <- read_results(pa$results)
check("04a, 04b and 05 register disjoint keys and the bundle builds (no duplicate keys)", bund$n_keys > 150L && bund$n_parts == 3L)
check("registered displays are formatted: AURC with a 95% CI at 3 digits, rates as percent",
      grepl("^0\\.[0-9]{3} \\(95% CI 0\\.[0-9]{3} to 0\\.[0-9]{3}\\)$", rj$gate_ext_aurc_optical_gate_ci) &&
        grepl("%$", rj$gate_ext_failure_rate) && rj$gate_entropy_choice %in% c("whole field of view", "region around the predicted mask"))
check("raw values in the bundle equal the CSV values",
      near(rj[["raw.gate_ext_aurc_optical_gate"]], a[["optical_gate"]], 1e-12) && near(rj[["raw.gate_ext_n_frames"]], 440 - n_nf_c2))
check("spline df and elastic-net alpha are registered from the stored gate object",
      sm$spline_df == gate$spec$df && sm$elastic_alpha == gate$alpha && rj$gate_spline_df == "3" && rj$gate_elastic_alpha == "0.5")
check("noninferiority keys are present because the lock carries a margin", !is.null(rj$gate_noninf_margin) && !is.null(rj$gate_ext_noninferior_gate_vs_entropy))
check("the real lock file and result folders are still untouched", identical(snap_real(), real_before))

cat(sprintf("\nruntime (synthetic, B = %d): 04a %.1f s, 04b %.1f s, 05 %.1f s\n", B_TEST, t_a, t_b, t_c))
cat(sprintf("%d failure(s)\n", length(failures)))
if (length(failures)) quit(status = 1)
