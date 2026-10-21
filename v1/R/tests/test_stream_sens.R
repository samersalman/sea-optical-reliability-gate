# Tests of 07_stream.R and 08_sensitivity.R on the synthetic data.
#   Rscript v1/R/tests/test_stream_sens.R        (re-runs itself with OPTICS_DATA=synthetic)
# Everything is written under work/synthetic/out/work/tests/stream_sens (the default synthetic lock, models and
# results are not touched). The synthetic inputs are read-only; the poison tests work on copies.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
r_dir <- dirname(.here)
for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R", "assoc_lib.R", "30_bundle.R",
            "04a_gate_fit_lock.R", "04b_external_eval.R", "07_stream.R", "08_sensitivity.R")) source(file.path(r_dir, f))

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
base <- file.path("work", "tests", "stream_sens")
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

# ---- run-length logic on toy sequences ------------------------------------------------------------------
r1 <- flag_runs(c(FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE), 1:7)
check("flag_runs: two runs (2 and 3 frames), positions reported", identical(r1$first, c(2L, 5L)) && identical(r1$last, c(3L, 7L)) && identical(r1$n_frames, c(2L, 3L)))
r2 <- flag_runs(c(TRUE, TRUE, TRUE, TRUE), c(1L, 2L, 4L, 5L))
check("flag_runs: a gap in the frame index breaks a run of flagged frames", identical(r2$n_frames, c(2L, 2L)) && identical(r2$first, c(1L, 3L)))
check("flag_runs: all flagged is one run, none flagged is none, a single frame works",
      identical(flag_runs(rep(TRUE, 6), 1:6)$n_frames, 6L) && nrow(flag_runs(rep(FALSE, 6), 1:6)) == 0L &&
        identical(flag_runs(TRUE, 7L)$n_frames, 1L) && nrow(flag_runs(FALSE, 7L)) == 0L)
check("flag_runs: isolated flags are runs of one; alternating gives n/2 runs", identical(flag_runs(rep(c(TRUE, FALSE), 5), 1:10)$n_frames, rep(1L, 5)))
check("flag_runs refuses an unsorted index, NA and a length mismatch",
      errs(flag_runs(c(TRUE, TRUE), c(2L, 1L)), "strictly increasing") && errs(flag_runs(c(TRUE, NA), 1:2), "without NA") &&
        errs(flag_runs(c(TRUE, TRUE), 1L), "same length"))
toy <- data.frame(video_patient = rep(c("A", "B"), c(12, 6)), t_index = c(0:11, 0:5), frame_idx = c(1:12, 11:16),
                  flagged = c(FALSE, rep(TRUE, 6), FALSE, rep(TRUE, 2), FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE))
rt <- runs_table(toy)
check("runs_table: runs per video with start/end frame, duration in seconds, ids restart per video",
      identical(rt$video_patient, c("A", "A", "B", "B")) && identical(rt$n_frames, c(6L, 2L, 2L, 3L)) &&
        identical(rt$run_id, c(1L, 2L, 1L, 2L)) && identical(rt$start_frame_idx, c(2L, 9L, 11L, 14L)) &&
        identical(rt$end_t_index, c(6L, 9L, 1L, 5L)) && near(rt$duration_s, rt$n_frames * STREAM_FRAME_PERIOD_S))
rs <- run_summary(rt, 18, sum(toy$flagged))
check("run_summary: count, median, max, and the shares of all and of flagged frames in runs of at least the long-run length",
      rs[["n_runs"]] == 4 && near(rs[["run_median_s"]], 2.5 * STREAM_FRAME_PERIOD_S) && near(rs[["run_max_s"]], 6 * STREAM_FRAME_PERIOD_S) &&
        near(rs[["share_frames_in_long_runs"]], 6 / 18) && near(rs[["share_flagged_in_long_runs"]], 6 / 13))
rs0 <- run_summary(runs_table(transform(toy, flagged = FALSE)), 18, 0)
check("run_summary with no flagged frame: zero runs, NA median and max, zero share", rs0[["n_runs"]] == 0 && is.na(rs0[["run_median_s"]]) && rs0[["share_frames_in_long_runs"]] == 0)

# ---- constants come from analysis_plan.yaml and keep their audited values ----------------------------------
check("07 constants are read from the stream: section of analysis_plan.yaml; values unchanged (1 s period, 5 s sustained run, 25 unfound frames)",
      identical(STREAM_FRAME_PERIOD_S, as.numeric(CFG$stream$frame_period_s)) && identical(STREAM_LONG_RUN_S, as.numeric(CFG$stream$long_run_s)) &&
        identical(STREAM_EXPECTED_EXCLUDED, as.integer(CFG$stream$expected_not_found)) &&
        STREAM_FRAME_PERIOD_S == 1 && STREAM_LONG_RUN_S == 5 && STREAM_EXPECTED_EXCLUDED == 25L)
check("08 XGBoost settings are read from sensitivity.xgboost; values unchanged (500 rounds, patience 20, eta 0.05, depth 3, seed from seeds.global)",
      SENS_XGB_ROUNDS == 500L && SENS_XGB_PATIENCE == 20L && identical(SENS_XGB$params$eta, 0.05) && SENS_XGB$params$max_depth == 3 &&
        SENS_XGB$params$min_child_weight == 5 && SENS_XGB$params$subsample == 0.8 && SENS_XGB$params$colsample_bytree == 0.8 &&
        SENS_XGB$params$nthread == 1 && SENS_XGB$params$objective == "binary:logistic" && SENS_XGB$params$eval_metric == "logloss" &&
        !"seed" %in% names(SENS_XGB$params))

# ---- 04a on synthetic data gives the lock and the models -------------------------------------------------
pa <- mk_paths("a")
dir.create(dirname(pa$lock), recursive = TRUE)
writeLines(c("# frozen at G2", "failure_iou_threshold: 0.75", "epochs:", "  unet: 30", "  segformer: 40",
             "noninferiority_margin_aurc: 0.01", "lock_fields_sha256: \"deadbeef\""), pa$lock)
sm <- quiet(fit_and_lock(pa))
lock <- yaml::read_yaml(pa$lock)
gate <- readRDS(p(lock$gate_rds_path))
real_dirs_ok <- function() identical(snap_real(), real_before)

# ---- 07: stream ------------------------------------------------------------------------------------------
t_07 <- system.time(st <- quiet(run_stream(pa, B_TEST, CORES)))[["elapsed"]]
man <- quiet(arrow::read_parquet(file.path(pa$derived, "manifest.parquet")))
fov <- quiet(arrow::read_parquet(file.path(pa$derived, "fov.parquet")))
opt <- quiet(arrow::read_parquet(file.path(pa$derived, "optics.parquet")))
ms <- man[man$partition == "stream", ]
nf_stream <- ms$frame_id[!fov$fov_found[match(ms$frame_id, fov$frame_id)]]
fl <- rd(pa, "stream_flags.csv")
ru <- rd(pa, "stream_runs.csv")
su <- rd(pa, "stream_summary.csv")
ov <- function(s, col = "est") su[[col]][su$scope == "overall" & su$statistic == s]

check("stream_flags.csv has the documented columns and one row per analysed frame",
      identical(names(fl), c("frame_id", "video_patient", "t_index", "frame_idx", "gate_score", "flagged")) &&
        nrow(fl) == nrow(ms) - length(nf_stream) && !anyDuplicated(fl$frame_id))
check("stream_runs.csv and stream_summary.csv have the documented columns",
      identical(names(ru), RUN_COLUMNS) && identical(names(su), c("scope", "video_patient", "statistic", "est", "lo", "hi", "n")))
check("frames without a detected field of view are excluded and counted (manifest total, exclusions, analysed)",
      length(nf_stream) > 0L && st$n_excluded_fov == length(nf_stream) && st$n_total == nrow(ms) && !any(fl$frame_id %in% nf_stream) &&
        ov("n_frames_analysed") == nrow(fl) && ov("n_frames_excluded_fov") == length(nf_stream))
thr_stored <- gate$flag_thresholds[["0.9"]]
thr_c1 <- unname(stats::quantile(sm$oof_link, 0.9, type = 1))
check("the flag threshold is the Center 1 out-of-fold quantile stored in the locked gate, unchanged",
      near(thr_stored, thr_c1) && near(ov("flag_threshold_90"), thr_stored) && near(st$thresholds[[1]], thr_stored) &&
        near(ov("flag_threshold_80"), gate$flag_thresholds[["0.8"]]))
opt_s <- opt[match(fl$frame_id, opt$frame_id), ]
check("gate_score is the locked gate's log-odds on the whole-FOV optics; flagged = score above the threshold",
      near(fl$gate_score, predict_gate(gate, opt_s, "link"), 1e-9) && identical(fl$flagged, fl$gate_score > thr_stored))
check("flags are ordered by video and frame index, t_index increases within a video",
      !is.unsorted(order(fl$video_patient, fl$frame_idx)) && all(tapply(fl$t_index, fl$video_patient, function(x) !is.unsorted(x, strictly = TRUE))))

# the share flagged: Center 1 out-of-fold close to the nominal 10% by construction; stream and Center 2 reported at the same threshold
check("Center 1 out-of-fold share flagged is at most the nominal 1 - coverage and close to it",
      ov("flagged_share_center1_oof_90") <= 0.1 + 1e-9 && ov("flagged_share_center1_oof_90") > 0.08 &&
        ov("flagged_share_center1_oof_80") <= 0.2 + 1e-9 && ov("flagged_share_center1_oof_80") > 0.17)
c2 <- quiet(read_eval_frames(pa$derived, "center2_labeled", "unet"))
check("Center 2 share flagged at the Center 1 threshold equals the direct count (Center 2 optics are shifted, so it is large)",
      near(ov("flagged_share_center2_90"), mean(predict_gate(gate, c2, "link") > thr_stored)) && ov("flagged_share_center2_90") > 0.4)
check("stream share flagged equals the mean of the flag column, with a video-clustered interval around it",
      near(ov("flagged_share_stream_90"), mean(fl$flagged)) && ov("flagged_share_stream_90", "lo") <= ov("flagged_share_stream_90") &&
        ov("flagged_share_stream_90") <= ov("flagged_share_stream_90", "hi"))

# runs recomputed independently from the flag file
indep_runs <- function(d) {
  d <- d[order(d$video_patient, d$frame_idx), ]
  new <- c(TRUE, d$video_patient[-1] != d$video_patient[-nrow(d)] | diff(d$frame_idx) != 1 | d$flagged[-1] != d$flagged[-nrow(d)])
  g <- cumsum(new)
  len <- tapply(d$flagged, g, length)
  flg <- tapply(d$flagged, g, function(x) x[1])
  sort(as.integer(len[flg]))
}
check("runs file: lengths equal an independent run-length computation and add up to the flagged frames",
      identical(sort(ru$n_frames), indep_runs(fl)) && sum(ru$n_frames) == sum(fl$flagged) && near(ru$duration_s, ru$n_frames * STREAM_FRAME_PERIOD_S))
check("runs never cross a gap in the frame index (the synthetic stream has gaps where frames were excluded)",
      all(ru$end_frame_idx - ru$start_frame_idx + 1 == ru$n_frames) && ov("n_frame_gaps") > 0)
long <- ru$duration_s >= STREAM_LONG_RUN_S
check("summary: runs, median, max and the shares in sustained runs agree with the runs file",
      ov("n_runs") == nrow(ru) && near(ov("run_median_s"), median(ru$duration_s)) && near(ov("run_max_s"), max(ru$duration_s)) &&
        near(ov("share_frames_in_long_runs"), sum(ru$n_frames[long]) / nrow(fl)) &&
        near(ov("share_flagged_in_long_runs"), sum(ru$n_frames[long]) / sum(fl$flagged)))
vid <- su[su$scope == "video" & su$statistic == "flagged_share", ]
check("per-video flagged share (Figure 3D) equals the mean flag per video, one row per video",
      nrow(vid) == length(unique(fl$video_patient)) &&
        near(vid$est[match(names(tapply(fl$flagged, fl$video_patient, mean)), vid$video_patient)], as.numeric(tapply(fl$flagged, fl$video_patient, mean))) &&
        near(ov("video_flagged_share_max"), max(vid$est)) && near(ov("video_flagged_share_min"), min(vid$est)))
sp_e <- su[su$statistic == "spearman_gate_ens_dis", ]
ens_s <- quiet(arrow::read_parquet(file.path(pa$derived, "stream.parquet")))
ens_s <- ens_s[ens_s$model == "unet", ]
ens_v <- ens_s$ens_dis[match(fl$frame_id, ens_s$frame_id)]
ent_v <- ens_s$entropy_fov[match(fl$frame_id, ens_s$frame_id)]
check("Spearman of the gate score with ensemble disagreement and entropy equals the rank correlation; planted signal is positive; CI brackets it",
      near(sp_e$est, cor(fl$gate_score, ens_v, method = "spearman")) &&
        near(su$est[su$statistic == "spearman_gate_entropy_fov"], cor(fl$gate_score, ent_v, method = "spearman")) &&
        sp_e$est > 0.1 && sp_e$lo <= sp_e$est && sp_e$est <= sp_e$hi && sp_e$n == nrow(fl))
check("share of stream frames outside the Center 1 training range is reported", is.finite(ov("outside_range_any")))

# the threshold is fixed on Center 1: changing every stream score and optic leaves it, and the Center 1 share, untouched
copy_derived <- function(tag, pred = NULL, stream = NULL, optics = NULL) {
  dir <- p(base, tag)
  dir.create(dir, recursive = TRUE)
  invisible(file.copy(file.path(pa$derived, c("manifest.parquet", "fov.parquet", "folds.csv")), dir))
  rdp <- function(f) quiet(arrow::read_parquet(file.path(pa$derived, f)))
  write <- function(df, f) quiet(arrow::write_parquet(df, file.path(dir, f)))
  write(if (is.null(pred)) rdp("predictions.parquet") else pred(rdp("predictions.parquet")), "predictions.parquet")
  write(if (is.null(stream)) rdp("stream.parquet") else stream(rdp("stream.parquet")), "stream.parquet")
  write(if (is.null(optics)) rdp("optics.parquet") else optics(rdp("optics.parquet")), "optics.parquet")
  dir
}
shift_stream <- copy_derived("shift_stream",
  stream = function(s) { s$ens_dis <- rev(s$ens_dis); s },
  optics = function(o) { i <- o$frame_id %in% ms$frame_id; o$under_frac[i] <- pmin(1, o$under_frac[i] * 5); o$haze_dcp[i] <- o$haze_dcp[i] + 0.2; o })
pb <- mk_paths("a", shift_stream)
pb$stats <- p(base, "shift_stats")
pb$results <- p(base, "shift_root")
st_b <- quiet(run_stream(pb, B_TEST, CORES))
su_b <- rd(pb, "stream_summary.csv")
ob <- function(s) su_b$est[su_b$scope == "overall" & su_b$statistic == s]
check("scoring a very different stream leaves the flag threshold and the Center 1 share unchanged but moves the stream share",
      near(ob("flag_threshold_90"), ov("flag_threshold_90")) && near(ob("flagged_share_center1_oof_90"), ov("flagged_share_center1_oof_90")) &&
        ob("flagged_share_stream_90") > ov("flagged_share_stream_90") + 0.1)

# lock integrity: a modified gate stops the run before any data are read; so does a changed frozen field
gate_file <- p(lock$gate_rds_path)
orig <- readBin(gate_file, "raw", file.size(gate_file))
g_mod <- gate
g_mod$flag_thresholds <- g_mod$flag_thresholds - 5
saveRDS(g_mod, gate_file)
pt <- pa
pt$stats <- p(base, "tamper_stats")
pt$derived <- p(base, "no_such_dir")
e1 <- tryCatch(quiet(run_stream(pt, 20L, 1L)), error = function(e) conditionMessage(e))
e2 <- tryCatch(quiet(run_sensitivity(pt, 20L, 1L)), error = function(e) conditionMessage(e))
check("07 and 08 stop with a SHA-256 mismatch after the gate RDS is modified, before any data are read",
      is.character(e1) && grepl("SHA-256 mismatch", e1) && grepl("gate_optical.rds", e1) && is.character(e2) && grepl("SHA-256 mismatch", e2))
check("the refused runs wrote nothing", !dir.exists(pt$stats))
writeBin(orig, gate_file)
lines <- readLines(pa$lock)
pc <- pt
pc$lock <- p(base, "tamper_threshold.yaml")
writeLines(sub("^failure_iou_threshold: .*", "failure_iou_threshold: 0.5", lines), pc$lock)
check("a lock edited after locking (failure threshold) stops 07 and 08 on lock_fields_sha256",
      errs(quiet(run_stream(pc, 20L, 1L)), "lock_fields_sha256 mismatch") && errs(quiet(run_sensitivity(pc, 20L, 1L)), "lock_fields_sha256 mismatch"))

# a gate whose stored thresholds are not the Center 1 out-of-fold quantiles is refused, even with a consistent lock
alt <- mk_paths("alt")
dir.create(dirname(alt$lock), recursive = TRUE)
dir.create(p(alt$models_rel), recursive = TRUE)
for (o in GATE_LOCK_OBJECTS) invisible(file.copy(p(lock[[paste0(o, "_path")]]), p(alt$models_rel, basename(lock[[paste0(o, "_path")]]))))
g_alt <- gate
g_alt$flag_thresholds <- g_alt$flag_thresholds + 0.5
saveRDS(g_alt, p(alt$models_rel, "gate_optical.rds"))
vals <- list(gate_rds_path = file.path(alt$models_rel, "gate_optical.rds"), gate_area_rds_path = file.path(alt$models_rel, "gate_area.rds"),
             gate_stack_rds_path = file.path(alt$models_rel, "gate_stack.rds"), gate_folds_rds_path = file.path(alt$models_rel, "gate_folds.rds"))
invisible(file.copy(pa$lock, alt$lock))
invisible(lock_update(alt$lock, vals))
invisible(lock_update(alt$lock, setNames(lapply(GATE_LOCK_OBJECTS, function(o) unname(tools::sha256sum(p(vals[[paste0(o, "_path")]])))), paste0(GATE_LOCK_OBJECTS, "_sha256"))))
invisible(lock_update(alt$lock, list(lock_fields_sha256 = lock_fields_hash(yaml::read_yaml(alt$lock)))))
check("a consistently re-locked gate with shifted flag thresholds is still refused (thresholds must be the Center 1 quantiles)",
      isTRUE(verify_lock_hashes(yaml::read_yaml(alt$lock))) && errs(quiet(run_stream(alt, 20L, 1L)), "differ from the Center 1 out-of-fold quantiles"))

# ---- 08: sensitivity analyses ---------------------------------------------------------------------------
SENS_COLUMNS <- c("variant", "label", "aggregation", "train_domains", "eval_partition", "eval_domain", "method", "n_frames", "n_patients",
                  "n_events", "n_excluded", "aurc", "aurc_lo", "aurc_hi", "diff_vs_random", "diff_vs_random_lo", "diff_vs_random_hi",
                  "diff_vs_entropy", "diff_vs_entropy_lo", "diff_vs_entropy_hi", "diff_vs_ens", "diff_vs_ens_lo", "diff_vs_ens_hi", "status")
t_08 <- system.time(se <- quiet(run_sensitivity(pa, B_TEST, CORES)))[["elapsed"]]
tb <- rd(pa, "sensitivity.csv")
gt <- function(v, col) tb[[col]][tb$variant == v & tb$method == "gate"]
check("sensitivity.csv: the declared columns, one row per variant x method (11 variants x 4 methods), declared method order",
      identical(names(tb), SENS_COLUMNS) && nrow(tb) == 44L && identical(unique(tb$variant), SENS_VARIANTS$variant) &&
        all(tapply(tb$method, tb$variant, function(m) identical(as.character(m), SENS_METHODS))))
check("every variant returns finite AURC with an interval around it, and finite paired differences for the gate",
      all(is.finite(tb$aurc) & tb$aurc_lo <= tb$aurc & tb$aurc <= tb$aurc_hi) &&
        all(is.finite(as.matrix(tb[tb$method == "gate", c("diff_vs_random", "diff_vs_random_lo", "diff_vs_random_hi", "diff_vs_ens", "diff_vs_entropy")]))))
check("differences are method minus reference (point estimates), NA against itself",
      near(tb$diff_vs_random[tb$method == "gate"], tb$aurc[tb$method == "gate"] - tb$aurc[tb$method == "random"]) &&
        near(tb$diff_vs_ens[tb$method == "gate"], tb$aurc[tb$method == "gate"] - tb$aurc[tb$method == "ens_disagreement"]) &&
        all(is.na(tb$diff_vs_random[tb$method == "random"])) && all(is.na(tb$diff_vs_ens[tb$method == "ens_disagreement"])) &&
        all(is.na(tb$diff_vs_entropy[tb$method == "entropy"])))
check("rows are labelled exploratory and Center 1-only, except the locked reference row",
      all(grepl("exploratory", tb$status[tb$variant != "locked"])) && all(grepl("locked primary", tb$status[tb$variant == "locked"])))
check("device_norm row states that the D3 normaliser uses Center 2 optics with no outcomes; the other exploratory rows keep the generic status",
      all(tb$status[tb$variant == "device_norm"] == SENS_STATUS_DEVICE_NORM) && grepl("Center 2 optics with no outcomes", SENS_STATUS_DEVICE_NORM, fixed = TRUE) &&
        all(tb$status[!tb$variant %in% c("locked", "device_norm")] == SENS_STATUS) && !grepl("Center 2", SENS_STATUS, fixed = TRUE))

# reference: the locked variant reproduces 04b exactly (same frames, same resamples)
res_b <- quiet(run_external_eval(pa, B_TEST, CORES))
ext <- res_b$external$table
tm <- c(random = "random", entropy = "entropy", ens_disagreement = "ens_disagreement", gate = "optical_gate")
lk <- tb[tb$variant == "locked", ]
check("locked variant: AURC and interval equal 04b's external table for random, entropy, ensemble and the gate",
      all(vapply(names(tm), function(m) {
        e <- ext[ext$method == tm[[m]], ]
        r <- lk[lk$method == m, ]
        near(r$aurc, e$aurc) && near(r$aurc_lo, e$aurc_lo) && near(r$aurc_hi, e$aurc_hi)
      }, NA)))
pdf <- res_b$external$diffs
pdg <- pdf[pdf$metric == "aurc" & pdf$comparison == "gate_vs_random", ]
check("locked variant: paired difference gate - random equals 04b's, interval included",
      near(gt("locked", "diff_vs_random"), pdg$diff) && near(gt("locked", "diff_vs_random_lo"), pdg$lo) && near(gt("locked", "diff_vs_random_hi"), pdg$hi))
check("patient-macro variant equals 04b's aurc_macro (point estimate and interval) for all four methods",
      all(vapply(names(tm), function(m) {
        e <- ext[ext$method == tm[[m]], ]
        r <- tb[tb$variant == "patient_macro" & tb$method == m, ]
        near(r$aurc, e$aurc_macro) && near(r$aurc_lo, e$aurc_macro_lo) && near(r$aurc_hi, e$aurc_macro_hi)
      }, NA)) && !near(gt("patient_macro", "aurc"), gt("locked", "aurc"), 1e-6))
ext_variants <- c("xgboost", "gam", "domain_cov", "background_only", "device_norm", "patient_macro", "lodo_d1_to_d3", "lodo_d2_to_d3")
check("planted effect: on Center 2 every variant's gate beats random ordering (CI of the difference below zero)",
      all(vapply(ext_variants, function(v) gt(v, "diff_vs_random_hi") < 0, NA)))
check("leave-one-domain-out within Center 1 (D1 -> D2, D2 -> D1): the gate also beats random",
      gt("lodo_d1_to_d2", "diff_vs_random_hi") < 0 && gt("lodo_d2_to_d1", "diff_vs_random_hi") < 0)
check("evaluation sets: Center 2 for the first nine variants (D3), Center 1 domain D2 or D1 for the two within-Center-1 ones",
      all(tb$eval_partition[tb$variant %in% c(ext_variants, "locked")] == "center2_labeled") &&
        identical(unique(tb$eval_domain[tb$variant == "lodo_d1_to_d2"]), "D2") && identical(unique(tb$eval_domain[tb$variant == "lodo_d2_to_d1"]), "D1") &&
        unique(tb$n_patients[tb$variant == "lodo_d1_to_d2"]) == 20L && unique(tb$n_patients[tb$variant == "lodo_d2_to_d1"]) == 41L)

# each fitter alone: new models differ from the locked gate but rank frames similarly
c1 <- quiet(read_sens_frames(pa$derived, "center1_labeled"))
c2s <- quiet(read_sens_frames(pa$derived, "center2_labeled"))
c1$fail <- c1$iou < 0.75
sc <- se$scores
rho <- function(v) cor(sc[[v]]$score, sc$locked$score, method = "spearman")
check("xgboost, GAM and domain-covariate scores differ from the locked gate and are strongly rank-correlated with it",
      all(vapply(c("xgboost", "gam", "domain_cov"), function(v) !isTRUE(all.equal(sc[[v]]$score, sc$locked$score)) && rho(v) > 0.6, NA)))
settings <- list(df = gate$spec$df, alpha = gate$alpha)
dom_a <- quiet(fit_domain_gate(c1, c1$fail, settings))
d2 <- c2s
d2$domain <- "D2"
dd <- dom_a(c2s) - dom_a(d2)
check("domain covariate: scoring Center 2 at another domain level shifts every log-odds by one constant, so the ranking is unchanged",
      near(dd, rep(dd[1], length(dd)), 1e-9) && dd[1] != 0)

# the XGBoost fitter follows the configured parameters: the run's scores are reproduced, and other settings change them
xgb_run <- quiet(fit_xgb_gate(c1, c1$fail, settings))(c2s)
check("XGBoost fitter reproduces the run's scores from the configured settings",
      isTRUE(all.equal(xgb_run, sc$xgboost$score, tolerance = 1e-8)))
SENS_XGB_kept <- SENS_XGB
SENS_XGB$params$max_depth <- 1L
SENS_XGB$params$eta <- 0.3
xgb_other <- quiet(fit_xgb_gate(c1, c1$fail, settings))(c2s)
SENS_XGB <- SENS_XGB_kept
check("changing max_depth and eta in sensitivity.xgboost changes the XGBoost scores (the fitter reads the configuration)",
      !isTRUE(all.equal(xgb_run, xgb_other, tolerance = 1e-6)))

# device-normalised features: statistics from optics only, per domain, Center 2 statistics for D3
c2s$fail <- c2s$iou < 0.75
nst <- domain_norm_stats(rbind(c1[c("domain", OPTICAL_METRICS)], c2s[c("domain", OPTICAL_METRICS)]))
n1 <- normalise_by_domain(c1, nst)
n2 <- normalise_by_domain(c2s, nst)
check("device normalisation: D1 and D2 use Center 1 statistics, D3 uses Center 2 statistics; each domain ends with mean 0 and SD 1",
      near(nst$D3$centre, colMeans(c2s[OPTICAL_METRICS]), 1e-12) && near(nst$D1$centre, colMeans(c1[c1$domain == "D1", OPTICAL_METRICS]), 1e-12) &&
        near(nst$D2$centre, colMeans(c1[c1$domain == "D2", OPTICAL_METRICS]), 1e-12) &&
        near(colMeans(n2[OPTICAL_METRICS]), 0, 1e-9) && near(apply(n2[OPTICAL_METRICS], 2, sd), 1, 1e-9) &&
        near(colMeans(n1[n1$domain == "D2", OPTICAL_METRICS]), 0, 1e-9))
check("device normalisation statistics refuse any frame table that carries outcome columns",
      errs(domain_norm_stats(cbind(c1[c("domain", OPTICAL_METRICS)], iou = c1$iou)), "setequal"))
check("device-normalised variant removes the domain offset: Center 2 raw and normalised metric medians differ, normalised are near 0",
      max(abs(vapply(OPTICAL_METRICS, function(m) median(n2[[m]]), 0))) < 0.5 && max(abs(vapply(OPTICAL_METRICS, function(m) median(c2s[[m]]) - median(c1[[m]]), 0) /
                                                                                       vapply(OPTICAL_METRICS, function(m) sd(c1[[m]]), 0))) > 0.5)

# no Center 2 outcome and no held-out-domain outcome enters any fit: scramble them and compare the scores
dom_of <- function(pid) { fo <- utils::read.csv(file.path(pa$derived, "folds.csv")); fo$domain[match(pid, fo$patient_id)] }
scramble <- function(domains) function(pr) {
  set.seed(77)
  i <- dom_of(pr$patient_id) %in% domains
  for (col in c("iou", "dice", "ens_dis", "entropy_fov", "entropy_roi", "pred_area_frac")) pr[[col]][i] <- runif(sum(i))
  pr
}
run_scores <- function(dir, variants) {
  pz <- pa
  pz$derived <- dir
  pz$stats <- p(base, paste0("stats_", basename(dir)))
  quiet(run_sensitivity(pz, 20L, 1L, variants = variants))
}
d_c2 <- copy_derived("poison_d3", pred = scramble("D3"))
d_d2d3 <- copy_derived("poison_d2d3", pred = scramble(c("D2", "D3")))
d_d1d3 <- copy_derived("poison_d1d3", pred = scramble(c("D1", "D3")))
fit_variants <- c("xgboost", "gam", "domain_cov", "background_only", "device_norm", "locked", "lodo_d1_to_d3", "lodo_d2_to_d3")
clean <- run_scores(pa$derived, SENS_VARIANTS$variant)
pz_c2 <- run_scores(d_c2, fit_variants)
same <- function(a, b, v) identical(a$scores[[v]], b$scores[[v]])
check("scrambling every Center 2 outcome and baseline leaves each variant's gate scores bit-identical (no Center 2 outcome enters a fit)",
      all(vapply(fit_variants, function(v) same(clean, pz_c2, v), NA)) &&
        !near(pz_c2$table$aurc[pz_c2$table$variant == "locked" & pz_c2$table$method == "random"], clean$table$aurc[clean$table$variant == "locked" & clean$table$method == "random"], 1e-6))
pz_a <- run_scores(d_d2d3, c("lodo_d1_to_d2", "lodo_d1_to_d3"))
pz_b <- run_scores(d_d1d3, c("lodo_d2_to_d1", "lodo_d2_to_d3"))
check("leave-one-domain-out never uses the held-out domain's outcomes: D1-trained scores are bit-identical when D2 and D3 outcomes are scrambled",
      same(clean, pz_a, "lodo_d1_to_d2") && same(clean, pz_a, "lodo_d1_to_d3"))
check("and D2-trained scores are bit-identical when D1 and D3 outcomes are scrambled",
      same(clean, pz_b, "lodo_d2_to_d1") && same(clean, pz_b, "lodo_d2_to_d3"))
check("the scrambled held-out outcomes do change the evaluation (the test has power)",
      !near(pz_a$table$aurc[pz_a$table$variant == "lodo_d1_to_d2" & pz_a$table$method == "random"],
            clean$table$aurc[clean$table$variant == "lodo_d1_to_d2" & clean$table$method == "random"], 1e-6))
check("leave-one-domain-out fits only on its training domain: D1-trained and D2-trained gates differ and both differ from the locked gate",
      !isTRUE(all.equal(sc$lodo_d1_to_d3$score, sc$lodo_d2_to_d3$score)) && !isTRUE(all.equal(sc$lodo_d1_to_d3$score, sc$locked$score)))

# background-only: frames without finite background metrics (fewer than 256 background pixels) are dropped and counted
pr0 <- quiet(arrow::read_parquet(file.path(pa$derived, "predictions.parquet")))
c1_ids <- pr0$frame_id[pr0$partition == "center1_labeled" & pr0$model == "unet"]
c2_ids <- pr0$frame_id[pr0$partition == "center2_labeled" & pr0$model == "unet"]
bad <- c(c1_ids[c(5, 50, 90)], c2_ids[c(3, 4)])
d_nan <- copy_derived("nan_bg", pred = function(pr) { pr$unet_bg_lum_mean[pr$frame_id %in% bad] <- NaN; pr })
nan_res <- run_scores(d_nan, c("background_only", "locked"))
nf <- quiet(arrow::read_parquet(file.path(pa$derived, "fov.parquet")))
bad_found <- bad[nf$fov_found[match(bad, nf$frame_id)]]
n_bad_c2 <- sum(bad_found %in% c2_ids)
tb_n <- nan_res$table
check("background-only: frames with non-finite background metrics are excluded from fit and evaluation, and the exclusion is counted",
      n_bad_c2 > 0L && unique(tb_n$n_excluded[tb_n$variant == "background_only"]) == n_bad_c2 &&
        unique(tb_n$n_frames[tb_n$variant == "background_only"]) == unique(tb_n$n_frames[tb_n$variant == "locked"]) - n_bad_c2 &&
        unique(tb_n$n_excluded[tb_n$variant == "locked"]) == 0L)

# output guard and unknown variants
pg <- pa
pg$stats <- file.path(V1, "results", "stats")
check("a sensitivity run pointed at the real results folder is refused in synthetic mode, and nothing is written there",
      errs(quiet(run_sensitivity(pg, 20L, 1L, variants = "locked")), "refusing to write outside") &&
        errs(quiet(run_stream(pg, 20L, 1L)), "refusing to write outside") && real_dirs_ok())
check("an unknown variant is an error", errs(quiet(run_sensitivity(pa, 20L, 1L, variants = "nope")), "unknown variant"))

# ---- registry and bundle -----------------------------------------------------------------------------------
reg_reset()
register_04a(sm)
write_results_part("04a_gate_fit_lock", root = pa$results)
register_04b(res_b, B_TEST)
write_results_part("04b_external_eval", root = pa$results)
register_07(st)
write_results_part("07_stream", root = pa$results)
register_08(se)
write_results_part("08_sensitivity", root = pa$results)
bund <- bundle_results(pa$results)
rj <- read_results(pa$results)
check("07 and 08 register keys that do not collide with 04a/04b (bundle builds)", bund$n_parts == 4L && bund$n_keys > 150L)
k07 <- grep("^stream_", names(rj), value = TRUE)
k08 <- grep("^sens_", names(rj), value = TRUE)
other_parts <- list.files(p("results", "parts"), pattern = "\\.json$", full.names = TRUE)
other_parts <- other_parts[!basename(other_parts) %in% c("07_stream.json", "08_sensitivity.json")]
other_keys <- unlist(lapply(other_parts, function(f) names(jsonlite::read_json(f)$entries)))
check("stream_ and sens_ keys are used by no other part (04a, 04b, 05, 03, 06 in the default synthetic results)",
      length(k07) > 30L && length(k08) > 40L && !any(grepl("^(stream_|sens_)", other_keys)))
check("registered displays: shares as percent, run lengths with seconds, correlations and AURC differences with a 95% CI",
      grepl("%$", rj$stream_flagged_share) && grepl(" s$", rj$stream_run_median_s) && grepl("^-?0\\.[0-9]{2} \\(95% CI -?0\\.[0-9]{2} to -?0\\.[0-9]{2}\\)$", rj$stream_spearman_ens_ci) &&
        grepl("^-?0\\.[0-9]{3} \\(95% CI -?0\\.[0-9]{3} to -?0\\.[0-9]{3}\\)$", rj$sens_xgboost_diff_random_ci) &&
        grepl("%$", rj$stream_c2_flagged_share))
check("raw values in the bundle equal the CSV values",
      near(rj[["raw.stream_flagged_share"]], ov("flagged_share_stream_90"), 1e-12) && near(rj[["raw.stream_n_frames"]], nrow(fl)) &&
        near(unlist(rj[["raw.sens_gam_aurc_ci"]])[1], gt("gam", "aurc"), 1e-12) && rj$sens_n_ext_variants == "8" &&
        near(rj[["raw.sens_n_ext_beat_random"]], sum(vapply(ext_variants, function(v) gt(v, "diff_vs_random_hi") < 0, NA))))
dn_note <- jsonlite::read_json(file.path(pa$results, "results", "parts", "08_sensitivity.json"))$entries$sens_device_norm_aurc_ci$note
check("registered note of the device_norm AURC carries the Center 2 optics statement",
      grepl("Center 2 optics with no outcomes", dn_note, fixed = TRUE))
check("the real lock file and result folders were not touched", real_dirs_ok())

cat(sprintf("\nruntime (synthetic, B = %d): 07 %.1f s, 08 %.1f s\n", B_TEST, t_07, t_08))
cat(sprintf("%d failure(s)\n", length(failures)))
if (length(failures)) quit(status = 1)
