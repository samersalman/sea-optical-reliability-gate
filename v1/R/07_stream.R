# 07: deployment stream. Apply the LOCKED optical gate to the unlabeled stream frames (whole-FOV optics) and describe
# what it would withhold. Descriptive only: the stream has no ground truth.
# The lock is verified first (lock_fields_sha256 and the RDS hashes). The flag threshold is the one stored in the
# locked gate (the Center 1 out-of-fold quantile of the log-odds at the chosen coverage, fixed at 04a before any
# stream or Center 2 score existed); it is checked against a recomputation from Center 1 and never re-estimated here.
#   Rscript R/07_stream.R                 (OPTICS_DATA=synthetic for development)
# Flag rule   a frame is withheld when its gate log-odds exceed the threshold of the primary coverage (the first of
#             endpoints.coverage_levels, 90%); the second level (80%) is reported for the share flagged only.
# Time        the stream is sampled at 1 frame/s; adjacency uses the file's frame index (manifest frame_idx), so a frame
#             excluded for lacking a field of view, or missing in the source video, breaks a run instead of being bridged.
# Outputs results/stats/stream_flags.csv      one row per analysed frame: frame_id, video_patient, t_index, frame_idx,
#                                             gate_score (log-odds), flagged (primary coverage)
#         results/stats/stream_runs.csv       one row per run of consecutive flagged frames
#         results/stats/stream_summary.csv    long table: scope (overall | video), video_patient, statistic, est, lo, hi, n
#         results/parts/07_stream.json

if (!exists("run_locked_evaluation", mode = "function") || !exists("assoc_boot_ci", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R", "assoc_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

# all three come from the stream: section of analysis_plan.yaml (frame period 1 s, sustained run 5 s, 25 unfound frames)
STREAM_FRAME_PERIOD_S <- as.numeric(CFG$stream$frame_period_s)     # data audit: consecutive frames 1 s apart
STREAM_LONG_RUN_S <- as.numeric(CFG$stream$long_run_s)             # a run of at least this many seconds counts as sustained
STREAM_EXPECTED_EXCLUDED <- as.integer(CFG$stream$expected_not_found)   # Gate G1: stream frames without a detected field of view
if (anyNA(c(STREAM_FRAME_PERIOD_S, STREAM_LONG_RUN_S, STREAM_EXPECTED_EXCLUDED)) ||
    length(c(STREAM_FRAME_PERIOD_S, STREAM_LONG_RUN_S, STREAM_EXPECTED_EXCLUDED)) != 3L) {
  stop("analysis_plan.yaml: stream needs frame_period_s, long_run_s and expected_not_found")
}

# ---- runs ---------------------------------------------------------------------------------------------------

# Runs of consecutive flagged frames in ONE video. flagged: logical; idx: strictly increasing integer frame index.
# Frames more than one step apart are not consecutive. Returns first and last position (in the input) and n_frames.
flag_runs <- function(flagged, idx) {
  n <- length(flagged)
  if (length(idx) != n || anyNA(flagged) || anyNA(idx) || is.unsorted(idx, strictly = TRUE)) {
    stop("flag_runs needs flagged and a strictly increasing frame index of the same length, without NA")
  }
  if (!any(flagged)) return(data.frame(first = integer(), last = integer(), n_frames = integer()))
  joined <- c(FALSE, flagged[-1L] & flagged[-n] & diff(idx) == 1L)
  grp <- cumsum(!joined)[flagged]
  pos <- which(flagged)
  first <- as.integer(tapply(pos, grp, min))
  last <- as.integer(tapply(pos, grp, max))
  data.frame(first = first, last = last, n_frames = last - first + 1L)
}

RUN_COLUMNS <- c("video_patient", "run_id", "start_frame_idx", "end_frame_idx", "start_t_index", "end_t_index",
                 "n_frames", "duration_s")

# fl: video_patient, t_index, frame_idx, flagged, ordered by video then frame_idx
runs_table <- function(fl) {
  out <- lapply(split(seq_len(nrow(fl)), fl$video_patient), function(i) {
    r <- flag_runs(fl$flagged[i], fl$frame_idx[i])
    if (!nrow(r)) return(NULL)
    data.frame(video_patient = fl$video_patient[i[1]], run_id = seq_len(nrow(r)),
               start_frame_idx = fl$frame_idx[i[r$first]], end_frame_idx = fl$frame_idx[i[r$last]],
               start_t_index = fl$t_index[i[r$first]], end_t_index = fl$t_index[i[r$last]],
               n_frames = r$n_frames, duration_s = r$n_frames * STREAM_FRAME_PERIOD_S, stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, out)
  if (is.null(out)) {
    out <- data.frame(video_patient = character(), run_id = integer(), start_frame_idx = integer(), end_frame_idx = integer(),
                      start_t_index = integer(), end_t_index = integer(), n_frames = integer(), duration_s = numeric())
  }
  rownames(out) <- NULL
  out[RUN_COLUMNS]
}

# n_frames / n_flagged: the frames of the video (or of all videos) the runs came from
run_summary <- function(runs, n_frames, n_flagged) {
  long <- runs$duration_s >= STREAM_LONG_RUN_S
  c(n_runs = nrow(runs),
    run_median_s = if (nrow(runs)) stats::median(runs$duration_s) else NA_real_,
    run_max_s = if (nrow(runs)) max(runs$duration_s) else NA_real_,
    share_frames_in_long_runs = sum(runs$n_frames[long]) / n_frames,
    share_flagged_in_long_runs = if (n_flagged > 0) sum(runs$n_frames[long]) / n_flagged else NA_real_)
}

# ---- data ---------------------------------------------------------------------------------------------------

# arrow warns "Invalid metadata$r" when a column selection drops columns named in the file's R metadata; harmless
.read_columns <- function(path, cols) {
  withCallingHandlers(as.data.frame(arrow::read_parquet(path, col_select = dplyr::all_of(cols))), warning = function(w) {
    if (grepl("Invalid metadata", conditionMessage(w), fixed = TRUE)) invokeRestart("muffleWarning")
  })
}

# Stream frames with their whole-FOV optics, U-Net entropy and ensemble disagreement, ordered by video and frame index.
# Frames without a detected field of view are excluded and counted; the count is checked against the manifest.
read_stream_frames <- function(derived) {
  man <- .read_columns(file.path(derived, "manifest.parquet"), c("frame_id", "partition", "patient_id", "frame_idx"))
  man <- man[man$partition == "stream", ]
  fov <- .read_columns(file.path(derived, "fov.parquet"), c("frame_id", "fov_found"))
  man$fov_found <- fov$fov_found[match(man$frame_id, fov$frame_id)]
  if (anyNA(man$fov_found)) stop("fov.parquet lacks rows for some stream frames")
  n_total <- nrow(man)
  n_excluded <- if (isTRUE(CFG$fov$exclude_not_found)) sum(!man$fov_found) else 0L
  if (!SYNTH) {
    if (n_total != CFG$dataset_counts_expected$stream_frames) stop("stream frames in the manifest: ", n_total)
    if (n_excluded != STREAM_EXPECTED_EXCLUDED) stop("stream frames without a field of view: ", n_excluded, ", expected ", STREAM_EXPECTED_EXCLUDED)
  }
  man <- man[man$fov_found | !isTRUE(CFG$fov$exclude_not_found), ]
  if (anyNA(man$frame_idx)) stop("stream frames without a frame index in the manifest")

  # predictions.parquet normally lacks the excluded frames; rows for them, if present, are dropped here
  pr <- as.data.frame(.gl_collect(arrow::open_dataset(file.path(derived, "stream.parquet")) |>
    dplyr::filter(model == "unet") |>
    dplyr::select(frame_id, video_patient, t_index, entropy_fov, ens_dis)))
  pr <- pr[pr$frame_id %in% man$frame_id, ]
  if (anyDuplicated(pr$frame_id) || !setequal(pr$frame_id, man$frame_id)) {
    stop("stream.parquet and the manifest disagree on which stream frames are analysed")
  }
  pr$frame_idx <- man$frame_idx[match(pr$frame_id, man$frame_id)]
  if (!identical(pr$video_patient, man$patient_id[match(pr$frame_id, man$frame_id)])) stop("video_patient differs from the manifest patient_id")

  op <- .read_columns(file.path(derived, "optics.parquet"), c("frame_id", OPTICAL_METRICS))
  op <- op[match(pr$frame_id, op$frame_id), ]
  if (anyNA(op$frame_id) || any(!is.finite(as.matrix(op[OPTICAL_METRICS])))) stop("optics missing or non-finite for analysed stream frames")
  fr <- cbind(pr, op[OPTICAL_METRICS])
  fr <- fr[order(fr$video_patient, fr$frame_idx), ]
  rownames(fr) <- NULL
  ord_ok <- tapply(fr$t_index, fr$video_patient, function(x) !is.unsorted(x, strictly = TRUE))
  if (!all(ord_ok)) stop("t_index does not follow the frame index within a video")
  attr(fr, "n_total") <- n_total
  attr(fr, "n_excluded_fov") <- n_excluded
  fr
}

# ---- analysis -----------------------------------------------------------------------------------------------

share_ci <- function(flag, cluster, strata, B, cores) {
  r <- assoc_boot_ci(function(rows) c(share = mean(flag[rows])), cluster, strata, B = B, cores = cores)
  c(est = r$est, lo = r$lo, hi = r$hi)
}

run_stream <- function(paths = default_paths(), B = CFG$statistics$bootstrap_resamples, cores = boot_cores()) {
  lock <- read_lock(paths$lock)
  verify_lock_hashes(lock)
  models <- load_gate_models(lock)
  gate <- models$gate
  cov <- unlist(CFG$endpoints$coverage_levels)
  cov_code <- as.character(round(cov * 100))
  thr <- gate$flag_thresholds[as.character(cov)]
  if (anyNA(thr) || length(thr) != length(cov)) stop("the locked gate has no flag threshold for coverage ", paste(cov, collapse = ", "))

  # the stored thresholds must be the Center 1 out-of-fold quantiles of the link, not something fitted afterwards
  c1 <- read_eval_frames(paths$derived, "center1_labeled", "unet")
  c1_link <- predict_oof(models$folds$optical_gate, c1, c1$fold)
  if (!isTRUE(all.equal(unname(stats::quantile(c1_link, cov, type = 1)), unname(thr), tolerance = 1e-8))) {
    stop("flag thresholds stored in the locked gate differ from the Center 1 out-of-fold quantiles")
  }

  fr <- read_stream_frames(paths$derived)
  fr$gate_score <- predict_gate(gate, fr, "link")
  fr$flagged <- fr$gate_score > thr[[1]]
  c2 <- read_eval_frames(paths$derived, "center2_labeled", "unet")
  c2_link <- predict_gate(gate, c2, "link")

  runs <- runs_table(fr)
  by_video <- split(seq_len(nrow(fr)), fr$video_patient)
  n_gaps <- vapply(by_video, function(i) sum(diff(fr$frame_idx[i]) > 1L), 0)

  rows <- list()
  add <- function(scope, video, statistic, est, lo = NA_real_, hi = NA_real_, n = NA_real_) {
    rows[[length(rows) + 1L]] <<- data.frame(scope = scope, video_patient = video, statistic = statistic, est = est, lo = lo,
                                             hi = hi, n = n, stringsAsFactors = FALSE)
  }
  ov <- function(statistic, est, lo = NA_real_, hi = NA_real_, n = NA_real_) add("overall", "", statistic, est, lo, hi, n)

  ov("n_frames_analysed", nrow(fr))
  ov("n_frames_excluded_fov", attr(fr, "n_excluded_fov"))
  ov("n_videos", length(by_video))
  ov("n_frame_gaps", sum(n_gaps))
  for (j in seq_along(cov)) {
    code <- cov_code[j]
    ov(paste0("flag_threshold_", code), unname(thr[[j]]))
    s <- share_ci(fr$gate_score > thr[[j]], fr$video_patient, NULL, B, cores)
    ov(paste0("flagged_share_stream_", code), s[["est"]], s[["lo"]], s[["hi"]], nrow(fr))
    s1 <- share_ci(c1_link > thr[[j]], c1$patient_id, c1$domain, B, cores)
    ov(paste0("flagged_share_center1_oof_", code), s1[["est"]], s1[["lo"]], s1[["hi"]], nrow(c1))
    s2 <- share_ci(c2_link > thr[[j]], c2$patient_id, NULL, B, cores)
    ov(paste0("flagged_share_center2_", code), s2[["est"]], s2[["lo"]], s2[["hi"]], nrow(c2))
  }
  per_video <- t(vapply(names(by_video), function(v) {
    i <- by_video[[v]]
    rs <- run_summary(runs[runs$video_patient == v, ], length(i), sum(fr$flagged[i]))
    c(n_frames = length(i), n_flagged = sum(fr$flagged[i]), flagged_share = mean(fr$flagged[i]),
      rs[c("n_runs", "run_median_s", "run_max_s", "share_frames_in_long_runs")], n_frame_gaps = n_gaps[[v]])
  }, numeric(8)))
  vs <- per_video[, "flagged_share"]
  ov("video_flagged_share_median", stats::median(vs), n = length(vs))
  ov("video_flagged_share_min", min(vs), n = length(vs))
  ov("video_flagged_share_max", max(vs), n = length(vs))
  rs <- run_summary(runs, nrow(fr), sum(fr$flagged))
  for (nm in names(rs)) ov(nm, rs[[nm]], n = nrow(runs))
  sp <- assoc_spearman_boot(cbind(ens_dis = fr$ens_dis, entropy_fov = fr$entropy_fov), fr$gate_score, fr$video_patient,
                            NULL, B = B, cores = cores)
  for (i in seq_len(nrow(sp))) ov(paste0("spearman_gate_", sp$variable[i]), sp$est[i], sp$lo[i], sp$hi[i], sp$n[i])
  rc <- range_check(gate, fr)
  ov("outside_range_any", rc$frac_outside[rc$feature == "any_feature"], n = nrow(fr))
  for (v in rownames(per_video)) {
    for (nm in colnames(per_video)) add("video", v, nm, per_video[v, nm])
  }
  summary_tab <- do.call(rbind, rows)

  flags <- fr[c("frame_id", "video_patient", "t_index", "frame_idx", "gate_score", "flagged")]
  write_csv_guarded(flags, file.path(paths$stats, "stream_flags.csv"))
  write_csv_guarded(runs, file.path(paths$stats, "stream_runs.csv"))
  write_csv_guarded(summary_tab, file.path(paths$stats, "stream_summary.csv"))
  list(flags = flags, runs = runs, summary = summary_tab, per_video = per_video, thresholds = thr, coverage = cov,
       cov_code = cov_code,
       n_excluded_fov = attr(fr, "n_excluded_fov"), n_total = attr(fr, "n_total"), c1_link = c1_link, c2_link = c2_link)
}

register_07 <- function(res) {
  sm <- res$summary
  ov <- function(statistic) {
    r <- sm[sm$scope == "overall" & sm$statistic == statistic, ]
    if (nrow(r) != 1L) stop("stream summary lacks statistic ", statistic)
    r
  }
  reg("stream_n_frames", as.integer(ov("n_frames_analysed")$est), note = "stream frames analysed (field of view found)")
  reg("stream_n_total", as.integer(res$n_total), note = "stream frames in the manifest")
  reg("stream_n_videos", as.integer(ov("n_videos")$est))
  reg("stream_fov_excluded", as.integer(res$n_excluded_fov), note = "stream frames without a detected field of view, excluded")
  reg("stream_frame_gaps", as.integer(ov("n_frame_gaps")$est), note = "places where analysed frames are more than one frame apart")
  reg("stream_coverage", res$coverage[1], fmt = "percent", note = "primary coverage: the share of Center 1 frames retained")
  c1_code <- res$cov_code[1]
  c2_code <- res$cov_code[2]
  reg("stream_flag_threshold", ov(paste0("flag_threshold_", c1_code))$est, digits = 3, note = "gate log-odds, Center 1 out-of-fold quantile fixed at 04a")
  pct <- function(key, statistic, note = NULL) {
    r <- ov(statistic)
    reg(key, r$est, fmt = "percent", note = note)
    if (is.finite(r$lo)) {
      reg(paste0(key, "_lo"), r$lo, fmt = "percent", note = note)
      reg(paste0(key, "_hi"), r$hi, fmt = "percent", note = note)
    }
  }
  pct("stream_flagged_share", paste0("flagged_share_stream_", c1_code), "95% CI clusters on videos")
  pct(paste0("stream_flagged_share_", c2_code), paste0("flagged_share_stream_", c2_code),
      paste0("at the threshold of coverage ", c2_code, "% fixed on Center 1"))
  pct("stream_c1_flagged_share", paste0("flagged_share_center1_oof_", c1_code), "Center 1 out-of-fold frames above the same threshold")
  pct("stream_c2_flagged_share", paste0("flagged_share_center2_", c1_code),
      "Center 2 frames above the Center 1 threshold; CI clusters on patients")
  pct("stream_video_flagged_median", "video_flagged_share_median")
  pct("stream_video_flagged_min", "video_flagged_share_min")
  pct("stream_video_flagged_max", "video_flagged_share_max")
  pct("stream_outside_range", "outside_range_any", "share of stream frames outside the Center 1 training range on at least one metric")
  reg("stream_n_runs", as.integer(ov("n_runs")$est), note = "runs of consecutive flagged frames")
  reg("stream_run_median_s", ov("run_median_s")$est, digits = 1, unit = "s")
  reg("stream_run_max_s", ov("run_max_s")$est, digits = 0, unit = "s")
  reg("stream_long_run_s", STREAM_LONG_RUN_S, unit = "s", note = "a run at least this long counts as sustained")
  pct("stream_share_in_long_runs", "share_frames_in_long_runs", "share of all analysed frames lying in a sustained run")
  pct("stream_flagged_in_long_runs", "share_flagged_in_long_runs", "share of flagged frames lying in a sustained run")
  baselines <- c(ens = "ens_dis", entropy = "entropy_fov")
  for (k in names(baselines)) {
    r <- ov(paste0("spearman_gate_", baselines[[k]]))
    .gl_reg_est_ci(paste0("stream_spearman_", k), r$est, r$lo, r$hi, 2, note = "gate log-odds vs the baseline score; CI clusters on videos")
  }
}

if (sys.nframe() == 0L) {
  paths <- default_paths()
  t0 <- Sys.time()
  B <- CFG$statistics$bootstrap_resamples
  res <- run_stream(paths, B)
  register_07(res)
  guard_write(file.path(paths$results, "results", "parts", "07_stream.json"))
  write_results_part("07_stream", root = paths$results)
  sm <- res$summary[res$summary$scope == "overall", ]
  cat(sprintf("07 stream: %d frames analysed of %d (%d without a field of view), %d videos\n",
              nrow(res$flags), res$n_total, res$n_excluded_fov, length(unique(res$flags$video_patient))))
  print(sm[, c("statistic", "est", "lo", "hi")], digits = 3, row.names = FALSE)
  cat(sprintf("(%.1f s, B = %d)\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), B))
}
