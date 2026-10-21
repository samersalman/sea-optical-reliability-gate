# 04a: fit the optical gate and its comparators on Center 1 ONLY, cross-fit them for the internal evaluation,
# save the objects and write their SHA-256 into config/analysis_lock.yaml. Run before any Center 2 outcome is read.
#   Rscript R/04a_gate_fit_lock.R            (OPTICS_DATA=synthetic for development; OPTICS_RELOCK=1 to replace a lock)
# Inputs  derived/predictions.parquet (partition center1_labeled, model unet), optics.parquet, fov.parquet, folds.csv
# Outputs work/models/gate_optical.rds, gate_area.rds, gate_stack.rds, gate_folds.rds; lock keys gate_rds_*,
#         gate_area_rds_*, gate_stack_rds_*, gate_folds_rds_* (path, sha256), entropy_baseline_choice, locked_at, and as
#         the last step lock_fields_sha256 (hash of every other lock key but locked_at; verify_lock_hashes checks it);
#         results/parts/04a_gate_fit_lock.json

if (!exists("fit_gate", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

INNER_FOLDS <- CFG$segmenters$cv_folds   # inner CV for lambda uses the same number of patient-grouped folds

.gate_on <- function(d) fit_gate(d, d$fail, OPTICAL_METRICS, patient_foldid(d$patient_id, INNER_FOLDS))

# one gate per held-out segmenter fold, fitted on the other folds; link = out-of-fold log-odds for every row
cross_fit_gates <- function(d) {
  gates <- list()
  link <- rep(NA_real_, nrow(d))
  for (f in sort(unique(d$fold))) {
    te <- d$fold == f
    gates[[as.character(f)]] <- .gate_on(d[!te, ])
    link[te] <- predict_gate(gates[[as.character(f)]], d[te, ], "link")
  }
  list(gates = gates, link = link)
}

fit_and_lock <- function(paths = default_paths(), relock = identical(Sys.getenv("OPTICS_RELOCK"), "1")) {
  lock <- read_lock(paths$lock)
  if (.gl_nonempty(lock$gate_rds_sha256) && !relock) {
    stop("gate_rds_sha256 is already recorded in ", paths$lock, "; set OPTICS_RELOCK=1 to replace it", call. = FALSE)
  }
  thr <- lock_failure_threshold(lock)

  d <- read_eval_frames(paths$derived, "center1_labeled", "unet")   # Arrow filter: no other partition is loaded
  stopifnot(!any(d$partition != "center1_labeled"))
  d$fail <- d$iou < thr
  d$risk <- 1 - d$iou

  outer <- cross_fit_gates(d)
  area_f <- stack_f <- list()
  for (f in names(outer$gates)) {
    tr <- d[d$fold != as.integer(f), ]
    area_f[[f]] <- fit_area(tr, tr$fail)
    inner <- cross_fit_gates(tr)   # the stack is trained on out-of-fold gate scores that never saw fold f
    stack_f[[f]] <- fit_stack(inner$link, tr$ens_dis, tr$fail, outer$gates[[f]])
  }

  gate <- .gate_on(d)
  cov <- unlist(CFG$endpoints$coverage_levels)
  gate$flag_thresholds <- setNames(unname(stats::quantile(outer$link, cov, type = 1)), as.character(cov))
  area <- fit_area(d, d$fail)
  stack <- fit_stack(outer$link, d$ens_dis, d$fail, gate)

  entropy_aurc <- c(entropy_fov = aurc(d$entropy_fov, d$risk), entropy_roi = aurc(d$entropy_roi, d$risk))
  choice <- names(entropy_aurc)[which.min(entropy_aurc)]

  objects <- list(gate_rds = list(gate, "gate_optical.rds"), gate_area_rds = list(area, "gate_area.rds"),
                  gate_stack_rds = list(stack, "gate_stack.rds"),
                  gate_folds_rds = list(list(optical_gate = outer$gates, predicted_area = area_f,
                                             gate_plus_ensemble = stack_f), "gate_folds.rds"))
  values <- list()
  for (key in names(objects)) {
    rel <- file.path(paths$models_rel, objects[[key]][[2]])
    path <- guard_write(p(rel))
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(objects[[key]][[1]], path)
    values[[paste0(key, "_path")]] <- rel
    values[[paste0(key, "_sha256")]] <- unname(tools::sha256sum(path))
  }
  values$entropy_baseline_choice <- choice
  values$locked_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  lock_update(paths$lock, values, init = if (file.exists(paths$lock)) list() else lock)
  # last step: seal every frozen field (threshold, margin, epochs, entropy choice, object paths and hashes)
  lock_update(paths$lock, list(lock_fields_sha256 = lock_fields_hash(yaml::read_yaml(paths$lock))))

  list(rows_loaded = attr(d, "rows_loaded"), rows_in_file = attr(d, "rows_in_file"), partitions = unique(d$partition),
       n_excluded_fov = attr(d, "n_excluded_fov"),
       n_frames = nrow(d), n_patients = length(unique(d$patient_id)), n_events = sum(d$fail),
       n_patients_with_events = length(unique(d$patient_id[d$fail])), threshold = thr,
       lambda = gate$lambda, spline_df = gate$spec$df, elastic_alpha = gate$alpha,
       entropy_aurc = entropy_aurc, entropy_choice = choice,
       flag_thresholds = gate$flag_thresholds, hashes = values[grepl("_sha256$", names(values))],
       oof_link = outer$link, frame_id = d$frame_id)
}

register_04a <- function(sm) {
  reg("gate_c1_n_frames", as.integer(sm$n_frames))
  reg("gate_c1_n_patients", as.integer(sm$n_patients))
  reg("gate_c1_failure_events", as.integer(sm$n_events))
  reg("gate_c1_failure_patients", as.integer(sm$n_patients_with_events))
  reg("gate_c1_failure_rate", sm$n_events / sm$n_frames, fmt = "percent")
  reg("gate_c1_fov_excluded", as.integer(sm$n_excluded_fov),
      note = "Center 1 frames with fov_found == FALSE dropped by 04a (0 when predictions.parquet already excludes them)")
  reg("gate_failure_iou", sm$threshold, digits = 2)
  reg("gate_n_features", length(OPTICAL_METRICS))
  reg("gate_spline_df", as.integer(sm$spline_df))
  reg("gate_elastic_alpha", sm$elastic_alpha, digits = 1)
  reg("gate_inner_folds", as.integer(INNER_FOLDS))
  reg("gate_c1_lambda", sm$lambda, digits = 4, note = "1-SE penalty of the final gate")
  reg("gate_entropy_fov_aurc_c1", sm$entropy_aurc[["entropy_fov"]], digits = 3, note = "Center 1 OOF, U-Net")
  reg("gate_entropy_roi_aurc_c1", sm$entropy_aurc[["entropy_roi"]], digits = 3, note = "Center 1 OOF, U-Net")
  reg("gate_entropy_choice", if (sm$entropy_choice == "entropy_fov") "whole field of view" else "region around the predicted mask",
      note = paste("lower Center 1 OOF AURC;", sm$entropy_choice))
}

if (sys.nframe() == 0L) {
  paths <- default_paths()
  t0 <- Sys.time()
  sm <- fit_and_lock(paths)
  register_04a(sm)
  guard_write(file.path(paths$results, "results", "parts", "04a_gate_fit_lock.json"))
  write_results_part("04a_gate_fit_lock", root = paths$results)
  cat(sprintf("04a: %d Center 1 frames (%d rows read of %d in the file; partitions: %s), %d patients, %d failures (IoU < %.2f)\n",
              sm$n_frames, sm$rows_loaded, sm$rows_in_file, paste(sm$partitions, collapse = ","), sm$n_patients,
              sm$n_events, sm$threshold))
  cat(sprintf("     lambda(1se) %.4f; entropy AURC fov %.4f roi %.4f -> %s\n", sm$lambda, sm$entropy_aurc[["entropy_fov"]],
              sm$entropy_aurc[["entropy_roi"]], sm$entropy_choice))
  cat(sprintf("     gate sha256 %s\n     lock %s (%.1f s)\n", sm$hashes$gate_rds_sha256, paths$lock,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
