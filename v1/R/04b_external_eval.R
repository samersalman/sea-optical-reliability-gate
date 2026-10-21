# 04b: evaluate the locked gate and baselines, internal (Center 1 out-of-fold) and external (Center 2).
# Stops unless lock_fields_sha256 (threshold, margin, epochs, entropy choice, object paths and hashes) and every locked
# RDS still match what 04a recorded; Center 2 rows are read only after that check (verify_lock_hashes).
#   Rscript R/04b_external_eval.R
# Outputs results/stats/gate_internal.csv, gate_external.csv, gate_paired_diffs.csv, risk_coverage_curves.csv,
#         gate_range_check.csv;
#         results/parts/04b_external_eval.json

if (!exists("run_locked_evaluation", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

run_external_eval <- function(paths = default_paths(), B = CFG$statistics$bootstrap_resamples, cores = boot_cores()) {
  res <- run_locked_evaluation(paths, "unet", B, cores)
  write_csv_guarded(res$internal$table, file.path(paths$stats, "gate_internal.csv"))
  write_csv_guarded(res$external$table, file.path(paths$stats, "gate_external.csv"))
  write_csv_guarded(rbind(res$internal$diffs, res$external$diffs), file.path(paths$stats, "gate_paired_diffs.csv"))
  write_csv_guarded(rbind(res$internal$curves, res$external$curves), file.path(paths$stats, "risk_coverage_curves.csv"))
  write_csv_guarded(rbind(res$internal$range, res$external$range), file.path(paths$stats, "gate_range_check.csv"))
  res
}

register_04b <- function(res, B) {
  reg("gate_boot_resamples", as.integer(B))
  register_gate_results(res$internal, "gate_int")
  register_gate_results(res$external, "gate_ext")
  register_counts(res$external, "gate_ext")
  reg("gate_ext_fov_excluded", as.integer(res$external$n_excluded_fov),
      note = "Center 2 frames with fov_found == FALSE dropped by 04b (0 when predictions.parquet already excludes them)")
  rc <- res$external$range
  reg("gate_ext_outside_range", rc$frac_outside[rc$feature == "any_feature"], fmt = "percent",
      note = "share of Center 2 frames with at least one optical metric outside the Center 1 training range")
  margin <- res$lock$noninferiority_margin_aurc
  if (is.numeric(margin) && length(margin) == 1L) {
    reg("gate_noninf_margin", margin, digits = 3, note = "AURC units, from the lock file")
    df <- res$external$diffs
    for (cmp in c("gate_vs_entropy", "gate_vs_ens")) {
      reg(paste0("gate_ext_noninferior_", cmp), as.integer(df$noninferior[df$comparison == cmp & df$metric == "aurc"]),
          note = "1 = upper 95% limit of the AURC difference is below the margin")
    }
  }
}

if (sys.nframe() == 0L) {
  paths <- default_paths()
  t0 <- Sys.time()
  B <- CFG$statistics$bootstrap_resamples
  res <- run_external_eval(paths, B)
  register_04b(res, B)
  guard_write(file.path(paths$results, "results", "parts", "04b_external_eval.json"))
  write_results_part("04b_external_eval", root = paths$results)
  for (part in c("internal", "external")) {
    t <- res[[part]]$table
    cat(sprintf("04b %s: %d frames, %d patients, %d failures in %d patients%s; rows read %d\n", part, t$n_frames[1],
                t$n_patients[1], t$n_events[1], t$n_patients_with_events[1],
                if (t$binary_descriptive[1]) " (binary metrics descriptive only)" else "", res[[part]]$rows_loaded))
    print(t[, c("method", "aurc", "aurc_lo", "aurc_hi", "auroc", "sel_risk_90", "cal_slope")], digits = 3, row.names = FALSE)
  }
  cat(sprintf("(%.1f s, B = %d)\n", as.numeric(difftime(Sys.time(), t0, units = "secs")), B))
}
