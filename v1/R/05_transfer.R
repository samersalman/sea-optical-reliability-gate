# 05: apply the U-Net-trained gate and comparators UNCHANGED to SegFormer outcomes (risk = 1 - IoU of the SegFormer
# rows), internal out-of-fold and external. Each model's own predicted area and entropy feed the baselines; the gate
# sees only optics. Lock verification first (lock_fields_sha256 and the RDS hashes), as in 04b.
#   Rscript R/05_transfer.R
# Outputs results/stats/gate_transfer.csv, gate_transfer_paired_diffs.csv, risk_coverage_curves_transfer.csv;
#         results/parts/05_transfer.json

if (!exists("run_locked_evaluation", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "stats_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

run_transfer <- function(paths = default_paths(), B = CFG$statistics$bootstrap_resamples, cores = boot_cores()) {
  res <- run_locked_evaluation(paths, "segformer", B, cores)
  write_csv_guarded(rbind(res$internal$table, res$external$table), file.path(paths$stats, "gate_transfer.csv"))
  write_csv_guarded(rbind(res$internal$diffs, res$external$diffs), file.path(paths$stats, "gate_transfer_paired_diffs.csv"))
  write_csv_guarded(rbind(res$internal$curves, res$external$curves), file.path(paths$stats, "risk_coverage_curves_transfer.csv"))
  res
}

register_05 <- function(res) {
  for (part in c("internal", "external")) {
    prefix <- paste0("transfer_", substr(part, 1, 3))
    register_gate_results(res[[part]], prefix)
    register_counts(res[[part]], prefix)
  }
}

if (sys.nframe() == 0L) {
  paths <- default_paths()
  t0 <- Sys.time()
  res <- run_transfer(paths)
  register_05(res)
  guard_write(file.path(paths$results, "results", "parts", "05_transfer.json"))
  write_results_part("05_transfer", root = paths$results)
  for (part in c("internal", "external")) {
    t <- res[[part]]$table
    cat(sprintf("05 %s: %d frames, %d failures in %d patients%s\n", part, t$n_frames[1], t$n_events[1],
                t$n_patients_with_events[1], if (t$binary_descriptive[1]) " (binary metrics descriptive only)" else ""))
    print(t[, c("method", "aurc", "aurc_lo", "aurc_hi", "auroc", "sel_risk_90", "cal_slope")], digits = 3, row.names = FALSE)
  }
  cat(sprintf("(%.1f s)\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
