# 02: cohort counts and acquisition-domain breakdowns from the manifest and the FOV table (no segmenter output).
# Reconciles to dataset_counts_expected in the plan (real data only), registers the numbers and writes
# results/parts/02_cohort.json. compute_cohort() is also what Figure 1A and Table 1 panel A read, so the
# exhibits and the registered numbers cannot drift apart.
#   Rscript R/02_cohort.R

if (!exists("load_manifest", mode = "function") || !exists("reg", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (exists("V1", envir = globalenv())) file.path(get("V1", envir = globalenv()), "R")
             else if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

COHORT_STRATA <- c("channel", "region", "scenario", "instrument", "file_format")

quart3 <- function(x) setNames(unname(quantile(x, c(0.25, 0.5, 0.75), type = 7)), c("q1", "median", "q3"))

# Printed and registered quartiles and medians are rounded half up. R's round() is half-even, which would print the
# D3 third quartile 34.5 as "34" but the D2 first quartile 49.5 as "50".
round_half_up <- function(x) floor(x + 0.5)
whole_half_up <- function(value, digits) fmt_num(round_half_up(value), digits)

# Resolution and centre of an acquisition domain, from the domains: section of the plan
domain_size_label <- function(d) sprintf("%d × %d", as.integer(CFG$domains[[d]]$size[[1]]), as.integer(CFG$domains[[d]]$size[[2]]))
domain_center <- function(d) as.integer(CFG$domains[[d]]$center)

# One row per stratum level and group (D1, D2, D3, Total): frames, patients, share of the group's frames.
cohort_strata <- function(lab, groups) {
  rows <- lapply(COHORT_STRATA, function(v) {
    lapply(names(groups), function(g) {
      d <- lab[groups[[g]](lab), ]
      lv <- sort(unique(lab[[v]]))
      do.call(rbind, lapply(lv, function(l) {
        sel <- d[[v]] == l
        data.frame(variable = v, level = l, group = g, n_frames = sum(sel),
                   n_patients = n_distinct(d$patient_id[sel]), n_group = nrow(d), stringsAsFactors = FALSE)
      }))
    })
  })
  out <- do.call(rbind, unlist(rows, recursive = FALSE))
  out$share <- out$n_frames / out$n_group
  tibble::as_tibble(out)
}

compute_cohort <- function(manifest = load_manifest(), fov = load_fov()) {
  m <- left_join(manifest, fov, by = "frame_id")
  lab <- filter(m, labeled)
  stream <- filter(m, partition == "stream")
  domains <- names(DOMAIN_COLOURS)

  groups <- c(setNames(lapply(domains, function(d) function(x) x$domain == d), domains),
              Total = function(x) rep(TRUE, nrow(x)))
  per_patient <- lab |> count(patient_id, domain, name = "n_frames")

  summarise_group <- function(sel_frames, sel_patients, label) {
    d <- lab[sel_frames, ]
    pp <- per_patient[sel_patients, ]
    found <- d[d$fov_found, ]
    f <- quart3(pp$n_frames)
    dia <- quart3(2 * found$fov_r)
    tibble::tibble(
      group = label, n_patients = nrow(pp), n_frames = nrow(d),
      fpp_q1 = f[["q1"]], fpp_median = f[["median"]], fpp_q3 = f[["q3"]], fpp_min = min(pp$n_frames), fpp_max = max(pp$n_frames),
      n_difficult = sum(d$scenario == "Difficult"), n_normal = sum(d$scenario == "Normal"),
      n_fov_found = nrow(found), n_fov_notfound = nrow(d) - nrow(found),
      diam_q1 = dia[["q1"]], diam_median = dia[["median"]], diam_q3 = dia[["q3"]]
    )
  }
  by_group <- bind_rows(
    lapply(domains, function(dm) summarise_group(lab$domain == dm, per_patient$domain == dm, dm)),
    summarise_group(rep(TRUE, nrow(lab)), rep(TRUE, nrow(per_patient)), "Total")
  )

  analysed <- function(x) sum(x$fov_found)
  counts <- list(
    n_images_total = nrow(m), n_frames_labeled = nrow(lab), n_patients_labeled = n_distinct(lab$patient_id),
    n_patients_c1 = n_distinct(lab$patient_id[lab$center == 1]), n_patients_c2 = n_distinct(lab$patient_id[lab$center == 2]),
    n_frames_c1 = sum(lab$center == 1), n_frames_c2 = sum(lab$center == 2),
    n_frames_stream = nrow(stream), n_patients_stream = n_distinct(stream$patient_id),
    n_frames_c1_analysed = analysed(lab[lab$center == 1, ]), n_frames_c2_analysed = analysed(lab[lab$center == 2, ]),
    n_frames_stream_analysed = analysed(stream), n_fov_found = analysed(m), n_fov_total = nrow(m)
  )
  counts$n_frames_labeled_analysed <- counts$n_frames_c1_analysed + counts$n_frames_c2_analysed
  counts$n_excl_fov_c1 <- counts$n_frames_c1 - counts$n_frames_c1_analysed
  counts$n_excl_fov_c2 <- counts$n_frames_c2 - counts$n_frames_c2_analysed
  counts$n_excl_fov_stream <- counts$n_frames_stream - counts$n_frames_stream_analysed
  counts$n_excl_fov_total <- counts$n_fov_total - counts$n_fov_found

  list(counts = counts, groups = by_group, strata = cohort_strata(lab, groups), per_patient = per_patient,
       n_videos_stream = n_distinct(stream$patient_id))
}

# The plan's expected counts hold for the real dataset only; the synthetic cohort mirrors patients, not frames.
# Besides the nine dataset counts, the frames without a detected field of view must match the Gate G1 exclusions
# (stream: section of the plan): Center 1 labeled, Center 2 labeled and stream.
check_cohort_counts <- function(counts, expected = CFG$dataset_counts_expected, excluded = CFG$stream, strict = !SYNTH) {
  map <- c(images_total = "n_images_total", labeled_frames = "n_frames_labeled", labeled_patients = "n_patients_labeled",
           center1_patients = "n_patients_c1", center2_patients = "n_patients_c2", center1_frames = "n_frames_c1",
           center2_frames = "n_frames_c2", stream_frames = "n_frames_stream", stream_patients = "n_patients_stream",
           excluded_center1_labeled = "n_excl_fov_c1", excluded_center2_labeled = "n_excl_fov_c2",
           excluded_stream = "n_excl_fov_stream")
  want <- c(unlist(expected[names(map)[1:9]]),
            excluded_center1_labeled = excluded$expected_not_found_center1_labeled,
            excluded_center2_labeled = excluded$expected_not_found_center2_labeled,
            excluded_stream = excluded$expected_not_found)
  if (length(want) != length(map) || anyNA(want)) stop("analysis_plan.yaml lacks expected counts or stream.expected_not_found*", call. = FALSE)
  got <- setNames(unlist(counts[map]), names(map))
  bad <- names(map)[got != want[names(map)]]
  if (length(bad)) {
    msg <- paste0(bad, ": got ", got[bad], ", expected ", want[bad], collapse = "; ")
    if (strict) stop("cohort counts do not reconcile to analysis_plan.yaml: ", msg, call. = FALSE)
    message("synthetic mode: counts differ from the real dataset (", msg, ")")
  }
  invisible(length(bad) == 0L)
}

cohort_stratum <- function(coh, variable, level, group = "Total") {
  r <- coh$strata[coh$strata$variable == variable & coh$strata$level == level & coh$strata$group == group, ]
  stopifnot(nrow(r) == 1L)
  r
}

register_cohort <- function(coh) {
  for (k in names(coh$counts)) reg(k, as.integer(coh$counts[[k]]))
  reg("fov_found_rate", coh$counts$n_fov_found / coh$counts$n_fov_total, fmt = "percent", digits = 2,
      note = "frames with a detected field of view, all 48,510 images")
  reg("n_cv_folds", as.integer(CFG$segmenters$cv_folds), note = "patient-grouped folds inside Center 1")

  g <- coh$groups
  for (i in seq_len(nrow(g))) {
    s <- if (g$group[i] == "Total") "lab" else tolower(g$group[i])   # "lab" = all labeled frames
    r <- g[i, ]
    if (s != "lab") {
      reg(paste0("n_patients_", s), as.integer(r$n_patients))
      reg(paste0("n_frames_", s), as.integer(r$n_frames))
    }
    for (k in c("fpp_q1", "fpp_median", "fpp_q3")) {
      reg(paste0(k, "_", s), r[[k]], digits = 0, fmt = whole_half_up, note = "labeled frames per patient, shown to whole frames (rounded half up)")
    }
    reg(paste0("fpp_min_", s), as.integer(r$fpp_min))
    reg(paste0("fpp_max_", s), as.integer(r$fpp_max))
    reg(paste0("n_difficult_", s), as.integer(r$n_difficult))
    reg(paste0("n_normal_", s), as.integer(r$n_normal))
    reg(paste0("pct_difficult_", s), r$n_difficult / r$n_frames, fmt = "percent")
    reg(paste0("pct_normal_", s), r$n_normal / r$n_frames, fmt = "percent")
    reg(paste0("n_fov_notfound_", s), as.integer(r$n_fov_notfound))
    reg(paste0("fov_found_rate_", s), r$n_fov_found / r$n_frames, fmt = "percent", digits = 2, note = "labeled frames")
    for (k in c("diam_q1", "diam_median", "diam_q3")) {
      reg(paste0("fov_", k, "_", s), r[[k]], digits = 0, fmt = whole_half_up, unit = "px",
          note = "2 * fov_r, native pixels, frames with a found field of view (D3: inscribed circle), rounded half up")
    }
    reg(paste0("pct_png_", s), cohort_stratum(coh, "file_format", "png", r$group)$share, fmt = "percent")
  }

  for (l in c("big", "small")) {
    reg(paste0("n_channel_", l, "_frames"), as.integer(cohort_stratum(coh, "channel", l)$n_frames))
    reg(paste0("n_channel_", l, "_patients"), as.integer(cohort_stratum(coh, "channel", l)$n_patients))
  }
  reg("pct_channel_big", cohort_stratum(coh, "channel", "big")$share, fmt = "percent")
  for (l in c("cervical", "lumbar")) {
    reg(paste0("n_region_", l, "_frames"), as.integer(cohort_stratum(coh, "region", l)$n_frames))
    reg(paste0("n_region_", l, "_patients"), as.integer(cohort_stratum(coh, "region", l)$n_patients))
  }
  for (l in sort(unique(coh$strata$level[coh$strata$variable == "instrument"]))) {
    reg(paste0("n_instr_", l, "_frames"), as.integer(cohort_stratum(coh, "instrument", l)$n_frames))
  }
  for (l in c("png", "jpg")) reg(paste0("n_format_", l, "_frames"), as.integer(cohort_stratum(coh, "file_format", l)$n_frames))
  invisible(NULL)
}

if (sys.nframe() == 0L) {
  coh <- compute_cohort()
  check_cohort_counts(coh$counts)
  register_cohort(coh)
  part <- write_results_part("02_cohort")
  cat("02_cohort: wrote", part, "\n")
  cc <- coh$counts
  cat(sprintf("labeled %s frames / %s patients (Center 1 %s / %s, Center 2 %s / %s); stream %s frames / %s patients\n",
              fmt_num(cc$n_frames_labeled, 0), cc$n_patients_labeled, fmt_num(cc$n_frames_c1, 0), cc$n_patients_c1,
              fmt_num(cc$n_frames_c2, 0), cc$n_patients_c2, fmt_num(cc$n_frames_stream, 0), cc$n_patients_stream))
  cat(sprintf("FOV not found: %d of %s images (Center 1 labeled %d, Center 2 %d, stream %d); analysed Center 1 %s, Center 2 %s, stream %s\n",
              cc$n_excl_fov_total, fmt_num(cc$n_fov_total, 0), cc$n_excl_fov_c1, cc$n_excl_fov_c2, cc$n_excl_fov_stream,
              fmt_num(cc$n_frames_c1_analysed, 0), fmt_num(cc$n_frames_c2_analysed, 0), fmt_num(cc$n_frames_stream_analysed, 0)))
  g <- coh$groups
  print(as.data.frame(transform(g, pct_difficult = round(100 * n_difficult / n_frames, 1))[
    c("group", "n_patients", "n_frames", "fpp_median", "fpp_q1", "fpp_q3", "fpp_min", "fpp_max", "pct_difficult",
      "n_fov_notfound", "diam_median")]), row.names = FALSE)
}
