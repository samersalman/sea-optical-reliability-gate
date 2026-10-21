# 21: supplementary tables. Every cell is read from a CSV under results/stats/ (03, 04b, 07, 08 and the post hoc 34
# diagnostics), so nothing is recomputed and no number is typed. Three-line style, flextable -> docx + csv, as Tables 1 to 3
# (the small writers below mirror the tbl_* helpers of 20_tables.R on purpose: this script does not source it, so a change
# there cannot break it). Registers no keys.
#   Table S1   metric distributions by domain and scenario, with the within-domain AUROC for Difficult (assoc_by_domain.csv)
#   Table S2   exploratory and secondary results: sensitivity variants with their non-zero coefficient counts, calibration
#              intercepts, specificity at fixed sensitivity, selective risk at 80% coverage, patient-averaged AURC,
#              within-domain internal results, flagged shares and stream correlations
#   Table S3   covariate shift between the centers and robustness of the odds ratios
#   Rscript R/21_supplement.R            (OPTICS_DATA=synthetic for development)
# Writes tables/TableS1.{docx,csv}, TableS2.{docx,csv}, TableS3.{docx,csv}; a table whose inputs are missing is skipped with a message.
# The tables that carry external-center results are built only after the analysis lock verifies (stats_lib.R verify_lock_hashes).

if (!exists("flex_three_line", mode = "function") || !exists("verify_lock_hashes", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (exists("V1", envir = globalenv())) file.path(get("V1", envir = globalenv()), "R")
             else if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "stats_lib.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

# the "95%" wording of the footnotes is typed, so refuse a plan that changes the level
stopifnot(identical(CFG$statistics$ci_level, 0.95))
SUP_STATUS <- "Post hoc diagnostic, not part of the locked or pre-specified analysis; descriptive, with no multiplicity adjustment."
SUP_FONT <- 7.5

# ---- small helpers -----------------------------------------------------------------------------------------------

sup_role <- function(center) {
  if (exists("CENTER_ROLE")) unname(CENTER_ROLE[as.character(center)]) else paste("Center", center)
}

sup_domain_label <- function(d) {
  dm <- CFG$domains[[d]]
  sprintf("%s: %s, %d × %d", d, sup_role(dm$center), as.integer(dm$size[[1]]), as.integer(dm$size[[2]]))
}

sup_read <- function(name, dir = p("results", "stats")) {
  path <- file.path(dir, name)
  if (!file.exists(path)) stop("missing input ", path, " (run the upstream script first)", call. = FALSE)
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = c("NA", ""))
}

sup_est_ci <- function(est, lo, hi, digits = 2) {
  ifelse(is.na(est), "", ifelse(is.na(lo) | is.na(hi), fmt_num(est, digits),
                                sprintf("%s (%s to %s)", fmt_num(est, digits), fmt_num(lo, digits), fmt_num(hi, digits))))
}

# an estimate that can run to thousands (a wide odds-ratio interval): whole numbers from 100 up
sup_wide <- function(x) ifelse(is.na(x), "", ifelse(abs(x) >= 100, fmt_num(x, 0), fmt_num(x, 2)))
sup_wide_ci <- function(est, lo, hi) ifelse(is.na(est), "", sprintf("%s (%s to %s)", sup_wide(est), sup_wide(lo), sup_wide(hi)))

sup_sci <- function(x, digits = 1) {
  e <- floor(log10(abs(x)))
  m <- round(x / 10^e, digits)
  up <- abs(m) >= 10
  m[up] <- m[up] / 10
  e[up] <- e[up] + 1
  sprintf("%.*fe%d", digits, m, as.integer(e))
}

# metric values: three decimals, scientific notation below the third decimal (as sharpness in Table 2)
sup_metric_val <- function(x) {
  out <- fmt_num(x, 3)
  small <- is.finite(x) & x != 0 & abs(x) < 0.001
  out[small] <- sup_sci(x[small])
  out[!is.finite(x)] <- ""
  out
}
sup_metric_iqr <- function(med, q1, q3) paste0(sup_metric_val(med), " [", sup_metric_val(q1), " to ", sup_metric_val(q3), "]")

sup_pct <- function(x, digits = 1) ifelse(is.na(x), "", fmt_num(100 * x, digits))
sup_pct_ci <- function(est, lo, hi, digits = 1) {
  ifelse(is.na(est), "", ifelse(is.na(lo) | is.na(hi), sup_pct(est, digits),
                                sprintf("%s (%s to %s)", sup_pct(est, digits), sup_pct(lo, digits), sup_pct(hi, digits))))
}

sup_yes <- function(x) ifelse(is.na(x), "", ifelse(x, "Yes", "No"))

# One row of a CSV/data frame by conditions, stops on 0 or several matches
sup_row <- function(df, ...) {
  cond <- list(...)
  keep <- rep(TRUE, nrow(df))
  for (nm in names(cond)) keep <- keep & !is.na(df[[nm]]) & df[[nm]] == cond[[nm]]
  if (sum(keep) != 1L) stop("expected one row for ", paste(names(cond), unlist(cond), sep = "=", collapse = ", "), ", found ", sum(keep), call. = FALSE)
  df[keep, , drop = FALSE]
}

# The lock must verify before a table that carries external-center results is built; a synthetic lock that predates
# lock_fields_sha256 is read as is (stats_lib.R / 20_tables.R do the same).
sup_lock <- function() {
  path <- p("config", "analysis_lock.yaml")
  if (!file.exists(path)) return(list())
  lock <- yaml::read_yaml(path)
  sealed <- any(vapply(c("gate_rds_sha256", "lock_fields_sha256"), function(k) .gl_nonempty(lock[[k]]), NA))
  if (sealed && (!SYNTH || .gl_nonempty(lock$lock_fields_sha256))) verify_lock_hashes(lock)
  lock
}

# ---- writers (mirror 20_tables.R: A4 landscape, 0.8 in margins, fixed widths, no title paragraphs) -----------------

sup_fix_widths <- function(ft, widths) {
  ft <- flextable::set_table_properties(ft, layout = "fixed")
  ft <- flextable::width(ft, j = seq_along(widths), width = widths)
  ft <- flextable::height_all(ft, height = 0.05, part = "all")
  flextable::valign(ft, valign = "top", part = "body")
}

# panel: list(title, df, widths, align, footnote, bold_rows, indent_rows, rule_after, merge_rows, labels, spanners, break_before)
#   labels    named character vector colname -> header text (a "\n" breaks the header line)
#   spanners  list(values, colwidths) for one extra header row above the column headings (domain or partition groups)
sup_flextable <- function(panel, font_size = panel$font_size %||% SUP_FONT) {
  df <- panel$df
  d <- df
  for (cl in names(d)[-1]) d[[cl]] <- gsub(" \\(", "\n(", gsub(" \\[", "\n[", d[[cl]]))   # interval on its own line, as in Tables 2 and 3
  ft <- flex_three_line(d, align = panel$align, font_size = font_size, bold_rows = panel$bold_rows,
                        indent_rows = panel$indent_rows, rule_after = panel$rule_after, footnote = panel$footnote)
  if (!is.null(panel$labels)) ft <- flextable::set_header_labels(ft, values = as.list(panel$labels))
  heavy <- officer::fp_border(color = "black", width = 1.5)
  thin <- officer::fp_border(color = "black", width = 0.75)
  if (!is.null(panel$spanners)) {
    sp <- panel$spanners
    ft <- flextable::add_header_row(ft, values = sp$values, colwidths = sp$colwidths, top = TRUE)
  }
  ft <- flextable::add_header_row(ft, values = panel$title, colwidths = ncol(df), top = TRUE)
  ft <- flextable::align(ft, part = "header", align = "center")
  ft <- flextable::align(ft, j = 1, part = "header", align = "left")
  ft <- flextable::align(ft, i = 1, part = "header", align = "left")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::border_remove(ft)
  ft <- flextable::hline_top(ft, border = heavy, part = "header")
  if (!is.null(panel$spanners)) {
    j0 <- 0L
    for (k in seq_along(panel$spanners$colwidths)) {
      w <- panel$spanners$colwidths[k]
      if (nzchar(panel$spanners$values[k])) ft <- flextable::hline(ft, i = 2, j = j0 + seq_len(w), border = thin, part = "header")
      j0 <- j0 + w
    }
  }
  ft <- flextable::hline_bottom(ft, border = thin, part = "header")
  ft <- flextable::hline_bottom(ft, border = heavy, part = "body")
  if (!is.null(panel$rule_after)) ft <- flextable::hline(ft, i = panel$rule_after, border = thin, part = "body")
  for (r in panel$merge_rows) ft <- flextable::merge_at(ft, i = r, j = seq_len(ncol(df)))
  ft <- flextable::padding(ft, j = seq_len(ncol(df))[-1], padding.left = 2, padding.right = 2, part = "all")
  sup_fix_widths(ft, panel$widths)
}

sup_write_docx <- function(fts, path, landscape = TRUE, new_page = integer()) {
  guard_write(path)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  doc <- officer::read_docx()
  doc <- officer::body_set_default_section(doc, officer::prop_section(
    page_size = officer::page_size(orient = if (landscape) "landscape" else "portrait"),
    page_margins = officer::page_mar(top = 0.8, bottom = 0.8, left = 0.8, right = 0.8, header = 0.4, footer = 0.4),
    type = "continuous"))
  for (i in seq_along(fts)) {
    if (i > 1L) doc <- if (i %in% new_page) officer::body_add_break(doc) else officer::body_add_par(doc, "")
    doc <- flextable::body_add_flextable(doc, fts[[i]], align = "center")
  }
  print(doc, target = path)
  invisible(path)
}

# panels stacked into one csv: a title row, the panel's column names as a row, then its rows, padded to the widest panel
sup_stack_csv <- function(panels) {
  w <- max(vapply(panels, function(pn) ncol(pn$df), 1L))
  pad <- function(m) cbind(m, matrix("", nrow(m), w - ncol(m)))
  rows <- lapply(panels, function(pn) rbind(pad(matrix(pn$title, 1L)), pad(matrix(names(pn$df), 1L)), pad(as.matrix(pn$df))))
  out <- do.call(rbind, rows)
  colnames(out) <- paste0("V", seq_len(w))
  out
}

sup_save <- function(name, panels, general_footnote) {
  last <- length(panels)
  panels[[last]]$footnote <- c(panels[[last]]$footnote, general_footnote)
  fts <- lapply(panels, sup_flextable)
  brk <- which(vapply(panels, function(pn) isTRUE(pn$break_before), NA))
  paths <- c(docx = sup_write_docx(fts, file.path(p("tables"), paste0(name, ".docx")), new_page = brk))
  csv <- file.path(p("tables"), paste0(name, ".csv"))
  guard_write(csv)
  m <- sup_stack_csv(panels)
  utils::write.table(m, csv, sep = ",", row.names = FALSE, col.names = FALSE, na = "", qmethod = "double", fileEncoding = "UTF-8")
  c(paths, csv = csv)
}

# ---- Table S1: metric distributions by domain and scenario -------------------------------------------------------------

sup_table_s1 <- function() {
  bd <- sup_read("assoc_by_domain.csv")
  doms <- names(CFG$domains)
  cols <- c("Metric", unlist(lapply(doms, function(d) paste(d, c("Normal", "Difficult", "AUROC")))))
  rows <- lapply(OPTICAL_METRICS, function(m) {
    cells <- unlist(lapply(doms, function(d) {
      r <- sup_row(bd, metric = m, domain = d)
      c(sup_metric_iqr(r$median_normal, r$q1_normal, r$q3_normal), sup_metric_iqr(r$median_difficult, r$q1_difficult, r$q3_difficult),
        sup_est_ci(r$auroc, r$auroc_lo, r$auroc_hi, 2))
    }))
    setNames(as.data.frame(as.list(c(METRIC_LABELS[[m]], cells)), stringsAsFactors = FALSE), cols)
  })
  df <- do.call(rbind, rows)
  n_of <- function(d, col) sup_row(bd, metric = OPTICAL_METRICS[1], domain = d)[[col]]
  labels <- c(Metric = "Metric")
  for (d in doms) {
    labels[paste(d, "Normal")] <- sprintf("Normal\nmedian [IQR]\n(n = %s)", fmt_num(n_of(d, "n_normal"), 0))
    labels[paste(d, "Difficult")] <- sprintf("Difficult\nmedian [IQR]\n(n = %s)", fmt_num(n_of(d, "n_difficult"), 0))
    labels[paste(d, "AUROC")] <- "AUROC for Difficult\n(95% CI)"
  }
  sc <- function(d) sup_row(bd, metric = OPTICAL_METRICS[1], domain = d)$scope
  ext <- doms[vapply(doms, function(d) grepl("external", sc(d)), NA)]
  d2 <- sup_row(bd, metric = OPTICAL_METRICS[1], domain = "D2")
  note <- c(
    SUP_STATUS,
    "IQR, interquartile range (first to third quartile); AUROC, area under the receiver operating characteristic curve for separating Difficult from Normal frames inside one domain (one half means no separation; below one half means the metric is lower in Difficult frames); CI, confidence interval; D1, D2 and D3, the acquisition domains named in the column groups (the metrics are defined in Table 2).",
    sprintf("The expert Difficult label is confounded with domain (%s%% of D2 frames are Difficult), which is why the AUROC is given within each domain. The AUROC interval is a patient-clustered bootstrap percentile interval (patients resampled within the domain, %s resamples).",
            fmt_num(100 * d2$n_difficult / (d2$n_normal + d2$n_difficult), 1), fmt_num(CFG$statistics$bootstrap_resamples, 0)),
    sprintf("%s: descriptive only, no inference is drawn from the external center. Values below 0.001 are printed in scientific notation, as for sharpness in Table 2.",
            paste(ext, collapse = " and ")))
  panel <- list(title = "Post hoc diagnostic. Optical metrics by domain and expert scenario", df = df,
                widths = c(1.2, rep(c(1.05, 1.05, 0.85), length(doms))), align = c("l", rep("c", ncol(df) - 1L)),
                labels = labels, footnote = note, font_size = 7,
                spanners = list(values = c("", unname(vapply(doms, sup_domain_label, ""))), colwidths = c(1L, rep(3L, length(doms)))))
  list(panels = list(panel), general = character())
}

# ---- Table S2: exploratory and secondary results ---------------------------------------------------------------------

S2_METHODS <- c("random", "predicted_area", "entropy", "ens_disagreement", "optical_gate", "gate_plus_ensemble")
S2_LABELS <- c(random = "Random", predicted_area = "Predicted area", entropy = "Pixel entropy",
               ens_disagreement = "Ensemble disagreement", optical_gate = "Optical gate", gate_plus_ensemble = "Gate + ensemble",
               oracle = "Oracle")

sup_s2_variants <- function() {
  sens <- sup_read("sensitivity.csv")
  nz <- sup_read("diag_gate_nonzero.csv")
  g <- sens[sens$method == "gate", ]
  rows <- lapply(seq_len(nrow(g)), function(i) {
    r <- g[i, ]
    z <- sup_row(nz, scope = "variant", id = r$variant)
    on <- if (grepl("^center2", r$eval_partition)) sup_role(2) else sprintf("%s, %s", sup_role(1), r$eval_domain)
    terms <- switch(z$model_family,
      xgboost = "not applicable (boosted trees)",
      gam = sprintf("%d of %d smooths", as.integer(z$n_nonzero_terms), as.integer(z$n_candidate_terms)),
      sprintf("%d of %d terms (%d metrics)", as.integer(z$n_nonzero_terms), as.integer(z$n_candidate_terms), as.integer(z$n_nonzero_metrics)))
    lab <- gsub("ormalised", "ormalized", gsub("Center 2", tolower(sup_role(2)), gsub("Center 1", tolower(sup_role(1)), r$label, fixed = TRUE), fixed = TRUE), fixed = TRUE)
    data.frame(Variant = lab, `Evaluated on` = on, `Non-zero penalized terms` = terms, `Constant score` = sup_yes(z$constant_score),
               AURC = sup_est_ci(r$aurc, r$aurc_lo, r$aurc_hi, 3),
               `Gate minus random` = sup_est_ci(r$diff_vs_random, r$diff_vs_random_lo, r$diff_vs_random_hi, 3),
               `Gate minus ensemble` = sup_est_ci(r$diff_vs_ens, r$diff_vs_ens_lo, r$diff_vs_ens_hi, 3),
               check.names = FALSE, stringsAsFactors = FALSE)
  })
  df <- do.call(rbind, rows)
  f <- sup_row(nz, scope = "final_gate", id = "final")
  fo <- nz[nz$scope == "fold_gate", ]
  note <- c(
    sprintf("Each variant is refitted on the training center only (the device-normalized variant also takes its external-center normalizer from external-center optics, without outcomes). The locked gate offers %d natural-spline terms to an elastic-net fit with the penalty chosen by the one-standard-error rule and keeps %d of them (%d metrics; %s failures per non-zero term); the %s fold gates keep %s to %s terms and %d of them kept none (a constant score).",
            as.integer(f$n_candidate_terms), as.integer(f$n_nonzero_terms), as.integer(f$n_nonzero_metrics), fmt_num(f$events_per_nonzero, 0),
            fmt_num(nrow(fo), 0), fmt_num(min(fo$n_nonzero_terms), 0), fmt_num(max(fo$n_nonzero_terms), 0), as.integer(sum(fo$constant_score))),
    "A constant score means every penalized coefficient was zero, so every evaluated frame received the same score and the AURC equals random order. The domain-covariate variant also carries an unpenalized domain term, not counted. For the additive model a smooth counts as non-zero when its effective degrees of freedom reach one tenth. The patient-macro row averages each patient's AURC.",
    "Gate minus random and gate minus ensemble are paired AURC differences on shared resamples; negative favors the gate.")
  list(title = "Panel A. Exploratory sensitivity variants of the gate (post hoc diagnostic columns: non-zero terms, constant score)", df = df,
       widths = c(1.9, 1.45, 2.1, 0.7, 1.3, 1.3, 1.3), align = c("l", "l", "l", "c", "c", "c", "c"),
       labels = c(AURC = "AURC\n(95% CI)", `Constant score` = "Constant\nscore", `Gate minus random` = "Gate minus\nrandom AURC\n(95% CI)",
                  `Gate minus ensemble` = "Gate minus\nensemble AURC\n(95% CI)", `Non-zero penalized terms` = "Non-zero penalized terms,\nof those offered"),
       footnote = note)
}

sup_s2_calibration <- function() {
  gi <- sup_read("gate_internal.csv")
  ge <- sup_read("gate_external.csv")
  meths <- c("predicted_area", "optical_gate", "gate_plus_ensemble")
  rows <- lapply(meths, function(m) {
    a <- sup_row(gi, method = m)
    b <- sup_row(ge, method = m)
    data.frame(Method = S2_LABELS[[m]], `Training center` = sup_est_ci(a$cal_intercept, a$cal_intercept_lo, a$cal_intercept_hi, 2),
               `External center` = sup_est_ci(b$cal_intercept, b$cal_intercept_lo, b$cal_intercept_hi, 2), check.names = FALSE, stringsAsFactors = FALSE)
  })
  df <- do.call(rbind, rows)
  list(title = "Panel B. Calibration intercept of the probabilistic methods", df = df, widths = c(2.2, 1.9, 1.9), align = c("l", "c", "c"),
       labels = c(`Training center` = "Training center, cross-fitted\nintercept (95% CI)", `External center` = "External center, locked gate\nintercept (95% CI)"),
       footnote = "Calibration intercept (calibration in the large): the shift on the log-odds scale that makes the mean predicted failure probability equal the observed failure rate with the slope fixed at one; zero is ideal, a negative value means failures are over-predicted.")
}

sup_s2_calib_offset <- function() {
  cb <- sup_read("diag_calibration.csv")
  lab <- c(center1_oof = "Training center, out-of-fold links", center1_final_in_sample = "Training center, final gate (in-sample)",
           center2_final = "External center, final gate")
  rows <- lapply(names(lab), function(s) {
    r <- sup_row(cb, scope = s)
    data.frame(Scope = lab[[s]], Frames = fmt_num(r$n_frames, 0), `Observed failure rate` = sup_pct_ci(r$observed_rate, r$observed_lo, r$observed_hi),
               `Mean predicted` = sup_pct_ci(r$mean_predicted, r$mean_predicted_lo, r$mean_predicted_hi),
               `Offset` = sup_pct_ci(r$offset, r$offset_lo, r$offset_hi), `SD of predicted` = sup_pct(r$sd_predicted),
               check.names = FALSE, stringsAsFactors = FALSE)
  })
  df <- do.call(rbind, rows)
  list(title = "Panel C. Mean predicted probability and observed failure rate of the optical gate (post hoc diagnostic)", df = df, break_before = TRUE,
       widths = c(2.6, 0.8, 1.6, 1.6, 1.6, 0.9), align = c("l", "c", "c", "c", "c", "c"),
       labels = c(`Observed failure rate` = "Observed failure\nrate, % (95% CI)", `Mean predicted` = "Mean predicted\nprobability, % (95% CI)",
                  Offset = "Predicted minus\nobserved, % (95% CI)", `SD of predicted` = "SD of predicted\nprobability, %"),
       footnote = "SD, standard deviation. Intervals are patient-clustered bootstrap percentile intervals (patients resampled within domain). The in-sample row applies the final gate to the frames it was fitted on and is shown only to separate fit from transport.")
}

sup_s2_secondary <- function() {
  gi <- sup_read("gate_internal.csv")
  ge <- sup_read("gate_external.csv")
  cov80 <- round(100 * unlist(CFG$endpoints$coverage_levels)[2])
  sens <- round(100 * unlist(CFG$endpoints$sens_levels))
  stats <- list(spec50 = list("spec_at_sens50", 2), spec90 = list("spec_at_sens90", 2), sel80 = list("sel_risk_80", 3), macro = list("aurc_macro", 3))
  mk <- function(tab, m) unlist(lapply(stats, function(s) {
    r <- sup_row(tab, method = m)
    sup_est_ci(r[[s[[1]]]], r[[paste0(s[[1]], "_lo")]], r[[paste0(s[[1]], "_hi")]], s[[2]])
  }))
  cols <- c("Method", paste(rep(c("Internal", "External"), each = 4), c("spec50", "spec90", "sel80", "macro")))
  rows <- lapply(S2_METHODS, function(m) setNames(as.data.frame(as.list(c(S2_LABELS[[m]], mk(gi, m), mk(ge, m))), stringsAsFactors = FALSE), cols))
  df <- do.call(rbind, rows)
  head4 <- c(sprintf("Specificity at %d%%\nsensitivity", sens[1]), sprintf("Specificity at %d%%\nsensitivity", sens[2]),
             sprintf("Selective risk at\n%d%% coverage", cov80), "Patient-averaged\nAURC")
  labels <- c(Method = "Method", setNames(rep(head4, 2), cols[-1]))
  desc <- c(gi$binary_descriptive, ge$binary_descriptive) %in% TRUE
  note <- c("Specificity at a given sensitivity: the share of non-failures kept when the cutoff is set to catch that share of failures. Selective risk: the mean risk (1 - IoU) of the lowest-scored frames kept at that coverage. Patient-averaged AURC: AURC computed inside each patient, then averaged with equal weight per patient. Random order: specificity is one minus the sensitivity, the other columns equal the mean risk.",
            "Intervals are patient-clustered bootstrap, 95% percentile intervals, as in Table 3. Internal is the training center, cross-fitted (out-of-fold); external is the external center scored by the locked gate.")
  if (any(desc)) note <- c(note, "Descriptive only: too few failures or patients with a failure in at least one partition for inferential weight on the binary measures.")
  list(title = "Panel D. Operating points, selective risk and patient-averaged AURC", df = df,
       widths = c(1.55, rep(1.06, 8)), align = c("l", rep("c", 8)), labels = labels, footnote = note,
       spanners = list(values = c("", "Internal: training center, cross-fitted", "External: external center, locked gate"), colwidths = c(1L, 4L, 4L)))
}

sup_s2_domain <- function() {
  wd <- sup_read("diag_within_domain.csv")
  doms <- unique(wd$domain)
  rows <- list()
  bold <- integer()
  indent <- integer()
  rule <- integer()
  cols <- c("Method", "AURC", "AUROC", "AURC minus random")
  for (d in doms) {
    t <- wd[wd$domain == d, ]
    head <- sprintf("%s (%s frames, %s patients, %s failures in %s patients)", sup_domain_label(d), fmt_num(t$n_frames[1], 0), fmt_num(t$n_patients[1], 0),
                    fmt_num(t$n_events[1], 0), fmt_num(t$n_patients_with_events[1], 0))
    if (length(rows)) rule <- c(rule, length(rows))
    rows[[length(rows) + 1L]] <- setNames(as.data.frame(as.list(c(head, "", "", "")), stringsAsFactors = FALSE), cols)
    bold <- c(bold, length(rows))
    for (m in c(S2_METHODS, "oracle")) {
      r <- sup_row(t, method = m)
      rows[[length(rows) + 1L]] <- setNames(data.frame(S2_LABELS[[m]], sup_est_ci(r$aurc, r$aurc_lo, r$aurc_hi, 3),
                                                       if (m %in% c("random", "oracle")) sup_est_ci(r$auroc, NA, NA, 2) else sup_est_ci(r$auroc, r$auroc_lo, r$auroc_hi, 2),
                                                       if (m == "random") "" else sup_est_ci(r$diff_aurc_vs_random, r$diff_aurc_vs_random_lo, r$diff_aurc_vs_random_hi, 3),
                                                       stringsAsFactors = FALSE), cols)
      indent <- c(indent, length(rows))
    }
  }
  df <- do.call(rbind, rows)
  list(title = "Panel E. Within-domain internal performance, training center out-of-fold (post hoc diagnostic)", df = df, break_before = TRUE,
       widths = c(3.7, 1.6, 1.6, 1.8), align = c("l", "c", "c", "c"), bold_rows = bold, indent_rows = indent, rule_after = rule, merge_rows = bold,
       labels = c(AURC = "AURC\n(95% CI)", AUROC = "AUROC for failure\n(95% CI)", `AURC minus random` = "AURC minus random\n(95% CI)"),
       footnote = "Each domain is resampled on its own (patients with replacement), so the intervals are wider than in Table 3. The paired difference to random order uses the same resamples; negative favors the method. Random order has an AUROC of one half by definition and the oracle orders by the true risk.")
}

sup_s2_flags <- function() {
  ss <- sup_read("stream_summary.csv")
  fl <- sup_read("diag_flag_shares.csv")
  ov <- function(stat) sup_row(ss[ss$scope == "overall", ], statistic = stat)
  fs <- function(scope, cov) sup_row(fl, scope = scope, statistic = "flagged_share", coverage = cov)
  codes <- as.character(round(100 * unlist(CFG$endpoints$coverage_levels)))
  cell_ss <- function(prefix, code) { r <- ov(paste0(prefix, "_", code)); sup_pct_ci(r$est, r$lo, r$hi) }
  cell_fl <- function(scope, code) { r <- fs(scope, as.numeric(code)); sup_pct_ci(r$est, r$lo, r$hi) }
  df <- data.frame(
    Scope = c("Training center, out-of-fold scores (nominal, by construction)", "Training center, final gate (in-sample; post hoc)",
              "External center, final gate", "Deployment stream, final gate"),
    `Cutoff 1` = c(cell_ss("flagged_share_center1_oof", codes[1]), cell_fl("center1_final", codes[1]), cell_ss("flagged_share_center2", codes[1]),
                   cell_ss("flagged_share_stream", codes[1])),
    `Cutoff 2` = c(cell_ss("flagged_share_center1_oof", codes[2]), cell_fl("center1_final", codes[2]), cell_ss("flagged_share_center2", codes[2]),
                   cell_ss("flagged_share_stream", codes[2])),
    check.names = FALSE, stringsAsFactors = FALSE)
  names(df)[2:3] <- sprintf("Flagged at the %s%% coverage cutoff", codes)
  thr <- vapply(codes, function(cd) ov(paste0("flag_threshold_", cd))$est, 0)
  band <- sup_row(fl, scope = "locked_cutoffs", statistic = "band_width_logodds")$est
  sd_f <- sup_row(fl, scope = "center1_final", statistic = "score_sd")$est
  sd_o <- sup_row(fl, scope = "center1_oof", statistic = "score_sd")$est
  b_f <- sup_row(fl, scope = "center1_final", statistic = "band_share")$est
  b_2 <- sup_row(fl, scope = "center2_final", statistic = "band_share")$est
  labs <- setNames(sprintf("Flagged, %% (95%% CI)\nat the %s%% coverage cutoff", codes), names(df)[2:3])
  note <- c(
    sprintf("Cutoffs are the locked training-center out-of-fold quantiles of the gate log-odds (%s and %s, a band of %s log-odds). Out-of-fold scores come from the five fold gates, so their shares equal the nominal %s%% and %s%% by construction; every other row applies the final locked gate. The final gate's scores spread less than the pooled fold scores (SD %s vs %s), so the same cutoffs do not give the nominal share on the frames the final gate was fitted on, and the band between the cutoffs holds %s%% of training-center frames and %s%% of external-center frames.",
            fmt_num(thr[1], 2), fmt_num(thr[2], 2), fmt_num(band, 2), fmt_num(100 - as.numeric(codes[1]), 0), fmt_num(100 - as.numeric(codes[2]), 0),
            fmt_num(sd_f, 2), fmt_num(sd_o, 2), fmt_num(100 * b_f, 1), fmt_num(100 * b_2, 1)),
    "The deployment stream has no ground truth. Intervals are patient-clustered bootstrap percentile intervals (clustered on video for the stream). The second row is the post hoc diagnostic; the other rows repeat the locked stream analysis.")
  list(title = "Panel F. Share of frames flagged at both locked cutoffs", df = df, break_before = TRUE, widths = c(3.8, 2.1, 2.1), align = c("l", "c", "c"), labels = labs, footnote = note)
}

sup_s2_stream <- function() {
  ss <- sup_read("stream_summary.csv")
  ov <- function(stat) sup_row(ss[ss$scope == "overall", ], statistic = stat)
  e <- ov("spearman_gate_ens_dis")
  h <- ov("spearman_gate_entropy_fov")
  df <- data.frame(Comparison = c("Gate score with ensemble disagreement", "Gate score with pixel entropy"),
                   `Spearman rho` = c(sup_est_ci(e$est, e$lo, e$hi, 2), sup_est_ci(h$est, h$lo, h$hi, 2)),
                   Frames = fmt_num(c(e$n, h$n), 0), check.names = FALSE, stringsAsFactors = FALSE)
  list(title = "Panel G. Rank correlation of the gate score with the two baselines on the deployment stream", df = df, widths = c(3.8, 2.1, 1.2),
       align = c("l", "c", "c"), labels = c(`Spearman rho` = "Spearman rho\n(95% CI)"),
       footnote = sprintf("Spearman rho, rank correlation between the gate log-odds and the baseline score across stream frames (%s videos); the interval is a bootstrap percentile interval clustered on video. Descriptive: the stream has no ground truth.", fmt_num(ov("n_videos")$est, 0)))
}

sup_table_s2 <- function() {
  general <- c(
    SUP_STATUS,
    "AURC, area under the risk-coverage curve: the mean risk (1 - IoU) of the frames a method keeps, averaged over all coverages (lower is better); IoU, intersection over union of the predicted and expert instrument masks; AUROC, area under the receiver operating characteristic curve for detecting a failure (IoU below the frozen failure threshold); CI, confidence interval; SD, standard deviation.",
    sprintf("Intervals are patient-clustered bootstrap percentile intervals (%s resamples, patients resampled within domain). The training center is Center 1 (domains D1 and D2); the external center is Center 2 (domain D3).",
            fmt_num(CFG$statistics$bootstrap_resamples, 0)))
  list(panels = list(sup_s2_variants(), sup_s2_calibration(), sup_s2_calib_offset(), sup_s2_secondary(), sup_s2_domain(), sup_s2_flags(),
                     sup_s2_stream()), general = general)
}

# ---- Table S3: covariate shift and odds-ratio robustness ---------------------------------------------------------------

sup_s3_shift <- function() {
  cs <- sup_read("diag_covariate_shift.csv")
  pm <- cs[cs$scope == "metric", ]
  pm <- pm[match(OPTICAL_METRICS, pm$metric), ]
  df <- data.frame(Metric = unname(METRIC_LABELS[pm$metric]), `Training median` = sup_metric_val(pm$median_c1),
                   `External median` = sup_metric_val(pm$median_c2), `Standardized difference` = fmt_num(pm$std_median_diff, 2),
                   `Below 5th` = sup_pct(pm$share_c2_below_p05), `Above 95th` = sup_pct(pm$share_c2_above_p95),
                   `Outside range` = sup_pct(pm$share_c2_below_min + pm$share_c2_above_max, 2),
                   `Separation` = fmt_num(pm$auroc_c2_vs_c1, 2), check.names = FALSE, stringsAsFactors = FALSE)
  u <- pm[pm$metric == "under_frac", ]
  note <- c(
    "Training center = Center 1 labeled analyzed frames, external center = Center 2 labeled analyzed frames; the metrics are those of Table 2.",
    sprintf("Standardized difference: external median minus training median, divided by the training SD. Share of external frames below the training 5th percentile and above the 95th percentile (about %s%% each would be expected without shift). Outside range: below the training minimum or above the training maximum (the extrapolation region of the gate's splines). Separation: AUROC for telling an external frame from a training frame by the metric alone (one half means no separation; below one half means the metric is lower in the external center).",
            fmt_num(100 * 0.05, 0)),
    sprintf("The under-exposed fraction has the largest share outside the training range (%s%% of external frames lie below the training minimum), but by only %s training SD on average among those frames; no other metric has more than %s%% of external frames outside the range, so the shifts in the other metrics are shifts inside it (see the standardized difference and the shares beyond the percentiles).",
            fmt_num(100 * u$share_c2_below_min, 1), fmt_num(u$mean_excursion_below_min_sd, 3),
            fmt_num(100 * max((pm$share_c2_below_min + pm$share_c2_above_max)[pm$metric != "under_frac"]), 2)))
  list(title = "Panel A. Shift of each optical metric between centers (post hoc diagnostic)", df = df,
       widths = c(1.95, 0.95, 0.95, 1.15, 1.0, 1.0, 1.0, 1.0), align = c("l", rep("c", 7)),
       labels = c(`Training median` = "Training center\nmedian", `External median` = "External center\nmedian",
                  `Standardized difference` = "Standardized\nmedian difference", `Below 5th` = "External frames below\ntraining 5th percentile, %",
                  `Above 95th` = "External frames above\ntraining 95th percentile, %", `Outside range` = "External frames outside\ntraining range, %",
                  Separation = "Separation\nAUROC"),
       footnote = note)
}

sup_s3_multi <- function() {
  cs <- sup_read("diag_covariate_shift.csv")
  mv <- function(name) sup_row(cs, scope = "multivariate", metric = name)
  a <- mv("domain_auroc"); b <- mv("domain_auroc_excl_under_frac"); c3 <- mv("domain_auroc_excl_top3")
  excl3 <- strsplit(sub("^without ", "", c3$note), ", ", fixed = TRUE)[[1]]
  df <- data.frame(
    Statistic = c("Center classifier AUROC, all 11 metrics", "Center classifier AUROC, without the under-exposed fraction",
                  sprintf("Center classifier AUROC, without the three most separating metrics (%s)", paste(tolower(METRIC_LABELS[excl3]), collapse = ", ")),
                  "Median Mahalanobis distance, training-center frames (out-of-fold)", "Median Mahalanobis distance, external-center frames",
                  "Ratio of the two medians", "External frames beyond the 95th percentile of the training out-of-fold distance, %"),
    `Estimate` = c(sup_est_ci(a$value, a$lo, a$hi, 3), sup_est_ci(b$value, b$lo, b$hi, 3), sup_est_ci(c3$value, c3$lo, c3$hi, 3),
                   fmt_num(mv("mahalanobis_median_c1_oof")$value, 2), fmt_num(mv("mahalanobis_median_c2")$value, 2),
                   fmt_num(mv("mahalanobis_ratio_c2_over_c1")$value, 2), sup_pct(mv("mahalanobis_share_c2_above_c1_p95")$value)),
    check.names = FALSE, stringsAsFactors = FALSE)
  list(title = "Panel B. Multivariate shift in the space of the 11 metrics (post hoc diagnostic)", df = df, widths = c(6.6, 1.9), align = c("l", "c"),
       labels = c(Estimate = "Estimate\n(95% CI)"),
       footnote = c("Center classifier: unpenalized logistic regression of external-center versus training-center frames on the 11 metrics standardized with the training mean and SD (sharpness on the log scale), scored out-of-fold with patient-grouped folds balanced inside each center; the AUROC interval is a patient-clustered bootstrap percentile interval of the out-of-fold scores. One half would mean the centers cannot be told apart.",
                    "Mahalanobis distance: distance of a frame from the training-center centroid in units that account for the correlation between metrics. Training-center frames are scored against a centroid and covariance estimated without their fold; external frames against those of all training-center frames."))
}

sup_s3_or <- function() {
  ors <- sup_read("diag_or_robustness.csv")
  ext <- sup_read("assoc_or_models.csv")
  ext <- ext[ext$scope == "center2_external" & ext$model == "univariable_adjusted", ]
  ors <- ors[match(OPTICAL_METRICS, ors$metric), ]
  ext <- ext[match(OPTICAL_METRICS, ext$metric), ]
  logs <- ors$or_scale != "per SD"
  df <- data.frame(Metric = paste0(unname(METRIC_LABELS[ors$metric]), ifelse(logs, "§", "")),
                   `Primary Wald` = sup_est_ci(ors$or_primary, ors$or_lo_wald, ors$or_hi_wald, 2),
                   `Primary bootstrap` = sprintf("%s to %s", fmt_num(ors$or_lo_boot, 2), fmt_num(ors$or_hi_boot, 2)),
                   `Ground-truth area` = sup_est_ci(ors$or_gtarea, ors$or_lo_gtarea, ors$or_hi_gtarea, 2),
                   `Excludes one Wald` = sup_yes(ors$excl1_wald), `Excludes one bootstrap` = sup_yes(ors$excl1_boot),
                   `Excludes one ground-truth` = sup_yes(ors$excl1_gtarea),
                   `External` = sup_wide_ci(ext$or_sd, ext$or_lo, ext$or_hi), check.names = FALSE, stringsAsFactors = FALSE)
  n_w <- sum(ors$excl1_wald); n_b <- sum(ors$excl1_boot); n_g <- sum(ors$excl1_gtarea)
  agree <- sum(ors$excl1_wald == ors$excl1_boot)
  note <- c(
    "OR, odds ratio of failure (IoU below the frozen failure threshold) per SD of the metric, from one logistic model per metric adjusted for domain, channel, region, image format, instrument class and the instrument area; CI, confidence interval; SD, standard deviation. Training-center out-of-fold frames, primary specification: predicted instrument area as published in Table 2.",
    sprintf("Primary, Wald: cluster-robust (patient) 95%% interval as in Table 2. Primary, bootstrap: patient-clustered, domain-stratified percentile 95%% interval of the same estimate (%s resamples, one set for all metrics). Ground-truth area: the same models with the expert-mask instrument area instead of the predicted area, because the predicted area is itself a consequence of the segmentation failure being modeled. The interval excludes one for %s of %d metrics (Wald), %s (bootstrap, agreeing on %s of %d) and %s (ground-truth area).",
            fmt_num(CFG$statistics$bootstrap_resamples, 0), fmt_num(n_w, 0), nrow(ors), fmt_num(n_b, 0), fmt_num(agree, 0), nrow(ors), fmt_num(n_g, 0)),
    "External center: the same primary specification fitted on external-center frames (cluster-robust Wald interval), descriptive only; the under-exposed fraction is nearly constant there, which is why its interval is so wide.",
    "§ Odds ratio per SD of the logarithm of the metric.")
  list(title = "Panel C. Robustness of the adjusted odds ratios for failure (post hoc diagnostic; external center descriptive)", df = df, break_before = TRUE,
       widths = c(1.75, 1.3, 1.15, 1.3, 0.8, 0.85, 0.85, 1.8), align = c("l", "c", "c", "c", "c", "c", "c", "c"),
       labels = c(`Primary Wald` = "Primary OR\n(Wald 95% CI)", `Primary bootstrap` = "Primary,\nbootstrap 95% CI",
                  `Ground-truth area` = "OR with ground-truth\narea (Wald 95% CI)", `Excludes one Wald` = "Excludes\none: Wald",
                  `Excludes one bootstrap` = "Excludes one:\nbootstrap", `Excludes one ground-truth` = "Excludes one:\nground-truth",
                  External = "External center OR\n(Wald 95% CI), descriptive"),
       footnote = note)
}

sup_table_s3 <- function() {
  list(panels = list(sup_s3_shift(), sup_s3_multi(), sup_s3_or()), general = SUP_STATUS)
}

# ---- build ---------------------------------------------------------------------------------------------------------------

SUP_INPUTS <- list(
  TableS1 = "assoc_by_domain.csv",
  TableS2 = c("sensitivity.csv", "diag_gate_nonzero.csv", "gate_internal.csv", "gate_external.csv", "diag_calibration.csv",
              "diag_within_domain.csv", "stream_summary.csv", "diag_flag_shares.csv"),
  TableS3 = c("diag_covariate_shift.csv", "diag_or_robustness.csv", "assoc_or_models.csv"))

build_supplement <- function() {
  sup_lock()   # before any table that carries external-center results is read
  builders <- list(TableS1 = sup_table_s1, TableS2 = sup_table_s2, TableS3 = sup_table_s3)
  out <- list()
  for (nm in names(builders)) {
    need <- file.path(p("results", "stats"), SUP_INPUTS[[nm]])
    gone <- need[!file.exists(need)]
    if (length(gone)) {
      message(nm, ": skipped, inputs not available yet: ", paste(basename(gone), collapse = ", "))
      next
    }
    tb <- builders[[nm]]()
    out[[nm]] <- sup_save(nm, tb$panels, tb$general)
    cat(nm, "->", paste(basename(out[[nm]]), collapse = ", "), "\n")
  }
  invisible(out)
}

if (sys.nframe() == 0L) {
  if (SYNTH && !.gl_nonempty(tryCatch(yaml::read_yaml(p("config", "analysis_lock.yaml"))$lock_fields_sha256, error = function(e) NULL)) &&
      !file.exists(file.path(p("results", "stats"), "diag_covariate_shift.csv"))) {
    message("21 supplement skipped: no sealed synthetic lock and no diagnostics output (run 04a, 08 and 34 on synthetic data first)")
    quit(status = 0)
  }
  build_supplement()
}
