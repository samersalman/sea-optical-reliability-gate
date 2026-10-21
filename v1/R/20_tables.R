# Tables. Table 1 panel A: cohort and acquisition domains, from compute_cohort() so the cells and the registered
# 02_cohort numbers cannot disagree. Panel B (segmenter performance by partition x model) is the only exhibit here that
# reads frame-level data (predictions.parquet, which holds Center 2 outcomes), so on real data it is built only once the
# analysis lock is sealed and verified; Tables 2 and 3 only format the CSVs under results/stats/ (03, 04b, 05, 06), so no
# statistic is recomputed. This script builds every table whose inputs exist and skips the rest with a message.
#   Rscript R/20_tables.R                      (OPTICS_DATA=synthetic for development)
# Writes tables/Table1.{docx,csv} (panel A + B), Table1_panelA|B.*, Table2.*, Table3.* and, when panel B was built,
# results/parts/20_tables.json (keys seg_<partition>_<model>_<stat>).

if (!exists("compute_cohort", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (exists("V1", envir = globalenv())) file.path(get("V1", envir = globalenv()), "R")
             else if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "02_cohort.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

# Returns the panel as a plain character data frame plus the row indices flex_three_line() needs, so a combined
# Table 1 can rbind panels (offset the indices by the rows above):
#   list(df, bold_rows (panel heading), indent_rows (sub-rows), footnote)
table1_panel_a <- function(coh = compute_cohort()) {
  doms <- names(DOMAIN_COLOURS)
  cols <- c(doms, "Total")
  g <- setNames(split(coh$groups, coh$groups$group), coh$groups$group)
  st <- coh$strata
  n0 <- function(x) fmt_num(x, 0)
  frac <- function(x) fmt_num(x, 1)
  # frame counts and pixel diameters are shown to whole units, rounded half up, as registered by 02_cohort.R
  nh <- function(x) n0(round_half_up(x))
  med_iqr <- function(r, pre) sprintf("%s [%s to %s]", nh(r[[paste0(pre, "median")]]), nh(r[[paste0(pre, "q1")]]), nh(r[[paste0(pre, "q3")]]))
  row <- function(label, vals) as.data.frame(c(list(Characteristic = label), setNames(as.list(vals), cols)), stringsAsFactors = FALSE)
  cell <- function(variable, level, patients) vapply(cols, function(gr) {
    r <- st[st$variable == variable & st$level == level & st$group == gr, ]
    pct <- sprintf("%s (%s)", n0(r$n_frames), frac(100 * r$share))
    if (patients) sprintf("%s; %s", n0(r$n_patients), pct) else pct
  }, character(1))

  roles <- vapply(doms, domain_role, "")
  rows <- list(
    row("Panel A. Cohort and acquisition domains", rep("", 4)),
    row("Center", c(roles, if (length(unique(roles)) > 1L) "Both centers" else unname(roles[1]))),
    row("Resolution, pixels", c(vapply(doms, domain_size_label, ""), "")),
    row("Patients, n", vapply(cols, function(gr) n0(g[[gr]]$n_patients), "")),
    row("Frames, n", vapply(cols, function(gr) n0(g[[gr]]$n_frames), "")),
    row("Frames per patient, median [interquartile range]", vapply(cols, function(gr) med_iqr(g[[gr]], "fpp_"), "")),
    row("Frames per patient, range", vapply(cols, function(gr) sprintf("%s to %s", n0(g[[gr]]$fpp_min), n0(g[[gr]]$fpp_max)), "")),
    row("Field-of-view diameter, pixels, median [interquartile range]", vapply(cols, function(gr) med_iqr(g[[gr]], "diam_"), ""))
  )
  idx <- length(rows)
  indent <- integer()
  block <- function(title, variable, levels, labels, patients = FALSE) {
    rows <<- c(rows, list(row(title, rep("", 4))))
    for (i in seq_along(levels)) {
      rows <<- c(rows, list(row(labels[i], cell(variable, levels[i], patients))))
      indent <<- c(indent, length(rows))
    }
  }
  by_total <- function(variable) {
    t <- st[st$variable == variable & st$group == "Total", ]
    t$level[order(-t$n_frames, t$level)]
  }
  cap <- function(x) paste0(toupper(substring(x, 1, 1)), substring(x, 2))
  block("Channel, patients; frames (%)", "channel", c("small", "big"), c("Small", "Big"), patients = TRUE)
  block("Region, patients; frames (%)", "region", c("lumbar", "cervical"), c("Lumbar", "Cervical"), patients = TRUE)
  block("Scenario, frames (%)", "scenario", c("Normal", "Difficult"), c("Normal", "Difficult"))
  inst <- by_total("instrument")
  block("Instrument, frames (%)", "instrument", inst, cap(inst))
  block("File format, frames (%)", "file_format", c("png", "jpg"), c("PNG", "JPG"))

  df <- do.call(rbind, rows)
  names(df) <- c("Characteristic", doms, "All annotated")
  nf <- coh$counts$n_excl_fov_c1 + coh$counts$n_excl_fov_c2
  footnote <- c(
    "D1 to D3 are the acquisition domains: D1 and D2 are the two image sizes at the training center and D3 is the external center. Annotated frames are those with an expert instrument mask. Medians are followed by the interquartile range (first to third quartile) in brackets. Percentages are of the frames in each column.",
    sprintf("Field-of-view diameter is twice the fitted radius in native pixels over annotated frames with a detected field of view (%s of %s annotated frames lack one and are excluded here); for D3 the radius is that of the inscribed circle.",
            n0(nf), n0(coh$counts$n_frames_labeled)),
    "Channel is the endoscope working-channel size class (small or big) as filed in the dataset. Channel and region (cervical or lumbar) are fixed within a patient; each patient contributes to one cell.")
  list(df = df, bold_rows = 1L, indent_rows = indent, footnote = footnote)
}

# fixed widths (inches, 6.85 in usable at 0.8 in margins) so the long row labels wrap instead of widening the table
TBL1A_WIDTHS <- c(2.45, rep(1.1, 4))
tbl_flex_panel_a <- function(a) tbl_fix_widths(flex_three_line(a$df, bold_rows = a$bold_rows, indent_rows = a$indent_rows,
                                                               footnote = a$footnote), TBL1A_WIDTHS)

save_table1_panel_a <- function(coh = compute_cohort()) {
  a <- table1_panel_a(coh)
  c(docx = tbl_write_docx(list(tbl_flex_panel_a(a)), file.path(p("tables"), "Table1_panelA.docx")),
    csv = tbl_write_csv(a$df, file.path(p("tables"), "Table1_panelA.csv")))
}

# ================================================================================================================
# Table 1 panel B, Table 2, Table 3
# ================================================================================================================

if (!exists("read_lock", mode = "function")) source(file.path(V1, "R", "stats_lib.R"), chdir = FALSE)

TBL_MODEL_LABELS <- c(unet = "U-Net", segformer = "SegFormer")
# the "95% CI" wording of Tables 2 and 3 is typed, so refuse a plan that changes the level
stopifnot(identical(CFG$statistics$ci_level, 0.95))
# Table 2 footnotes any optical-metric VIF above statistics.vif_flag (plan); read where it is used
# The network has one logit, so the baseline is the mean binary entropy of the predicted foreground probability:
# "Pixel entropy" (METHOD_LABELS, 00_setup.R).
TBL_METHOD_LABELS <- METHOD_LABELS
TBL_FAMILY_LABELS <- PERTURB_FAMILY_LABELS      # the names of Figure 2B and 2C
# plain-language quantity and computation per metric; no numbers, so thresholds and block sizes live only in the plan
TBL_METRIC_DEF <- c(
  lum_mean = "Overall brightness (exposure): mean luminance across the field of view",
  contrast_rms = "Global contrast: standard deviation of luminance across the field of view",
  contrast_local = "Local contrast and fine detail: mean standard deviation of luminance within small image blocks",
  sharp_tenengrad = "Focus sharpness: mean squared Sobel gradient of luminance (Tenengrad); sensor noise also raises it",
  under_frac = "Under-exposure: share of pixels with very low luminance",
  sat_frac = "Over-exposure: share of pixels with any color channel at or near full scale",
  spec_frac = "Glare: share of pixels that are bright and color-poor relative to the frame's own brightness and color",
  haze_dcp = "Haze, smoke or turbidity: mean dark-channel value (darkest color channel in small patches)",
  red_dom = "Blood absorption proxy: mean red fraction of total color, red / (red + green + blue)",
  illum_slope = "Uneven illumination: slope of luminance against distance from the field-of-view center (negative = darker toward the rim)",
  lum_entropy = "Tonal range: Shannon entropy of the luminance histogram"
)
stopifnot(setequal(names(TBL_METRIC_DEF), OPTICAL_METRICS))
# Table 2 marks. The dagger is the known limitation (noise x sharpness), as in Figure 2C; the section mark is the
# "largest |rho| family" fallback.
MK_LIMIT <- "†"; MK_LOG <- "‡"; MK_NO_EXPECTED <- "§"; MK_RIDGE <- "¶"

# The failure threshold lives in the lock once Gate G2 has run (else the plan). A lock that carries a hash must verify
# before its threshold is read (CONTRACTS.md); a synthetic lock written before lock_fields_sha256 existed is read as is.
tbl_read_lock <- function() {
  path <- p("config", "analysis_lock.yaml")
  if (!file.exists(path)) return(list())
  yaml::read_yaml(path)
}

tbl_lock <- function() {
  lock <- tbl_read_lock()
  sealed <- any(vapply(c("gate_rds_sha256", "lock_fields_sha256"), function(k) .gl_nonempty(lock[[k]]), NA))
  if (sealed && (!SYNTH || .gl_nonempty(lock$lock_fields_sha256))) verify_lock_hashes(lock)
  lock
}

# Panel B reads Center 2 outcomes (predictions.parquet), so on real data it waits for a sealed lock: lock_fields_sha256
# present and verified (verify_lock_hashes stops on a mismatch). Returns NULL when panel B may be built, else the reason.
# Synthetic mode keeps working with a lock that predates lock_fields_sha256.
tbl_panel_b_blocked <- function(lock) {
  if (!.gl_nonempty(lock$lock_fields_sha256)) {
    if (SYNTH) return(NULL)
    return("the analysis lock has no lock_fields_sha256, so Center 2 outcomes stay closed; run 04a_gate_fit_lock.R first")
  }
  verify_lock_hashes(lock)
  NULL
}

tbl_read_stats <- function(name, dir = p("results", "stats")) {
  path <- file.path(dir, name)
  if (!file.exists(path)) stop("missing input ", path, " (run the upstream script first)", call. = FALSE)
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = c("NA", ""))
}

tbl_est_ci <- function(est, lo, hi, digits) {
  if (is.na(est)) return("")
  if (is.na(lo) || is.na(hi)) return(fmt_num(est, digits))
  sprintf("%s (%s to %s)", fmt_num(est, digits), fmt_num(lo, digits), fmt_num(hi, digits))
}

lc_first <- function(x) paste0(tolower(substring(x, 1, 1)), substring(x, 2))

# The csv keeps each cell on one line; the docx moves the interval (or the second item) to its own line so the
# columns stay narrow. Same text otherwise.
tbl_wrap_docx <- function(df, cols, pattern, replacement) {
  for (cl in cols) df[[cl]] <- sub(pattern, replacement, df[[cl]])
  df
}

# One docx holding one or more flextables, A4, narrow margins, no title paragraphs (captions come from the manuscript).
tbl_write_docx <- function(fts, path, landscape = FALSE, new_page = integer()) {
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

tbl_write_csv <- function(df, path) {
  guard_write(path)
  utils::write.csv(df, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
  invisible(path)
}

# fixed widths in inches (sum to the usable page width), top-aligned cells; the three-line borders are untouched
tbl_fix_widths <- function(ft, widths) {
  ft <- flextable::set_table_properties(ft, layout = "fixed")
  ft <- flextable::width(ft, j = seq_along(widths), width = widths)
  ft <- flextable::height_all(ft, height = 0.05, part = "all")   # autofit's row heights leave a blank line under two-line cells
  flextable::valign(ft, valign = "top", part = "body")
}

# ---- Table 1 panel B: segmenter performance ----------------------------------------------------------------

# Center 1 rows must be out-of-fold, Center 2 rows from the final model, and every analysed frame needs a prediction from
# both models: a missing row is an error, not a silent drop.
tbl_check_seg_frames <- function(ft) {
  need <- c(outer(names(TBL_MODEL_LABELS), c("source", "iou", "dice"), paste, sep = "_"))
  if (!all(need %in% names(ft))) stop("predictions.parquet lacks rows for both models", call. = FALSE)
  if (!all(stats::complete.cases(ft[need]))) {
    stop("analysed labeled frames without a prediction row: ", sum(!stats::complete.cases(ft[need])), call. = FALSE)
  }
  for (m in names(TBL_MODEL_LABELS)) {
    src <- ft[[paste0(m, "_source")]]
    if (!all(src[ft$center == 1L] == "oof") || !all(src[ft$center == 2L] == "final")) {
      stop(m, ": Center 1 predictions must be out-of-fold and Center 2 predictions from the final model", call. = FALSE)
    }
  }
  ft
}

# Labeled frames with a found FOV (load_frame_table drops the rest) and both models' predictions.
tbl_seg_frames <- function() tbl_check_seg_frames(load_frame_table(with_predictions = TRUE))

TBL_SEG_STATS <- c("n_frames", "n_patients", "iou_median", "iou_q1", "iou_q3", "dice_median", "dice_q1", "dice_q3",
                   "fail_n", "fail_share", "fail_patients")

# one row per model x partition; partitions: c1 (all Center 1), c1_<domain> for each Center 1 domain, c2 (Center 2)
tbl_seg_summary <- function(ft, thr) {
  doms <- vapply(CFG$domains, function(d) as.integer(d$center), 1L)
  c1_doms <- names(doms)[doms == 1L]
  parts <- c(list(c1 = ft$center == 1L),
             setNames(lapply(c1_doms, function(d) ft$center == 1L & ft$domain == d), paste0("c1_", tolower(c1_doms))),
             list(c2 = ft$center == 2L))
  labels <- c(c1 = sprintf("%s, all (%s)", CENTER_ROLE[["1"]], paste(c1_doms, collapse = " and ")),
              setNames(sprintf("%s (%s)", c1_doms, vapply(c1_doms, domain_size_label, "")), paste0("c1_", tolower(c1_doms))),
              c2 = sprintf("%s (%s)", CENTER_ROLE[["2"]], paste(sprintf("%s, %s", names(doms)[doms == 2L], vapply(names(doms)[doms == 2L], domain_size_label, "")), collapse = "; ")))
  # the registry notes keep their original wording (labels of the numbers-ledger), so the registered part stays unchanged
  ledger <- c(c1 = sprintf("All Center 1 (%s)", paste(c1_doms, collapse = " and ")), labels[paste0("c1_", tolower(c1_doms))],
              c2 = sprintf("Center 2 (%s)", paste(sprintf("%s, %s", names(doms)[doms == 2L], vapply(names(doms)[doms == 2L], domain_size_label, "")), collapse = "; ")))
  q3 <- function(x) stats::quantile(x, c(0.5, 0.25, 0.75), names = FALSE)
  rows <- list()
  for (m in names(TBL_MODEL_LABELS)) for (pt in names(parts)) {
    d <- ft[parts[[pt]], ]
    if (!nrow(d)) stop("no analysed frames in partition ", pt, call. = FALSE)
    iou <- d[[paste0(m, "_iou")]]
    dice <- d[[paste0(m, "_dice")]]
    qi <- q3(iou); qd <- q3(dice); fail <- iou < thr
    rows[[length(rows) + 1L]] <- data.frame(
      part = pt, model = m, label = labels[[pt]], ledger_label = ledger[[pt]], n_frames = nrow(d), n_patients = length(unique(d$patient_id)),
      iou_median = qi[1], iou_q1 = qi[2], iou_q3 = qi[3], dice_median = qd[1], dice_q1 = qd[2], dice_q3 = qd[3],
      fail_n = sum(fail), fail_share = mean(fail), fail_patients = length(unique(d$patient_id[fail])),
      stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

# Same return shape as table1_panel_a (df, bold_rows, indent_rows, footnote) plus `summary`, the numbers behind the cells.
# The lock check comes first: nothing is read from predictions.parquet before it passes.
table1_panel_b <- function(coh = compute_cohort(), frames = NULL, thr = NULL, lock = tbl_read_lock()) {
  why <- tbl_panel_b_blocked(lock)
  if (!is.null(why)) stop("Table 1 panel B not built: ", why, call. = FALSE)
  if (is.null(thr)) thr <- lock_failure_threshold(lock)
  if (is.null(frames)) frames <- tbl_seg_frames()
  in_lock <- is.numeric(lock$failure_iou_threshold) && length(lock$failure_iou_threshold) == 1L && isTRUE(all.equal(lock$failure_iou_threshold, thr))
  sm <- tbl_seg_summary(frames, thr)
  cn <- coh$counts
  for (pt in c(c1 = "c1", c2 = "c2")) {
    got <- sm$n_frames[sm$part == pt & sm$model == "unet"]
    want <- cn[[paste0("n_frames_", pt, "_analysed")]]
    if (got != want) stop("Table 1B has ", got, " ", pt, " frames but the cohort has ", want, " analysed frames", call. = FALSE)
  }
  cols <- c("Partition", "Frames, n", "Patients, n", "Median IoU [interquartile range]", "Median Dice [interquartile range]",
            "Failures, n (%)", "Patients with at least one failure, n")
  q <- function(r, pre) sprintf("%s [%s to %s]", fmt_num(r[[paste0(pre, "_median")]], 3), fmt_num(r[[paste0(pre, "_q1")]], 3),
                                fmt_num(r[[paste0(pre, "_q3")]], 3))
  blank <- function(label) setNames(as.data.frame(as.list(c(label, rep("", length(cols) - 1L))), stringsAsFactors = FALSE), cols)
  rows <- list(blank("Panel B. Segmenter performance"))
  bold <- 1L
  indent <- integer()
  for (m in names(TBL_MODEL_LABELS)) {
    rows <- c(rows, list(blank(TBL_MODEL_LABELS[[m]])))
    bold <- c(bold, length(rows))
    for (i in which(sm$model == m)) {
      r <- sm[i, ]
      rows <- c(rows, list(setNames(data.frame(r$label, fmt_num(r$n_frames, 0), fmt_num(r$n_patients, 0), q(r, "iou"), q(r, "dice"),
                                               sprintf("%s (%s)", fmt_num(r$fail_n, 0), fmt_num(100 * r$fail_share, 1)),
                                               fmt_num(r$fail_patients, 0), stringsAsFactors = FALSE), cols)))
      indent <- c(indent, length(rows))
    }
  }
  df <- do.call(rbind, rows)
  footnote <- c(
    "IoU, intersection over union of the predicted and expert instrument masks; Dice overlap = twice the shared area over the summed areas of the two masks. Both are computed at native resolution inside the field of view for any instrument (class-agnostic); an empty prediction scores zero. Statistics pool frames across patients; medians are followed by the interquartile range (first to third quartile) in brackets.",
    sprintf("A failure is an IoU below %s (%s). Training-center rows are out-of-fold predictions, each frame scored by the model that did not see its patient; external-center rows come from the final model trained on all of the training center. Patients with at least one failure are counted; no per-patient rate is given.",
            fmt_num(thr, 2), if (in_lock) "the frozen failure threshold" else "the failure threshold in the analysis plan; the analysis lock carries none"),
    sprintf("Frames without a detected field of view are excluded: %s of %s training-center frames and %s of %s external-center frames (%s of %s annotated frames analyzed).",
            fmt_num(cn$n_excl_fov_c1, 0), fmt_num(cn$n_frames_c1, 0), fmt_num(cn$n_excl_fov_c2, 0), fmt_num(cn$n_frames_c2, 0),
            fmt_num(cn$n_frames_labeled_analysed, 0), fmt_num(cn$n_frames_labeled, 0)))
  list(df = df, bold_rows = bold, indent_rows = indent, footnote = footnote, summary = sm, threshold = thr)
}

tbl_flex_panel_b <- function(b) {
  ft <- flex_three_line(tbl_wrap_docx(b$df, names(b$df)[4:5], " \\[", "\n["),
                        align = c("l", "c", "c", "c", "c", "c", "c"), bold_rows = b$bold_rows, indent_rows = b$indent_rows,
                        footnote = b$footnote)
  ft <- flextable::set_header_labels(ft, values = setNames(list("Median IoU\n[interquartile\nrange]", "Median Dice\n[interquartile\nrange]",
                                                                "Patients with\nat least one\nfailure, n"),
                                                           names(b$df)[c(4, 5, 7)]))
  for (r in b$bold_rows) ft <- flextable::merge_at(ft, i = r, j = seq_len(ncol(b$df)))
  tbl_fix_widths(ft, c(1.85, 0.65, 0.7, 0.95, 0.95, 0.75, 0.95))
}

# Registers the numbers the Results need that no earlier script registered. The exclusion counts already exist as
# n_excl_fov_c1|c2 (02_cohort) and the Table 2 and 3 numbers as assoc_*, perturb_*, gate_* and transfer_* keys.
register_seg_keys <- function(sm) {
  desc <- c(n_frames = "analysed frames", n_patients = "patients", iou_median = "median IoU", iou_q1 = "first quartile of IoU",
            iou_q3 = "third quartile of IoU", dice_median = "median Dice", dice_q1 = "first quartile of Dice",
            dice_q3 = "third quartile of Dice", fail_n = "failure events (IoU below the failure threshold)",
            fail_share = "failure events as a share of frames", fail_patients = "patients with at least one failure")
  for (i in seq_len(nrow(sm))) {
    r <- sm[i, ]
    for (s in TBL_SEG_STATS) {
      note <- sprintf("%s, %s: %s", TBL_MODEL_LABELS[[r$model]], r$ledger_label, desc[[s]])
      key <- paste("seg", r$part, r$model, s, sep = "_")
      v <- r[[s]]
      if (s %in% c("n_frames", "n_patients", "fail_n", "fail_patients")) reg(key, as.integer(v), note = note)
      else if (s == "fail_share") reg(key, v, fmt = "percent", digits = 1, note = note)
      else reg(key, v, digits = 3, note = note)
    }
  }
  invisible(sm)
}

save_table1_panel_b <- function(coh = compute_cohort(), b = table1_panel_b(coh)) {
  c(docx = tbl_write_docx(list(tbl_flex_panel_b(b)), file.path(p("tables"), "Table1_panelB.docx")),
    csv = tbl_write_csv(b$df, file.path(p("tables"), "Table1_panelB.csv")))
}

# Table 1 = panel A over panel B in one document (the panels have different columns, so two flextables). The csv stacks
# them padded to the wider panel; panel B's column names are a data row.
save_table1 <- function(coh = compute_cohort(), b = table1_panel_b(coh)) {
  a <- table1_panel_a(coh)
  fa <- tbl_flex_panel_a(a)
  w <- max(ncol(a$df), ncol(b$df))
  pad <- function(m) cbind(m, matrix("", nrow(m), w - ncol(m)))
  stack <- rbind(pad(as.matrix(a$df)), pad(matrix(names(b$df), 1L)), pad(as.matrix(b$df)))
  colnames(stack) <- c(names(a$df), rep("", w - ncol(a$df)))
  c(docx = tbl_write_docx(list(fa, tbl_flex_panel_b(b)), file.path(p("tables"), "Table1.docx"), new_page = 2L),
    csv = tbl_write_csv(stack, file.path(p("tables"), "Table1.csv")))
}

# ---- Table 2: metric dictionary and evidence ----------------------------------------------------------------

# "0.426 (0.382 to 0.463)" -> "0.426 [0.382 to 0.463]" (the registered strings of 03_association, brackets for an
# interquartile range); a number in scientific notation ("4.11e-04", sharpness) is reprinted as "4.1e-4", the format of
# Figure 1B and the Figure 2A axis
tbl_iqr_brackets <- function(x) {
  if (!all(grepl("^[^ ]+ \\(.+\\)$", x))) stop("unexpected median (IQR) text in assoc_table2.csv: ", paste(x, collapse = " | "), call. = FALSE)
  x <- sub("^([^ ]+) \\((.+)\\)$", "\\1 [\\2]", x)
  vapply(x, function(z) {
    m <- gregexpr("[0-9.]+e[+-][0-9]+", z)
    regmatches(z, m) <- lapply(regmatches(z, m), function(v) sci_label(as.numeric(v)))
    z
  }, "", USE.NAMES = FALSE)
}

# expected family (or families) per metric, else the family with the largest |rho| (marked); known-limitation cell marked
tbl_perturb_cells <- function(rho) {
  fam_order <- names(TBL_FAMILY_LABELS)
  one <- function(m) {
    r <- rho[rho$metric == m, ]
    exp_rows <- r[r$expected_pair %in% TRUE, ]
    if (nrow(exp_rows)) {
      r <- exp_rows[order(match(exp_rows$family, fam_order)), ]
      mark <- ifelse(r$known_limitation %in% TRUE, MK_LIMIT, "")
    } else {
      r <- r[which.max(abs(r$rho)), ]
      mark <- MK_NO_EXPECTED
    }
    fam <- paste0(TBL_FAMILY_LABELS[r$family], mark)
    est <- vapply(seq_len(nrow(r)), function(i) tbl_est_ci(r$rho[i], r$rho_lo[i], r$rho_hi[i], 2), "")
    c(family = paste(fam, collapse = "; "), rho = paste(est, collapse = "; "), no_expected = !nrow(exp_rows), limit = any(r$known_limitation %in% TRUE))
  }
  do.call(rbind, lapply(OPTICAL_METRICS, one))
}

table2 <- function(thr = lock_failure_threshold(tbl_lock()), stats_dir = p("results", "stats")) {
  a2 <- tbl_read_stats("assoc_table2.csv", stats_dir)
  rho <- tbl_read_stats("perturb_rho.csv", stats_dir)
  vif <- tbl_read_stats("assoc_vif.csv", stats_dir)
  dg <- tbl_read_stats("assoc_diagnostics.csv", stats_dir)
  bd <- tbl_read_stats("assoc_by_domain.csv", stats_dir)
  vif_flag <- CFG$statistics$vif_flag
  if (!is.numeric(vif_flag) || length(vif_flag) != 1L) stop("analysis_plan.yaml statistics.vif_flag must be one number", call. = FALSE)
  a2 <- a2[match(OPTICAL_METRICS, a2$metric), ]
  if (anyNA(a2$metric)) stop("assoc_table2.csv lacks some of the optical metrics", call. = FALSE)
  pc <- tbl_perturb_cells(rho)
  ridge <- a2$or_method != "cluster-robust logistic"
  logs <- a2$or_scale != "per SD"
  or_txt <- paste0(a2$or_text, ifelse(logs, MK_LOG, ""), ifelse(ridge, MK_RIDGE, ""))
  cols <- c("Metric", "Physical quantity and computation", "Normal, median [interquartile range]", "Difficult, median [interquartile range]",
            "AUROC for Difficult (95% CI)", "Adjusted OR for failure per SD (95% CI)", "Perturbation family",
            "Spearman rho with severity (95% CI)")
  df <- setNames(data.frame(a2$label, unname(TBL_METRIC_DEF[a2$metric]), tbl_iqr_brackets(a2$normal_median_iqr),
                            tbl_iqr_brackets(a2$difficult_median_iqr), a2$auroc_text, or_txt, pc[, "family"], pc[, "rho"],
                            stringsAsFactors = FALSE), cols)

  d1 <- dg[dg$scope == "center1_oof" & dg$model == "univariable_adjusted", ]
  aliased <- unique(stats::na.omit(d1$aliased_terms))
  n_n <- a2$n_normal[1]; n_d <- a2$n_difficult[1]
  # share of Difficult among the analysed D2 frames (assoc_by_domain.csv), not the cohort, which still counts frames
  # without a detected field of view
  d2 <- bd[bd$scope == "center1_oof" & bd$domain == "D2" & bd$metric == OPTICAL_METRICS[1], ]
  d2_share <- if (nrow(d2) == 1L) d2$n_difficult / (d2$n_normal + d2$n_difficult) else NA_real_
  # the covariates (domain, instrument, ...) also have VIFs; the caveat is about the optical metrics only
  vmax <- vif[vif$scope == "center1_oof" & vif$term %in% OPTICAL_METRICS, ]
  if (!nrow(vmax)) stop("assoc_vif.csv has no center1_oof rows for the optical metrics", call. = FALSE)
  vmax <- vmax[which.max(vmax$vif), ]
  n_sev <- max(unlist(CFG$perturbation$severities))
  fn <- c(
    "AUROC, area under the receiver operating characteristic curve for separating Difficult from Normal frames (one half means no separation; values below one half mean the metric is lower in Difficult frames); CI, confidence interval; OR, odds ratio; SD, standard deviation; rho, Spearman rank correlation coefficient. Medians are followed by the interquartile range (first to third quartile) in brackets.",
    sprintf("Medians, AUROC and odds ratios use training-center out-of-fold frames: %s Normal and %s Difficult frames, with %s U-Net failures (IoU below %s, the frozen failure threshold). The expert label is confounded with acquisition domain%s, so the pooled AUROC partly reflects domain; within-domain AUROC values are given in Supplementary Table S1. The AUROC interval is a patient-clustered bootstrap percentile interval (patients resampled within domain); odds-ratio intervals are cluster-robust by patient and rho intervals are patient-clustered bootstrap intervals.",
            fmt_num(n_n, 0), fmt_num(n_d, 0), fmt_num(d1$events[1], 0), fmt_num(thr, 2),
            if (is.na(d2_share)) "" else sprintf(" (%s%% of D2 frames are Difficult)", fmt_num(100 * d2_share, 1))),
    sprintf("Odds ratios come from one logistic model per metric, adjusted for domain, channel, region, image format, instrument class and predicted instrument area, with patient-clustered (cluster-robust) standard errors.%s",
            if (length(aliased)) sprintf(" Terms dropped as single-level or aliased in these models: %s.", paste(aliased, collapse = "; ")) else ""),
    sprintf("Perturbation: Spearman correlation between the metric and synthetic perturbation severity (zero is unperturbed, the maximum is %s; %s frames from %s patients); 95%% CI from a patient-clustered bootstrap. The families shown are those the perturbation is built to move (names as in Figure 2).",
            fmt_num(n_sev, 0), fmt_num(max(rho$n_frames), 0), fmt_num(max(rho$n_patients), 0)))
  if (vmax$vif > vif_flag) {
    fn <- c(fn, sprintf("The metrics are correlated: in a joint linear model of all of them the largest variance inflation factor among the optical metrics was %s (%s; flag level %s), so the one-metric-at-a-time odds ratios should not be read as independent effects.",
                        fmt_num(vmax$vif, 1), METRIC_LABELS[[vmax$term]], fmt_num(vif_flag, 0)))
  }
  # one line per mark, in the order of the marks themselves
  if (any(pc[, "limit"] == "TRUE")) fn <- c(fn, paste(MK_LIMIT, "Known limitation: gradient energy counts added noise as sharpness, so sharpness rises with noise (the same mark as in Figure 2C)."))
  if (any(logs)) fn <- c(fn, paste(MK_LOG, "Odds ratio per SD of the logarithm of the metric."))
  if (any(pc[, "no_expected"] == "TRUE")) fn <- c(fn, paste(MK_NO_EXPECTED, "No perturbation family is built to move this metric; the family with the largest absolute correlation is shown."))
  if (any(ridge)) fn <- c(fn, paste(MK_RIDGE, "Ridge logistic regression with a patient-bootstrap interval, used because events per variable fell below the plan minimum."))
  list(df = df, footnote = fn, threshold = thr)
}

save_table2 <- function(t2 = table2()) {
  d <- tbl_wrap_docx(t2$df, names(t2$df)[3:4], " \\[", "\n[")
  d <- tbl_wrap_docx(d, c("AUROC for Difficult (95% CI)", "Adjusted OR for failure per SD (95% CI)"), " \\(", "\n(")
  for (cl in c("Perturbation family", "Spearman rho with severity (95% CI)")) d[[cl]] <- gsub("; ", "\n", d[[cl]])
  ft <- flex_three_line(d, align = c("l", "l", "c", "c", "c", "c", "l", "c"), font_size = 8, footnote = t2$footnote)
  ft <- flextable::set_header_labels(ft, values = list(
    `Normal, median [interquartile range]` = "Normal, median\n[interquartile\nrange]",
    `Difficult, median [interquartile range]` = "Difficult, median\n[interquartile\nrange]",
    `AUROC for Difficult (95% CI)` = "AUROC for\nDifficult\n(95% CI)",
    `Adjusted OR for failure per SD (95% CI)` = "Adjusted OR for\nfailure per SD\n(95% CI)",
    `Spearman rho with severity (95% CI)` = "Spearman rho with\nseverity (95% CI)"))
  ft <- tbl_fix_widths(ft, c(1.05, 2.3, 1.0, 1.0, 1.0, 1.1, 1.1, 1.45))
  c(docx = tbl_write_docx(list(ft), file.path(p("tables"), "Table2.docx"), landscape = TRUE),
    csv = tbl_write_csv(t2$df, file.path(p("tables"), "Table2.csv")))
}

# ---- Table 3: gate versus baselines ---------------------------------------------------------------------------

TBL3_METHODS <- c("random", "predicted_area", "entropy", "ens_disagreement", "optical_gate", "gate_plus_ensemble")
TBL3_DIGITS <- c(aurc = 3, auroc = 2, sel_risk_90 = 3, sel_risk_80 = 3, cal_slope = 2)
TBL3_PARTS <- c(internal = CENTER_ROLE[["1"]], external = CENTER_ROLE[["2"]])
# column-group headings over the two partitions
TBL3_SPANNERS <- c(internal = "Training center, cross-fitted", external = "External center, locked gate")
cap_first <- function(x) paste0(toupper(substring(x, 1, 1)), substring(x, 2))
MK_DESCRIPTIVE <- "†"

# External AURC difference, pixel entropy minus ensemble disagreement, from the locked-variant rows of sensitivity.csv
# (same resamples as gate_external.csv). NULL when the file or the rows are missing or do not reproduce the external table.
tbl_entropy_vs_ens <- function(ge, stats_dir) {
  path <- file.path(stats_dir, "sensitivity.csv")
  if (!file.exists(path)) return(NULL)
  sn <- tbl_read_stats("sensitivity.csv", stats_dir)
  r <- sn[sn$variant == "locked" & sn$method == "entropy", ]
  if (nrow(r) != 1L || anyNA(r[c("diff_vs_ens", "diff_vs_ens_lo", "diff_vs_ens_hi")])) return(NULL)
  ref <- ge$aurc[ge$partition == "external" & ge$method == "entropy"]
  if (length(ref) != 1L || abs(ref - r$aurc) > 1e-9) {
    message("table3: sensitivity.csv locked entropy AURC differs from gate_external.csv, so the entropy vs ensemble row is left out")
    return(NULL)
  }
  list(diff = r$diff_vs_ens, lo = r$diff_vs_ens_lo, hi = r$diff_vs_ens_hi)
}

table3 <- function(lock = tbl_lock(), thr = lock_failure_threshold(lock), stats_dir = p("results", "stats")) {
  rd <- function(f) tbl_read_stats(f, stats_dir)
  gi <- rd("gate_internal.csv"); ge <- rd("gate_external.csv"); gt <- rd("gate_transfer.csv")
  pu <- rd("gate_paired_diffs.csv"); ps <- rd("gate_transfer_paired_diffs.csv")
  res <- list(unet = list(tab = rbind(gi, ge), diff = pu), segformer = list(tab = gt, diff = ps))
  stat_names <- names(TBL3_DIGITS)
  sel_lab <- function(s) sprintf("Selective risk at %s%% coverage", sub("sel_risk_", "", s))
  stat_labels <- c(aurc = "AURC", auroc = "AUROC", sel_risk_90 = sel_lab("sel_risk_90"), sel_risk_80 = sel_lab("sel_risk_80"),
                   cal_slope = "Calibration slope")
  cols <- c("Method", unlist(lapply(names(TBL3_PARTS), function(pt) paste(TBL3_PARTS[[pt]], stat_labels))))

  one_row <- function(label, cells) setNames(as.data.frame(as.list(c(label, cells)), stringsAsFactors = FALSE), cols)
  method_cells <- function(tab, method) {
    unlist(lapply(names(TBL3_PARTS), function(pt) {
      r <- tab[tab$partition == pt & tab$method == method, ]
      stopifnot(nrow(r) == 1L)
      vapply(stat_names, function(s) {
        txt <- tbl_est_ci(r[[s]], r[[paste0(s, "_lo")]], r[[paste0(s, "_hi")]], TBL3_DIGITS[[s]])
        if (nzchar(txt) && s %in% c("auroc", "cal_slope") && isTRUE(r$binary_descriptive)) txt <- paste0(txt, MK_DESCRIPTIVE)
        txt
      }, "")
    }))
  }
  cmp <- GATE_COMPARISONS
  short <- c(optical_gate = TBL_METHOD_LABELS[["optical_gate"]], gate_plus_ensemble = TBL_METHOD_LABELS[["gate_plus_ensemble"]])
  nm <- function(x) if (x %in% names(short)) short[[x]] else lc_first(TBL_METHOD_LABELS[[x]])
  # Pixel entropy vs ensemble disagreement is not in gate_paired_diffs.csv; the locked-variant rows of sensitivity.csv hold its
  # external AURC difference on the same resamples (nothing is computed here). Internal and the other metrics stay blank.
  ent_ens <- tbl_entropy_vs_ens(ge, stats_dir)
  diff_cells <- function(tab, dif, i) {
    unlist(lapply(names(TBL3_PARTS), function(pt) {
      desc <- isTRUE(tab$binary_descriptive[tab$partition == pt][1])
      vapply(stat_names, function(s) {
        r <- dif[dif$partition == pt & dif$comparison == cmp$comparison[i] & dif$metric == s, ]
        if (!nrow(r)) return("")
        txt <- tbl_est_ci(r$diff, r$lo, r$hi, TBL3_DIGITS[[s]])
        if (s == "auroc" && desc) paste0(txt, MK_DESCRIPTIVE) else txt
      }, "")
    }))
  }

  rows <- list(); bold <- integer(); ital <- integer(); indent <- integer(); sections <- integer()
  add <- function(r, kind) {
    rows[[length(rows) + 1L]] <<- r
    n <- length(rows)
    if (kind == "section") { bold <<- c(bold, n); sections <<- c(sections, n) }
    if (kind == "sub") { ital <<- c(ital, n); sections <<- c(sections, n) }
    if (kind == "row") indent <<- c(indent, n)
  }
  heads <- c(unet = "U-Net outcomes (primary)", segformer = "SegFormer outcomes (cross-architecture transfer)")
  for (m in names(res)) {
    add(one_row(heads[[m]], rep("", length(cols) - 1L)), "section")
    for (mt in TBL3_METHODS) add(one_row(TBL_METHOD_LABELS[[mt]], method_cells(res[[m]]$tab, mt)), "row")
    add(one_row("Paired differences, first minus second", rep("", length(cols) - 1L)), "sub")
    for (i in seq_len(nrow(cmp))) {
      add(one_row(sprintf("%s vs %s", cap_first(nm(cmp$a[i])), nm(cmp$b[i])), diff_cells(res[[m]]$tab, res[[m]]$diff, i)), "row")
    }
    if (m == "unet" && !is.null(ent_ens)) {
      cells <- setNames(rep("", length(cols) - 1L), cols[-1])
      cells[paste(TBL3_PARTS[["external"]], stat_labels[["aurc"]])] <- tbl_est_ci(ent_ens$diff, ent_ens$lo, ent_ens$hi, TBL3_DIGITS[["aurc"]])
      add(one_row(sprintf("%s vs %s", TBL_METHOD_LABELS[["entropy"]], nm("ens_disagreement")), cells), "row")
    }
  }
  df <- do.call(rbind, rows)

  # footnotes: every number below is read from the CSVs, the lock or the plan
  cnt <- function(tab, pt) tab[tab$partition == pt, ][1, ]
  cnts <- function(pt) {
    u <- cnt(res$unet$tab, pt); s <- cnt(res$segformer$tab, pt)
    sprintf("%s: %s frames from %s patients; U-Net %s failures in %s patients, SegFormer %s failures in %s patients",
            TBL3_PARTS[[pt]], fmt_num(u$n_frames, 0), fmt_num(u$n_patients, 0), fmt_num(u$n_events, 0),
            fmt_num(u$n_patients_with_events, 0), fmt_num(s$n_events, 0), fmt_num(s$n_patients_with_events, 0))
  }
  cov_pct <- sub("sel_risk_", "", stat_names[startsWith(stat_names, "sel_risk_")])
  orc <- function(pt) fmt_num(res$unet$tab$aurc[res$unet$tab$partition == pt & res$unet$tab$method == "oracle"], 3)
  have_ee <- !is.null(ent_ens)
  entropy_txt <- c(entropy_fov = "the whole field of view", entropy_roi = "the region around the predicted mask")[lock$entropy_baseline_choice]
  fn <- c(
    sprintf("AURC, area under the risk-coverage curve: the mean risk (1 - IoU) of the frames a method keeps, averaged over all coverages (lower is better). AUROC, area under the receiver operating characteristic curve for detecting a failure (IoU below %s, the frozen failure threshold). Selective risk at %s%% and %s%% coverage, the mean risk of the %s%% and %s%% of frames ranked most reliable (lower is better). Calibration slope is shown only for methods that output a probability; an ideal slope equals one.",
            fmt_num(thr, 2), cov_pct[1], cov_pct[2], cov_pct[1], cov_pct[2]),
    sprintf("Intervals are patient-clustered bootstrap, 95%% percentile intervals (%s resamples, patients resampled within domain, one set of resamples shared by all methods). Training-center scores are cross-fitted (out-of-fold); external-center frames are scored by the locked gate. %s. %s.",
            fmt_num(CFG$statistics$bootstrap_resamples, 0), cnts("internal"), cnts("external")),
    "Random order is analytic: AURC and selective risk equal the mean risk and AUROC is one half. Paired differences are the first method minus the second on the shared resamples; a negative difference favors the first method for AURC and selective risk, a positive one for AUROC.",
    sprintf("SegFormer rows: the gate, the predicted-area model and the ensemble stack were fitted on U-Net failures and applied unchanged; each segmenter's own predicted area and entropy feed its baselines.%s Oracle (ordering by the true risk) reaches an AURC of %s internal and %s external for U-Net.",
            if (length(entropy_txt)) sprintf(" Pixel entropy, the mean binary entropy of the predicted foreground probability, is averaged over %s.", entropy_txt) else "", orc("internal"), orc("external")))
  fn <- c(fn, sprintf("Paired differences cover AURC, AUROC and selective risk at %s%% coverage (not at %s%%).%s", cov_pct[1], cov_pct[2],
                      if (have_ee) " The pixel entropy vs ensemble disagreement row is for the external center and AURC only; it comes from the locked-variant rows of the sensitivity analysis, which use the same resamples." else ""))
  margin <- na.omit(unique(c(pu$margin, ps$margin)))
  ni <- pu[pu$metric == "aurc" & !is.na(pu$noninferior), ]
  if (length(margin) && nrow(ni)) {
    verdict <- function(cmpid, pt) c(`TRUE` = "non-inferior", `FALSE` = "not shown non-inferior")[[as.character(ni$noninferior[ni$comparison == cmpid & ni$partition == pt])]]
    fn <- c(fn, sprintf("Non-inferiority margin %s in AURC (fixed in the analysis lock before external outcomes were opened; non-inferior when the upper 95%% limit of the difference is below the margin). Optical gate vs pixel entropy: %s in the training center, %s in the external center. Optical gate vs ensemble disagreement: %s in the training center, %s in the external center.",
                        fmt_num(margin[1], 3), verdict("gate_vs_entropy", "internal"), verdict("gate_vs_entropy", "external"),
                        verdict("gate_vs_ens", "internal"), verdict("gate_vs_ens", "external")))
  } else {
    fn <- c(fn, "No non-inferiority margin was retained in the analysis lock, so the comparisons with pixel entropy and ensemble disagreement are descriptive.")
  }
  if (any(c(gi$binary_descriptive, ge$binary_descriptive, gt$binary_descriptive) %in% TRUE)) {
    fn <- c(fn, sprintf("%s Descriptive only: fewer than %s failures or fewer than %s patients with a failure, so AUROC and calibration carry no inferential weight and AURC stays primary.",
                        MK_DESCRIPTIVE, fmt_num(CFG$endpoints$min_external_failure_events, 0), fmt_num(CFG$endpoints$min_external_patients_with_events, 0)))
  }
  list(df = df, bold_rows = bold, italic_rows = ital, indent_rows = indent, section_rows = sections, footnote = fn,
       stat_labels = stat_labels, threshold = thr)
}

save_table3 <- function(coh = NULL, t3 = table3()) {
  d <- t3$df
  num_cols <- names(d)[-1]
  d <- tbl_wrap_docx(d, num_cols, " \\(", "\n(")
  ft <- flex_three_line(d, align = c("l", rep("c", length(num_cols))), font_size = 7, bold_rows = t3$bold_rows,
                        indent_rows = t3$indent_rows, footnote = t3$footnote)
  k <- length(t3$stat_labels)
  lab <- setNames(rep(unname(t3$stat_labels), 2L), num_cols)
  lab <- sub("Selective risk at ", "Selective risk\nat ", sub(" coverage", "\ncoverage", lab))
  lab <- sub("Calibration slope", "Calibration\nslope", lab)
  ft <- flextable::set_header_labels(ft, values = as.list(c(Method = "Method", lab)))
  ft <- flextable::add_header_row(ft, values = c("", unname(TBL3_SPANNERS)), colwidths = c(1, k, k), top = TRUE)
  ft <- flextable::align(ft, part = "header", align = "center")
  ft <- flextable::align(ft, j = 1, part = "header", align = "left")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::italic(ft, i = t3$italic_rows, part = "body")
  for (r in t3$section_rows) ft <- flextable::merge_at(ft, i = r, j = seq_len(ncol(d)))
  heavy <- officer::fp_border(color = "black", width = 1.5)
  thin <- officer::fp_border(color = "black", width = 0.75)
  ft <- flextable::border_remove(ft)
  ft <- flextable::hline_top(ft, border = heavy, part = "header")
  ft <- flextable::hline(ft, i = 1, j = 1L + seq_len(k), border = thin, part = "header")
  ft <- flextable::hline(ft, i = 1, j = 1L + k + seq_len(k), border = thin, part = "header")
  ft <- flextable::hline_bottom(ft, border = thin, part = "header")
  ft <- flextable::hline_bottom(ft, border = heavy, part = "body")
  ft <- flextable::hline(ft, i = t3$bold_rows[-1] - 1L, border = thin, part = "body")
  ft <- flextable::padding(ft, j = seq_len(ncol(d))[-1], padding.left = 2, padding.right = 2, part = "all")   # room for "(-0.015 to -0.007)"
  ft <- tbl_fix_widths(ft, c(1.55, rep(0.85, 2L * k)))
  c(docx = tbl_write_docx(list(ft), file.path(p("tables"), "Table3.docx"), landscape = TRUE),
    csv = tbl_write_csv(t3$df, file.path(p("tables"), "Table3.csv")))
}

# ---- build ------------------------------------------------------------------------------------------------------

tbl_inputs <- function() {
  st <- function(...) file.path(p("results", "stats"), c(...))
  list(table1_panel_b = p("derived", "predictions.parquet"),
       table2 = st("assoc_table2.csv", "perturb_rho.csv", "assoc_vif.csv", "assoc_diagnostics.csv", "assoc_by_domain.csv"),
       table3 = st("gate_internal.csv", "gate_external.csv", "gate_transfer.csv", "gate_paired_diffs.csv", "gate_transfer_paired_diffs.csv"))
}

build_tables <- function(coh = compute_cohort()) {
  say <- function(name, paths) cat(name, "->", paste(basename(paths), collapse = ", "), "\n")
  skip <- function(name) {
    gone <- tbl_inputs()[[name]]
    gone <- gone[!file.exists(gone)]
    if (length(gone)) message(name, ": skipped, inputs not available yet: ", paste(sub(paste0(V1, "/"), "", gone, fixed = TRUE), collapse = ", "))
    length(gone) > 0L
  }
  out <- list()
  out$table1_panel_a <- save_table1_panel_a(coh)
  say("save_table1_panel_a", out$table1_panel_a)
  # the lock is checked before panel B (and so before any Center 2 outcome) is touched
  lock <- tbl_read_lock()
  why <- tbl_panel_b_blocked(lock)
  if (!is.null(why)) message("table1_panel_b: skipped, ", why)
  if (is.null(why) && !skip("table1_panel_b")) {
    b <- table1_panel_b(coh, lock = lock)
    out$table1_panel_b <- save_table1_panel_b(coh, b)
    say("save_table1_panel_b", out$table1_panel_b)
    out$table1 <- save_table1(coh, b)
    say("save_table1", out$table1)
    register_seg_keys(b$summary)
    write_results_part("20_tables")
  } else message("save_table1: skipped (panel B missing), only Table1_panelA was written")
  if (!skip("table2")) { out$table2 <- save_table2(); say("save_table2", out$table2) }
  if (!skip("table3")) { out$table3 <- save_table3(coh); say("save_table3", out$table3) }
  invisible(out)
}

if (sys.nframe() == 0L) build_tables()
