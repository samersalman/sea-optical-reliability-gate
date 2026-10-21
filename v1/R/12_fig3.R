# Figure 3: gate performance. A, risk-coverage on the training center (cross-fitted, U-Net); B, the same on the locked
# external center (U-Net); C, cross-architecture transfer (U-Net-trained gate and baselines scored on SegFormer
# outcomes, external center); D, which stream frames the locked gate flags in each of the 20 videos.
# Everything is read from results/stats/*.csv written by 04b, 05 and 07; nothing is recomputed here.
#   Rscript R/12_fig3.R      (OPTICS_DATA=synthetic for development)

if (!exists("read_stat_csv", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (exists("V1", envir = globalenv())) file.path(get("V1", envir = globalenv()), "R")
             else if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "11_fig2.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

FIG3_METHODS <- names(METHOD_COLOURS)
# The network has one logit, so the baseline is the mean binary entropy of the predicted foreground probability:
# "Pixel entropy" (METHOD_LABELS, 00_setup.R).
FIG3_METHOD_LABELS <- METHOD_LABELS
FIG3_BAND_METHODS <- c("optical_gate", "entropy", "ens_disagreement")     # bands only where the comparison matters
FIG3_LINEWIDTH <- c(random = 0.35, predicted_area = 0.5, entropy = 0.5, ens_disagreement = 0.5,
                    optical_gate = 0.9, gate_plus_ensemble = 0.5, oracle = 0.35)
stopifnot(setequal(names(FIG3_LINEWIDTH), FIG3_METHODS))
FIG3_COVERAGE_LEVELS <- unlist(CFG$endpoints$coverage_levels)
FIG3_AURC_DIGITS <- 3              # as registered for AURC in 04b
FIG3_LABEL_PT <- 7
FIG3_TICK_HALF <- 0.4              # half-height of a flag tick, in row units
FIG3_HEIGHT_MM <- 172

RC_COLS <- c("partition", "model", "method", "coverage", "selective_risk", "lo", "hi")
GATE_COLS <- c("partition", "model", "method", "aurc")
# partition and model behind panels A, B and C
FIG3_RC_PANELS <- list(a = c("internal", "unet"), b = c("external", "unet"), c = c("external", "segformer"))

# One y maximum for A to C, so heights can be compared across panels: the highest point of any method's curve (not of
# the bands, which explode where a handful of frames is kept).
fig3_ytop <- function(curves) {
  top <- vapply(FIG3_RC_PANELS, function(pm) max(curves$selective_risk[curves$partition == pm[1] & curves$model == pm[2]], na.rm = TRUE), 0)
  max(top)
}

# ---- A to C: risk-coverage ----------------------------------------------------------------------------------
# curves: risk_coverage_curves + _transfer stacked; gate: gate_internal + gate_external + gate_transfer stacked.
fig3_rc_panel <- function(curves, gate, part, mdl, title, ytop = NULL) {
  need_cols(curves, RC_COLS, "risk-coverage table"); need_cols(gate, GATE_COLS, "gate table")
  cv <- curves[curves$partition == part & curves$model == mdl, ]
  if (!nrow(cv)) stop("No risk-coverage rows for ", part, " / ", mdl, call. = FALSE)
  stop_unknown(cv$method, FIG3_METHODS, "method")
  if (!setequal(cv$method, FIG3_METHODS)) stop("Risk-coverage curves for ", part, " / ", mdl, " lack: ",
                                              paste(setdiff(FIG3_METHODS, cv$method), collapse = ", "), call. = FALSE)
  ga <- gate[gate$partition == part & gate$model == mdl, ]
  aurc <- ga$aurc[match(FIG3_METHODS, ga$method)]
  if (anyNA(aurc)) stop("AURC missing in the gate tables for ", part, " / ", mdl, call. = FALSE)
  # one legend that is also the AURC table: name, then the value read from the gate CSV
  legend_labels <- setNames(paste0(FIG3_METHOD_LABELS[FIG3_METHODS], "  ", fmt_num(aurc, FIG3_AURC_DIGITS)), FIG3_METHODS)

  cv$method <- factor(cv$method, FIG3_METHODS)
  bands <- cv[cv$method %in% FIG3_BAND_METHODS, ]
  # The y range follows the curves, not the bands (and is shared by A to C, see fig3_ytop): at the lowest coverages a
  # handful of frames makes the bands explode, so wide bands are clipped at the panel edge rather than allowed to
  # flatten the rest.
  if (is.null(ytop)) ytop <- max(cv$selective_risk, na.rm = TRUE)

  ggplot(cv, aes(coverage, selective_risk)) +
    geom_vline(xintercept = FIG3_COVERAGE_LEVELS, linewidth = 0.25, colour = "grey70", linetype = "11") +
    annotate("text", x = FIG3_COVERAGE_LEVELS, y = 0, label = scales::percent(FIG3_COVERAGE_LEVELS, accuracy = 1),
             vjust = -0.5, size = FIG3_LABEL_PT, size.unit = "pt", colour = "grey40", family = FONT) +
    geom_ribbon(data = bands, aes(ymin = lo, ymax = hi, fill = method), alpha = 0.15, colour = NA, na.rm = TRUE) +
    geom_line(aes(colour = method, linetype = method, linewidth = method), na.rm = TRUE) +
    scale_colour_manual(values = METHOD_COLOURS, breaks = FIG3_METHODS, labels = legend_labels, name = "Method, AURC") +
    scale_linetype_manual(values = METHOD_LINETYPES, breaks = FIG3_METHODS, labels = legend_labels, name = "Method, AURC") +
    scale_linewidth_manual(values = FIG3_LINEWIDTH, breaks = FIG3_METHODS, labels = legend_labels, name = "Method, AURC") +
    scale_fill_manual(values = METHOD_COLOURS, guide = "none") +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1), labels = scales::label_percent(accuracy = 1),
                       expand = expansion(mult = 0.01)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.03))) +
    coord_cartesian(ylim = c(0, ytop), clip = "on") +
    guides(colour = guide_legend(ncol = 2, title.position = "top", byrow = FALSE)) +
    labs(title = title, x = "Coverage (share of frames kept)", y = "Selective risk (1 − IoU)") +
    theme_optics() +
    theme(legend.key.width = unit(5.5, "mm"), legend.key.height = unit(3, "mm"),
          legend.key.spacing.y = unit(0.3, "mm"), legend.justification = "left",
          legend.text = element_text(size = 7), legend.title = element_text(size = 7))
}

# ---- D: stream flag timeline ---------------------------------------------------------------------------------
fig3_panel_d <- function(flags, summ, period_s = CFG$stream$frame_period_s) {
  share <- summ[summ$scope == "video" & summ$statistic == "flagged_share", c("video_patient", "est")]
  names(share)[2] <- "share"
  if (!setequal(share$video_patient, flags$video_patient)) {
    stop("stream_summary.csv and stream_flags.csv list different videos", call. = FALSE)
  }
  share <- share[order(-share$share, share$video_patient), ]      # highest share on top
  share$row <- rev(seq_len(nrow(share)))
  share$lab <- paste0(fmt_num(100 * share$share, 1), "%")

  fl <- merge(flags, share[c("video_patient", "row")], by = "video_patient")
  fl$minute <- fl$frame_idx * period_s / 60
  extent <- stats::aggregate(minute ~ video_patient + row, fl, range)
  extent <- data.frame(extent[c("video_patient", "row")], t0 = extent$minute[, 1], t1 = extent$minute[, 2])
  ticks <- fl[fl$flagged, ]
  overall <- function(stat) summ$est[summ$scope == "overall" & summ$statistic == stat]
  subtitle <- paste0(fmt_num(100 * overall("flagged_share_stream_90"), 1), "% of stream frames flagged (training-center reference ",
                     fmt_num(100 * overall("flagged_share_center1_oof_90"), 1), "%)")

  n <- nrow(share)
  # one-line key, drawn as a legend; "flagged" is defined by the threshold fixed at the primary coverage (plan)
  key <- c(span = "Span of analyzed frames", flag = sprintf("Flagged frame (%s coverage threshold)", scales::percent(FIG3_COVERAGE_LEVELS[1], accuracy = 1)))
  ggplot() +
    geom_segment(data = extent, aes(x = t0, xend = t1, y = row, yend = row, colour = key[["span"]]), linewidth = 1.1,
                 lineend = "butt", key_glyph = "path") +
    geom_segment(data = ticks, aes(x = minute, xend = minute, y = row - FIG3_TICK_HALF, yend = row + FIG3_TICK_HALF, colour = key[["flag"]]),
                 linewidth = 0.3, key_glyph = "vpath") +
    geom_text(data = share, aes(x = Inf, y = row, label = lab), hjust = -0.2, size = FIG3_LABEL_PT, size.unit = "pt", family = FONT) +
    # column header above the panel (not on a row's height), so it cannot crowd the first video whatever n is
    annotate("text", x = Inf, y = Inf, label = "Flagged", hjust = -0.2, vjust = -0.8, size = FIG3_LABEL_PT, size.unit = "pt",
             fontface = "bold", family = FONT) +
    scale_colour_manual(values = setNames(c("grey85", METHOD_COLOURS[["optical_gate"]]), key), breaks = unname(key), name = NULL,
                        guide = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = c(1.1, 0.5)))) +
    scale_y_continuous(breaks = share$row, labels = share$video_patient, limits = c(0.4, n + 0.6), expand = c(0, 0)) +
    scale_x_continuous(expand = expansion(mult = c(0.01, 0.02))) +
    coord_cartesian(clip = "off") +
    labs(title = "Stream flags by video", subtitle = subtitle, x = "Time in video (min)", y = "Video, sorted by share flagged") +
    theme_optics() +
    theme(plot.subtitle = element_text(size = 6.5, colour = "grey30", hjust = 0), plot.margin = margin(3, 30, 3, 3),
          axis.text.y = element_text(size = 7), axis.title.y = element_text(margin = margin(r = 4)),
          legend.position = "bottom", legend.justification = "left", legend.key.width = unit(7, "mm"),
          legend.key.height = unit(3, "mm"), legend.text = element_text(size = 7))
}

# ---- assembly -----------------------------------------------------------------------------------------------
fig3_data <- function() {
  rc <- c("risk_coverage_curves", "risk_coverage_curves_transfer")
  gt <- c("gate_internal", "gate_external", "gate_transfer")
  list(curves = dplyr::bind_rows(lapply(rc, read_stat_csv, cols = RC_COLS)),
       gate = dplyr::bind_rows(lapply(gt, read_stat_csv, cols = GATE_COLS)),
       flags = read_stat_csv("stream_flags", c("video_patient", "frame_idx", "flagged")),
       summ = read_stat_csv("stream_summary", c("scope", "video_patient", "statistic", "est")))
}

fig3_panels <- function(d = fig3_data()) {
  ytop <- fig3_ytop(d$curves)
  list(a = fig3_rc_panel(d$curves, d$gate, "internal", "unet", "Training center, cross-fitted", ytop),
       b = fig3_rc_panel(d$curves, d$gate, "external", "unet", "External center, locked gate", ytop),
       c = fig3_rc_panel(d$curves, d$gate, "external", "segformer", "Transfer to SegFormer, external center", ytop),
       d = fig3_panel_d(d$flags, d$summ))
}

fig3 <- function(panels = fig3_panels()) {
  patchwork::wrap_plots(panels$a, panels$b, panels$c, panels$d, ncol = 2) +
    patchwork::plot_annotation(tag_levels = "A", theme = FIG2_TAG_THEME)
}

FIG3_STAT_FILES <- c("risk_coverage_curves", "risk_coverage_curves_transfer", "gate_internal", "gate_external",
                     "gate_transfer", "stream_flags", "stream_summary")

if (sys.nframe() == 0L) {
  missing_stats <- FIG3_STAT_FILES[!file.exists(file.path(p("results", "stats"), paste0(FIG3_STAT_FILES, ".csv")))]
  if (length(missing_stats)) {
    message("Figure 3 skipped: results/stats/", paste(missing_stats, collapse = ".csv, "), ".csv not found")
  } else {
    paths <- save_fig(fig3(), "Figure3", height_mm = FIG3_HEIGHT_MM)
    cat("Figure 3 written:", paste(basename(paths), collapse = ", "), "\n")
  }
}
