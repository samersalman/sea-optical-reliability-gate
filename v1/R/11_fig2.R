# Figure 2: A, distributions of six optical metrics by domain and scenario (from the frame table);
# B, perturbation dose-response (IoU vs severity, one facet per family); C, heat-map of Spearman rho between
# each metric and severity. B and C only read results/stats/perturb_*.csv (written by 06_perturb.R), never recompute.
#   Rscript R/11_fig2.R      (OPTICS_DATA=synthetic for development)
# Panel A alone is always previewed; the full figure is written when the perturbation CSVs exist.

if (!exists("load_frame_table", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (exists("V1", envir = globalenv())) file.path(get("V1", envir = globalenv()), "R")
             else if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

# Display order and axis transform per metric: heavy right tails and exact zeros (under-exposed and specular
# fractions) use a square-root axis, sharpness spans a decade so it is logarithmic.
FIG2A_METRICS <- c("lum_mean", "under_frac", "spec_frac", "sharp_tenengrad", "haze_dcp", "red_dom")
FIG2A_TRANSFORM <- c(lum_mean = "identity", under_frac = "sqrt", spec_frac = "sqrt",
                     sharp_tenengrad = "log10", haze_dcp = "identity", red_dom = "identity")
FIG2A_SCENARIO_ALPHA <- c(Normal = 0.3, Difficult = 0.95)

# same "4.1e-4" notation as Figure 1B and Table 2 (sci_label, 00_setup.R)
fig2_sci <- function(x) sci_label(x)

fig2_panel_a <- function(ft = load_frame_table(with_predictions = FALSE)) {
  stopifnot(all(FIG2A_METRICS %in% names(ft)))
  long <- ft |>
    select(domain, scenario, all_of(FIG2A_METRICS)) |>
    tidyr::pivot_longer(all_of(FIG2A_METRICS), names_to = "metric", values_to = "value") |>
    mutate(metric = factor(metric, FIG2A_METRICS, METRIC_LABELS[FIG2A_METRICS]),
           scenario = factor(scenario, c("Normal", "Difficult")))

  # one free-scale facet per metric, each with its own transform; ggplot2 has one scale per facet column,
  # so the six facets are built separately and joined
  one <- function(m) {
    lab <- METRIC_LABELS[[m]]
    d <- long[long$metric == lab, ]
    tr <- FIG2A_TRANSFORM[[m]]
    brk <- switch(m,
      under_frac = c(0, 0.05, 0.2, 0.5), spec_frac = c(0, 0.02, 0.1, 0.2, 0.3),
      sharp_tenengrad = c(1e-4, 3e-4, 1e-3, 3e-3), waiver())
    lab_fn <- switch(m, sharp_tenengrad = fig2_sci, waiver())
    ggplot(d, aes(domain, value, fill = domain, colour = domain, alpha = scenario, group = interaction(domain, scenario))) +
      geom_violin(position = position_dodge(0.9), width = 0.9, scale = "width", linewidth = 0.25, bw = "nrd0") +
      geom_boxplot(position = position_dodge(0.9), width = 0.16, outlier.shape = NA, linewidth = 0.3,
                   fill = "white", alpha = 1, colour = "black", coef = 0, whisker.linetype = "blank", staplewidth = 0) +
      scale_y_continuous(transform = tr, breaks = brk, labels = lab_fn, expand = expansion(mult = c(0.02, 0.04))) +
      scale_fill_domain(guide = "none") + scale_colour_domain(guide = "none") +
      scale_alpha_manual(values = FIG2A_SCENARIO_ALPHA, name = "Scenario") +
      labs(title = lab, x = NULL, y = NULL) +
      theme_optics()
  }
  plots <- lapply(FIG2A_METRICS, one)
  patchwork::wrap_plots(plots, ncol = 3, guides = "collect") &
    guides(alpha = guide_legend(override.aes = list(fill = "grey30", colour = "grey30"))) &
    theme(legend.position = "bottom")
}

fig2_panel_a_preview <- function() {
  g <- fig2_panel_a()
  dir <- p("figures")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(dir, "Figure2_panelA_preview.png")
  ragg::agg_png(path, width = round(CFG$exhibits$width_mm / 25.4 * CFG$exhibits$dpi), height = round(85 / 25.4 * CFG$exhibits$dpi),
                units = "px", res = CFG$exhibits$dpi, background = "white")
  print(g)
  dev.off()
  path
}

# ---- shared helpers for the stats-driven panels (also used by 12_fig3.R) --------------------------------------
need_cols <- function(df, cols, what) {
  miss <- setdiff(cols, names(df))
  if (length(miss)) stop(what, " is missing columns: ", paste(miss, collapse = ", "), call. = FALSE)
  invisible(df)
}

read_stat_csv <- function(name, cols = character()) {
  path <- file.path(p("results", "stats"), paste0(name, ".csv"))
  if (!file.exists(path)) {
    stop("File not found: ", path, "\n  (written by the stats scripts; use OPTICS_DATA=synthetic for development)", call. = FALSE)
  }
  need_cols(tibble::as_tibble(utils::read.csv(path, stringsAsFactors = FALSE)), cols, paste0(name, ".csv"))
}

FIG2_FAMILIES <- c("defocus", "haze", "exposure_under", "exposure_over", "blood_absorption", "glare", "noise")
FIG2_FAMILY_LABELS <- PERTURB_FAMILY_LABELS      # shared with Table 2
stopifnot(identical(names(FIG2_FAMILY_LABELS), FIG2_FAMILIES))
# Models are told apart by line type and grey level, not colour: the Okabe-Ito hues are already taken by
# domains (Figures 1, 2A) and methods (Figure 3), so a third meaning would blur the exhibits.
FIG2_MODEL_LABELS <- c(unet = "U-Net", segformer = "SegFormer")
FIG2_MODEL_COLOURS <- c(unet = "#000000", segformer = "#6E6E6E")
FIG2_MODEL_LINETYPES <- c(unet = "solid", segformer = "longdash")
# diverging scale for rho: purple (negative) to gold (positive). Not the domain hues of panel A (blue, vermilion,
# green): the old orange sat within one colour-difference unit of the D2 vermilion, gold is about 40 units away.
# Purple vs gold is the blue-yellow axis, which the common colour-vision deficiencies keep.
FIG2_RHO_COLOURS <- c(neg = "#5E3C99", pos = "#E6AB02")
FIG2_RHO_WHITE_BELOW <- -0.75      # print white only on the darkest purple; gold and the paler fills take black
FIG2_CELL_TEXT_PT <- 7
FIG2_NOTE_PT <- 7                  # footers under panels B and C
# column headers stay at 6 pt: "absorption" is about 11.5 mm wide at 7 pt, wider than a heat-map cell (about 10 mm), so
# the headers of neighbouring families would collide
FIG2C_HEADER_PT <- 6
FIG2B_TICK_HALF <- 0.035           # half-height of the severity tick on the collapse-rule line, as a share of the y range
FIG2_WRAP <- 9                     # heat-map column labels wrap at this many characters

stop_unknown <- function(x, known, what) {
  bad <- setdiff(unique(x), known)
  if (length(bad)) stop("Unknown ", what, ": ", paste(bad, collapse = ", "), call. = FALSE)
}

# ---- Panel B: dose-response ---------------------------------------------------------------------------------
FIG2B_COLS <- c("model", "family", "severity", "median_iou", "median_lo", "median_hi")
FIG2B_THR_COLS <- c("model", "family", "threshold_iou", "severity_interp", "never_reached")
FIG2C_COLS <- c("family", "metric", "rho", "expected_pair", "known_limitation")

# Shared y range of panel B, from the plotted data (medians and band edges of every facet), never from 0 to 1 by
# default: when no family gets near the collapse rule the curves would be squashed into the top of the panel.
# Lower bound = the lowest plotted value (and the rule level when a line is drawn) rounded down to a clean tick;
# upper bound = 1 (IoU), raised only if a value exceeds it, so nothing is clipped.
fig2b_ylim <- function(dr, level = NULL) {
  vals <- c(dr$median_iou, dr$median_lo, dr$median_hi, level)
  vals <- vals[is.finite(vals)]
  lo_raw <- min(vals)
  hi <- max(1, vals)
  step <- if (hi - lo_raw <= 0.3) 0.05 else if (hi - lo_raw <= 0.6) 0.1 else 0.25
  lo <- floor(lo_raw / step + 1e-9) * step
  list(limits = round(c(lo, hi), 4), step = step)
}

fig2_panel_b <- function(dr = read_stat_csv("perturb_doseresponse", FIG2B_COLS),
                         thr = read_stat_csv("perturb_threshold", FIG2B_THR_COLS)) {
  need_cols(dr, FIG2B_COLS, "dose-response table"); need_cols(thr, FIG2B_THR_COLS, "threshold table")
  stop_unknown(c(dr$family, thr$family), FIG2_FAMILIES, "perturbation family")
  stop_unknown(c(dr$model, thr$model), names(FIG2_MODEL_LABELS), "model")
  level <- unique(thr$threshold_iou)
  if (length(level) != 1L || !is.finite(level)) stop("perturb_threshold.csv needs one finite threshold_iou", call. = FALSE)
  fam <- function(x) factor(x, FIG2_FAMILIES, FIG2_FAMILY_LABELS)
  mod <- function(x) factor(x, names(FIG2_MODEL_LABELS), FIG2_MODEL_LABELS)

  dr <- mutate(dr, family = fam(family), model = mod(model))
  # The line is perturbation.rule_first_below_iou (the collapse rule, from perturb_threshold.csv), a descriptive
  # criterion; it is not the failure threshold of the gate. No tick when the median never reaches it;
  # severity_interp is the linear crossing point.
  mk <- paste0("Median first below IoU ", format(level))
  cross <- thr |>
    filter(!never_reached, is.finite(severity_interp)) |>
    mutate(family = fam(family), model = mod(model), y = level, key = mk)
  # The rule is drawn only in facets where it is relevant: a model crosses it there, or the plotted data
  # (medians and bands) span it. Elsewhere it would only stretch the axis.
  span <- dr |>
    group_by(family) |>
    summarise(lo = min(c(median_iou, median_lo), na.rm = TRUE), hi = max(c(median_iou, median_hi), na.rm = TRUE), .groups = "drop")
  shown <- levels(dr$family)[levels(dr$family) %in% union(as.character(cross$family), as.character(span$family[span$lo <= level & span$hi >= level]))]
  yl <- fig2b_ylim(dr, if (length(shown)) level)
  lims <- yl$limits
  brk <- round(seq(lims[1], lims[2], by = yl$step), 4)
  if (length(shown) && level >= lims[1] && level <= lims[2]) {
    # the rule level is a tick of its own; a regular tick closer than 6% of the range would collide with its label
    brk <- sort(c(brk[abs(brk - level) >= 0.06 * diff(lims)], level))
  }
  digits <- if (all(abs(brk * 10 - round(brk * 10)) < 1e-9)) 1L else 2L
  rng <- diff(lims)
  # every row says the rule was not reached: the line is dropped from the facets and the footer says so
  none_reached <- nrow(thr) > 0L && all(thr$never_reached %in% TRUE)
  note <- if (none_reached) paste0("Collapse rule (median IoU ", format(level), ") not reached in any family")
  # the lower end of the shared axis is read from the data, so the note appears only when the axis does not start at zero
  axis_note <- if (lims[1] > 0) paste0("Vertical axis starts at ", formatC(lims[1], format = "f", digits = digits))
  note <- if (length(c(note, axis_note))) paste(c(note, axis_note), collapse = "\n")

  g <- ggplot(dr, aes(severity, median_iou, colour = model, fill = model, linetype = model))
  if (length(shown)) {
    g <- g + geom_hline(data = tibble(family = factor(shown, levels(dr$family)), y = level), aes(yintercept = y),
                        inherit.aes = FALSE, linewidth = 0.3, linetype = "22", colour = "grey45")
  }
  g <- g +
    geom_ribbon(aes(ymin = median_lo, ymax = median_hi), alpha = 0.16, colour = NA, na.rm = TRUE) +
    geom_line(linewidth = 0.5, na.rm = TRUE) +
    geom_point(size = 0.8, na.rm = TRUE)
  if (nrow(cross)) {
    g <- g +
      geom_segment(data = mutate(cross, half = FIG2B_TICK_HALF * rng),
                   aes(x = severity_interp, xend = severity_interp, y = y - half, yend = y + half, colour = model, linewidth = key),
                   inherit.aes = FALSE, lineend = "butt", key_glyph = "vpath", na.rm = TRUE,
                   show.legend = c(colour = FALSE, linewidth = TRUE)) +
      scale_linewidth_manual(values = setNames(0.8, mk), name = NULL) +
      guides(linewidth = guide_legend(order = 2, override.aes = list(colour = "black")))
  }
  if (length(shown)) {
    # label sits under the line in the first facet that draws it, above the line when there is no room below
    below <- (level - lims[1]) / rng >= 0.3
    g <- g + geom_text(data = tibble(family = factor(shown[1], levels(dr$family)), severity = min(dr$severity),
                                     median_iou = level + (if (below) -0.04 else 0.04) * rng,
                                     lab = paste0("Collapse rule\nmedian IoU\n", format(level))),   # 3 lines: facets are only about 13 mm wide
                       aes(severity, median_iou, label = lab), inherit.aes = FALSE, hjust = 0, vjust = if (below) 1 else 0,
                       lineheight = 0.9, size = 6, size.unit = "pt", colour = "grey35", family = FONT)
  }
  g +
    facet_wrap(vars(family), ncol = 4, labeller = labeller(family = label_wrap_gen(12))) +
    scale_colour_manual(values = setNames(FIG2_MODEL_COLOURS, FIG2_MODEL_LABELS), name = NULL) +
    scale_fill_manual(values = setNames(FIG2_MODEL_COLOURS, FIG2_MODEL_LABELS), name = NULL) +
    scale_linetype_manual(values = setNames(FIG2_MODEL_LINETYPES, FIG2_MODEL_LABELS), name = NULL) +
    scale_x_continuous(breaks = sort(unique(dr$severity)), expand = expansion(add = 0.3)) +
    scale_y_continuous(limits = lims, breaks = brk, labels = function(x) formatC(x, format = "f", digits = digits),
                       expand = expansion(mult = c(0.01, 0.02))) +
    guides(colour = guide_legend(order = 1, override.aes = list(shape = NA, size = 0.5)),
           linetype = guide_legend(order = 1), fill = guide_legend(order = 1)) +
    labs(x = "Severity level (0 = unperturbed)", y = "Median IoU", caption = note) +
    theme_optics() +
    theme(legend.position = "bottom", legend.box = "horizontal", legend.key.width = unit(7, "mm"),
          legend.spacing.x = unit(3, "mm"), legend.justification = "left",
          panel.spacing.x = unit(1.5, "mm"), panel.spacing.y = unit(2.5, "mm"), strip.text = element_text(size = 7, face = "bold", hjust = 0),
          plot.caption = element_text(size = FIG2_NOTE_PT, hjust = 0, colour = "grey25", margin = margin(t = 2)),
          plot.caption.position = "plot")
}

# ---- Panel C: rho heat-map ----------------------------------------------------------------------------------
fig2_panel_c <- function(rho = read_stat_csv("perturb_rho", FIG2C_COLS)) {
  need_cols(rho, FIG2C_COLS, "rho table")
  stop_unknown(rho$family, FIG2_FAMILIES, "perturbation family")
  stop_unknown(rho$metric, OPTICAL_METRICS, "optical metric")
  if (anyDuplicated(rho[c("family", "metric")])) stop("perturb_rho.csv has duplicated family x metric rows", call. = FALSE)
  d <- rho |>
    mutate(family = factor(family, FIG2_FAMILIES, FIG2_FAMILY_LABELS),
           metric = factor(metric, rev(OPTICAL_METRICS), rev(METRIC_LABELS[OPTICAL_METRICS])),
           cell_text = ifelse(is.na(rho), "", paste0(fmt_num(rho, 2), ifelse(known_limitation, "\u2020", ""))),
           cell_ink = ifelse(!is.na(rho) & rho < FIG2_RHO_WHITE_BELOW, "white", "black"))
  outlined <- filter(d, expected_pair)
  key <- "Expected pair"

  ggplot(d, aes(family, metric)) +
    geom_tile(aes(fill = rho), colour = "white", linewidth = 0.4) +
    geom_tile(data = mutate(outlined, key = key), aes(colour = key), fill = NA, linewidth = 0.55) +
    geom_text(aes(label = cell_text, colour = NULL), colour = d$cell_ink, size = FIG2_CELL_TEXT_PT, size.unit = "pt", family = FONT) +
    scale_fill_gradient2(low = FIG2_RHO_COLOURS[["neg"]], mid = "white", high = FIG2_RHO_COLOURS[["pos"]], midpoint = 0,
                         limits = c(-1, 1), breaks = c(-1, -0.5, 0, 0.5, 1), na.value = "grey90",
                         name = "Spearman rho with severity",
                         guide = guide_colourbar(order = 1, title.position = "top", barwidth = unit(34, "mm"), barheight = unit(2.4, "mm"),
                                                 ticks.colour = "black", ticks.linewidth = 0.3, frame.colour = NA)) +
    scale_colour_manual(values = setNames("black", key), name = NULL,
                        guide = guide_legend(order = 2, override.aes = list(fill = NA, linewidth = 0.55))) +
    scale_x_discrete(position = "top", labels = scales::label_wrap(FIG2_WRAP), expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(x = NULL, y = NULL,
         caption = if (any(d$known_limitation %in% TRUE)) "\u2020 Known limitation: gradient energy rises with added noise" else NULL) +
    theme_optics() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          plot.caption = element_text(size = FIG2_NOTE_PT, hjust = 0, colour = "grey25", margin = margin(t = 2)),
          plot.caption.position = "plot",
          axis.text.x.top = element_text(size = FIG2C_HEADER_PT, lineheight = 0.9, vjust = 0),
          axis.text.y = element_text(size = 7),
          legend.box = "horizontal", legend.box.just = "bottom", legend.spacing.x = unit(4, "mm"),
          legend.justification = "left")
}

# ---- assembly -----------------------------------------------------------------------------------------------
# Panel A is itself a patchwork of six facets, so it is wrapped (otherwise its sub-facets would be tagged).
FIG2_HEIGHT_MM <- 178
FIG2_TAG_THEME <- theme(plot.tag = element_text(face = "bold", size = 10, family = FONT))

fig2 <- function(panel_a = fig2_panel_a(), panel_b = fig2_panel_b(), panel_c = fig2_panel_c()) {
  top <- patchwork::wrap_elements(full = panel_a) + FIG2_TAG_THEME
  bottom <- panel_b + panel_c + patchwork::plot_layout(widths = c(0.47, 0.53))
  (top / bottom) + patchwork::plot_layout(heights = c(85, 93)) +
    patchwork::plot_annotation(tag_levels = "A", theme = FIG2_TAG_THEME)
}

FIG2_STAT_FILES <- c("perturb_doseresponse", "perturb_threshold", "perturb_rho")

if (sys.nframe() == 0L) {
  cat("Figure 2A preview:", fig2_panel_a_preview(), "\n")
  missing_stats <- FIG2_STAT_FILES[!file.exists(file.path(p("results", "stats"), paste0(FIG2_STAT_FILES, ".csv")))]
  if (length(missing_stats)) {
    message("Figure 2 skipped (panels B and C): results/stats/", paste(missing_stats, collapse = ".csv, "), ".csv not found")
  } else {
    paths <- save_fig(fig2(), "Figure2", height_mm = FIG2_HEIGHT_MM)
    cat("Figure 2 written:", paste(basename(paths), collapse = ", "), "\n")
  }
}
