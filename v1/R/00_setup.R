# Sourced by every R script: `source(file.path(<anything above R/>, "00_setup.R"))`.
# Only defines things (paths, config, palette, theme, exhibit helpers); safe to source twice.
#
# Paths   p(...)            path under v1/, redirected in synthetic mode (see below)
#         rp(...)           path under the project root (dataset images and masks)
# Mode    OPTICS_DATA=synthetic  reads derived/* from work/synthetic/ and writes figures/,
#         tables/, results/, work/ and config/analysis_lock.yaml under work/synthetic/out/.
#         config/analysis_plan.yaml, R/ and python/ always resolve to the real files.
# Stats   set_seeds(which)  seeds from analysis_plan.yaml
# Exhibit save_fig(), save_table(), flex_three_line(), theme_optics(), scale_*_domain/method()

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

.optics_find_v1 <- function() {
  env <- Sys.getenv("OPTICS_V1", "")
  if (nzchar(env)) return(normalizePath(env, mustWork = TRUE))
  file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  sourced <- unlist(lapply(sys.frames(), function(fr) fr$ofile))   # when called through source()
  starts <- c(if (length(file_arg)) dirname(normalizePath(file_arg[1], mustWork = FALSE)),
              if (length(sourced)) dirname(normalizePath(sourced[1], mustWork = FALSE)), getwd())
  for (s in starts) {
    d <- s
    repeat {
      if (file.exists(file.path(d, "config", "analysis_plan.yaml"))) return(normalizePath(d))
      if (file.exists(file.path(d, "v1", "config", "analysis_plan.yaml"))) return(normalizePath(file.path(d, "v1")))
      parent <- dirname(d)
      if (parent == d) break
      d <- parent
    }
  }
  stop("Cannot find v1/config/analysis_plan.yaml. Run from inside the project or set OPTICS_V1.")
}

V1 <- .optics_find_v1()
ROOT <- dirname(V1)
CFG <- yaml::read_yaml(file.path(V1, "config", "analysis_plan.yaml"))

SYNTH <- identical(Sys.getenv("OPTICS_DATA", ""), "synthetic")
SYNTH_DIR <- file.path(V1, "work", "synthetic")
if (SYNTH && is.null(getOption("optics.synth_msg"))) {
  options(optics.synth_msg = TRUE)
  message("OPTICS_DATA=synthetic: reading ", SYNTH_DIR, ", writing ", file.path(SYNTH_DIR, "out"))
}

p <- function(...) {
  rel <- file.path(...)
  if (!SYNTH || !nzchar(rel)) return(file.path(V1, rel))
  top <- strsplit(rel, "/", fixed = TRUE)[[1]][1]
  if (top %in% c("R", "python") || rel %in% c("config/analysis_plan.yaml", "config/CONTRACTS.md")) {
    return(file.path(V1, rel))
  }
  rest <- sub("^[^/]+/?", "", rel)
  if (top == "derived") return(if (nzchar(rest)) file.path(SYNTH_DIR, rest) else SYNTH_DIR)
  file.path(SYNTH_DIR, "out", rel)
}

rp <- function(...) file.path(ROOT, ...)

set_seeds <- function(which = "global") {
  seed <- CFG$seeds[[which]]
  if (is.null(seed)) stop("No seed named '", which, "' in analysis_plan.yaml")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  invisible(seed)
}

OPTICAL_METRICS <- unlist(CFG$optical_metrics)

# Short labels for exhibits (full definitions live in Table 2).
METRIC_LABELS <- c(
  lum_mean = "Mean luminance", contrast_rms = "RMS contrast",
  contrast_local = "Local contrast", sharp_tenengrad = "Sharpness (Tenengrad)",
  under_frac = "Under-exposed fraction", sat_frac = "Saturated fraction",
  spec_frac = "Specular fraction", haze_dcp = "Haze (dark channel)",
  red_dom = "Red dominance", illum_slope = "Illumination fall-off",
  lum_entropy = "Luminance entropy"
)
stopifnot(setequal(names(METRIC_LABELS), OPTICAL_METRICS))

# ---- palette -----------------------------------------------------------------
FONT <- "Helvetica"
OKABE_ITO <- unlist(CFG$okabe_ito)
DOMAIN_COLOURS <- vapply(CFG$domains, function(d) d$colour, character(1))
# Exhibits name the two centers by role ("Training center", "External center"), as the text does; the plan's own
# domain labels carry the numbers (Center 1, Center 2), so the printed labels are rebuilt from the domains: section.
CENTER_ROLE <- c(`1` = "Training center", `2` = "External center")
domain_role <- function(d) unname(CENTER_ROLE[as.character(CFG$domains[[d]]$center)])
DOMAIN_LABELS <- vapply(names(CFG$domains), function(d) {
  sprintf("%s, %d \u00d7 %d", domain_role(d), as.integer(CFG$domains[[d]]$size[[1]]), as.integer(CFG$domains[[d]]$size[[2]]))
}, character(1))

# One name per perturbation family in Figure 2B, Figure 2C and Table 2.
PERTURB_FAMILY_LABELS <- c(defocus = "Defocus", haze = "Haze", exposure_under = "Exposure (under)",
                           exposure_over = "Exposure (over)", blood_absorption = "Blood absorption",
                           glare = "Specular glare", noise = "Sensor noise")

# Sharpness is printed as "4.1e-4" in Figure 1B, the Figure 2A axis and Table 2 (digits = mantissa decimals).
sci_label <- function(x, digits = 1) {
  out <- rep(NA_character_, length(x))        # scale breaks can be NA or outside the plotted range
  ok <- is.finite(x) & x > 0
  s <- sprintf("%.*e", digits, x[ok])
  out[ok] <- paste0(sub("e.*$", "", s), "e", as.integer(sub("^.*e", "", s)))
  out
}

# Fixed per method in Figure 3. The three domain hues (blue, vermilion, green) are not reused;
# the gate is black and thick so it stands out, the ensemble hybrid is a darker shade of the ensemble.
METHOD_COLOURS <- c(
  random = "#999999", predicted_area = "#56B4E9", entropy = "#E69F00",
  ens_disagreement = "#CC79A7", optical_gate = "#000000",
  gate_plus_ensemble = "#7B2D6B", oracle = "#4D4D4D"
)
METHOD_LABELS <- c(
  random = "Random", predicted_area = "Predicted area", entropy = "Pixel entropy",
  ens_disagreement = "Ensemble disagreement", optical_gate = "Optical gate",
  gate_plus_ensemble = "Gate + ensemble", oracle = "Oracle"
)
METHOD_LINETYPES <- c(
  random = "dashed", predicted_area = "solid", entropy = "solid",
  ens_disagreement = "solid", optical_gate = "solid", gate_plus_ensemble = "solid",
  oracle = "dotted"
)

# breaks fix the legend order; methods or domains absent from the data are not shown
scale_colour_domain <- function(...) scale_colour_manual(values = DOMAIN_COLOURS, breaks = names(DOMAIN_COLOURS), labels = DOMAIN_LABELS, ...)
scale_fill_domain <- function(...) scale_fill_manual(values = DOMAIN_COLOURS, breaks = names(DOMAIN_COLOURS), labels = DOMAIN_LABELS, ...)
scale_colour_method <- function(...) scale_colour_manual(values = METHOD_COLOURS, breaks = names(METHOD_COLOURS), labels = METHOD_LABELS, ...)
scale_linetype_method <- function(...) scale_linetype_manual(values = METHOD_LINETYPES, breaks = names(METHOD_LINETYPES), labels = METHOD_LABELS, ...)
scale_color_domain <- scale_colour_domain
scale_color_method <- scale_colour_method

# ---- theme -------------------------------------------------------------------
# Sized for a 180 mm wide figure printed at full width: 7 pt text, 8 pt strip and title.
theme_optics <- function(base_size = 7, base_family = FONT) {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      geom = element_geom(family = base_family, fontsize = base_size),
      axis.line = element_line(colour = "black", linewidth = 0.3),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      axis.text = element_text(colour = "black", size = base_size - 0.5),
      axis.title = element_text(size = base_size),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = base_size + 0.5, hjust = 0),
      legend.position = "bottom",
      legend.title = element_text(size = base_size),
      legend.text = element_text(size = base_size - 0.5),
      legend.key.size = unit(3, "mm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(-2, 0, 0, 0),
      plot.title = element_text(face = "bold", size = base_size + 1, hjust = 0),
      plot.tag = element_text(face = "bold", size = base_size + 3),
      plot.margin = margin(3, 4, 3, 3)
    )
}

# ---- figure output -----------------------------------------------------------
.optics_env <- new.env(parent = emptyenv())

# cairo_pdf is the default, but R's cairo module needs XQuartz libraries on macOS and is
# often unloadable; Quartz PDF embeds Helvetica as a TrueType subset.
.open_pdf <- function(path, w_in, h_in) {
  if (is.null(.optics_env$pdf_engine)) {
    ok <- tryCatch({
      suppressWarnings(cairo_pdf(tempfile(fileext = ".pdf"), width = 1, height = 1, family = FONT))
      opened <- names(dev.cur()) != "null device"
      if (opened) dev.off()
      opened
    }, error = function(e) FALSE)
    .optics_env$pdf_engine <- if (ok) "cairo" else if (isTRUE(unname(capabilities("aqua")))) "quartz" else "pdf"
  }
  switch(.optics_env$pdf_engine,
    cairo = cairo_pdf(path, width = w_in, height = h_in, family = FONT, onefile = FALSE),
    quartz = quartz(file = path, type = "pdf", width = w_in, height = h_in, family = FONT),
    pdf = pdf(path, width = w_in, height = h_in, family = FONT, useDingbats = FALSE)
  )
}

save_fig <- function(plot, name, width_mm = CFG$exhibits$width_mm, height_mm,
                     dpi = CFG$exhibits$dpi) {
  if (missing(height_mm)) stop("save_fig needs height_mm")
  dir <- p("figures")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- c(png = file.path(dir, paste0(name, ".png")),
             tif = file.path(dir, paste0(name, ".tif")),
             pdf = file.path(dir, paste0(name, ".pdf")))
  w_px <- round(width_mm / 25.4 * dpi)
  h_px <- round(height_mm / 25.4 * dpi)

  draw <- function(open) {
    open()
    on.exit(dev.off())
    print(plot)
  }
  draw(function() ragg::agg_png(paths[["png"]], width = w_px, height = h_px, units = "px",
                                res = dpi, background = "white"))
  draw(function() ragg::agg_tiff(paths[["tif"]], width = w_px, height = h_px, units = "px",
                                 res = dpi, background = "white", compression = "lzw"))
  draw(function() .open_pdf(paths[["pdf"]], width_mm / 25.4, height_mm / 25.4))
  invisible(paths)
}

# ---- number formatting -------------------------------------------------------
fmt_num <- function(x, digits = 2, big.mark = ",", na = "NA") {
  x <- round(x, digits) + 0   # + 0 turns -0 into 0
  out <- formatC(x, format = "f", digits = digits, big.mark = big.mark)
  out[is.na(x)] <- na
  unname(out)
}

fmt_ci <- function(est, lo, hi, digits = 2, sep = " to ") {
  paste0(fmt_num(est, digits), " (", fmt_num(lo, digits), sep, fmt_num(hi, digits), ")")
}

fmt_p <- function(p, digits = 3) {
  floor_p <- 10^-digits
  out <- ifelse(p < floor_p, paste0("<", fmt_num(floor_p, digits)), fmt_num(p, digits))
  out[is.na(p)] <- "NA"
  unname(out)
}

# T6's results_lib.R defines its own fmt_num/fmt_ci/fmt_p (different CI text). Scripts that source
# both should call these tbl_* names for table cells; they are not redefined elsewhere.
tbl_num <- fmt_num
tbl_ci <- fmt_ci
tbl_p <- fmt_p

# ---- tables ------------------------------------------------------------------
# Three-line table: heavy rule above the header, thin rule under it, heavy rule at the bottom.
# labels: named character vector colname -> header text. bold_rows / indent_rows / rule_after
# are row indices of the body (panel headings, sub-rows, panel breaks).
flex_three_line <- function(df, labels = NULL, align = NULL, font_size = 8,
                            bold_rows = NULL, indent_rows = NULL, rule_after = NULL,
                            footnote = NULL) {
  ft <- flextable::flextable(df)
  if (!is.null(labels)) ft <- flextable::set_header_labels(ft, values = as.list(labels))
  if (!is.null(footnote)) ft <- flextable::add_footer_lines(ft, values = footnote)
  ft <- flextable::border_remove(ft)
  ft <- flextable::font(ft, fontname = FONT, part = "all")
  ft <- flextable::fontsize(ft, size = font_size, part = "all")
  if (!is.null(footnote)) ft <- flextable::fontsize(ft, size = font_size - 1, part = "footer")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::padding(ft, padding.top = 2, padding.bottom = 2, part = "all")
  align <- align %||% c("l", rep("c", ncol(df) - 1))
  ft <- flextable::align(ft, j = seq_len(ncol(df)), part = "all",
                         align = c(l = "left", r = "right", c = "center")[align])
  heavy <- officer::fp_border(color = "black", width = 1.5)
  thin <- officer::fp_border(color = "black", width = 0.75)
  ft <- flextable::hline_top(ft, border = heavy, part = "header")
  ft <- flextable::hline_bottom(ft, border = thin, part = "header")
  ft <- flextable::hline_bottom(ft, border = heavy, part = "body")
  if (!is.null(rule_after)) ft <- flextable::hline(ft, i = rule_after, border = thin, part = "body")
  if (!is.null(bold_rows)) ft <- flextable::bold(ft, i = bold_rows, part = "body")
  if (!is.null(indent_rows)) {
    ft <- flextable::padding(ft, i = indent_rows, j = 1, padding.left = 10, part = "body")
  }
  flextable::autofit(ft)
}

save_table <- function(ft, df, name, landscape = FALSE) {
  dir <- p("tables")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- c(docx = file.path(dir, paste0(name, ".docx")), csv = file.path(dir, paste0(name, ".csv")))
  section <- officer::prop_section(
    page_size = officer::page_size(orient = if (landscape) "landscape" else "portrait"),
    type = "continuous"
  )
  flextable::save_as_docx(ft, path = paths[["docx"]], pr_section = section)
  utils::write.csv(df, paths[["csv"]], row.names = FALSE, na = "", fileEncoding = "UTF-8")
  invisible(paths)
}
