# Tests for T18: Figure 2 panels B and C (11_fig2.R) and Figure 3 (12_fig3.R).
#   Rscript v1/R/tests/test_fig23.R        (re-runs itself with OPTICS_DATA=synthetic)
# Panels are built from the synthetic results/stats/*.csv and compared with those files; stress variants (a threshold
# that is never reached, NA bands, an NA rho cell, 20 videos) are built in memory. Refuses to run on real data so no
# real exhibit can be touched.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
R_DIR <- dirname(.here)
for (f in c("00_setup.R", "01_load.R", "11_fig2.R", "12_fig3.R")) source(file.path(R_DIR, f))

failures <- character()
check <- function(name, expr) {
  res <- tryCatch(if (isTRUE(expr)) "ok" else "FALSE", error = function(e) paste("error:", conditionMessage(e)))
  cat(sprintf("%-4s %s\n", if (res == "ok") "ok" else "FAIL", name))
  if (res != "ok") { cat("     ", res, "\n"); failures <<- c(failures, name) }
}
errs <- function(expr, pattern) {
  msg <- tryCatch({ force(expr); "" }, error = function(e) conditionMessage(e))
  nzchar(msg) && grepl(pattern, msg)
}
# every warning raised while building and drawing the object(s)
warnings_of <- function(expr) {
  w <- character()
  tmp <- tempfile(fileext = ".png")
  ragg::agg_png(tmp, width = 900, height = 900)
  on.exit({ dev.off(); unlink(tmp) })
  withCallingHandlers({ obj <- force(expr); print(obj) },
                      warning = function(cnd) { w <<- c(w, conditionMessage(cnd)); invokeRestart("muffleWarning") })
  w
}
layer_of <- function(pl, geom) pl$layers[[which(vapply(pl$layers, function(l) class(l$geom)[1], "") == geom)[1]]]
# layers without their own data take the plot's, so count rows from the built plot
built_rows <- function(pl, geom, nth = 1L) {
  i <- which(vapply(pl$layers, function(l) class(l$geom)[1], "") == geom)[nth]
  nrow(ggplot_build(pl)$data[[i]])
}
png_dim <- function(path) {
  b <- readBin(path, "raw", 24)
  c(readBin(b[17:20], "integer", size = 4, endian = "big"), readBin(b[21:24], "integer", size = 4, endian = "big"))
}
pdf_width_mm <- function(path) {
  b <- readBin(path, "raw", file.size(path))
  txt <- rawToChar(replace(b, as.integer(b) == 0 | as.integer(b) > 127, as.raw(32)))
  box <- as.numeric(strsplit(gsub("/MediaBox *\\[|\\]", "", regmatches(txt, regexpr("/MediaBox *\\[[^]]+\\]", txt))), " +")[[1]])
  (box[3] - box[1]) / 72 * 25.4
}
EXPECTED_PX <- round(CFG$exhibits$width_mm / 25.4 * CFG$exhibits$dpi)

dr <- read_stat_csv("perturb_doseresponse"); thr <- read_stat_csv("perturb_threshold"); rho <- read_stat_csv("perturb_rho")

# ---- Figure 2B -------------------------------------------------------------------------------------------------
cat("-- Figure 2B\n")
pb <- fig2_panel_b(dr, thr)
check("2B is a ggplot with one facet per family, in the planned order, labelled from FIG2_FAMILY_LABELS",
      inherits(pb, "ggplot") && identical(as.character(ggplot_build(pb)$layout$layout$family), unname(FIG2_FAMILY_LABELS)) &&
        nrow(ggplot_build(pb)$layout$layout) == 7L)
check("2B draws every dose-response row: both models x 7 families x severities 0 to 5",
      built_rows(pb, "GeomLine") == nrow(dr) && nrow(dr) == 2 * 7 * 6 &&
        all(table(dr$model, dr$family) == 6L))
check("2B bands come from median_lo and median_hi, one point per dose-response row, ticks are the only segment layer",
      identical(layer_of(pb, "GeomRibbon")$mapping$ymin |> rlang::as_label(), "median_lo") &&
        built_rows(pb, "GeomPoint", 1L) == nrow(dr) &&
        sum(vapply(pb$layers, function(l) class(l$geom)[1] == "GeomPoint", NA)) == 1L &&
        sum(vapply(pb$layers, function(l) class(l$geom)[1] == "GeomSegment", NA)) == 1L)
cross_layer <- layer_of(pb, "GeomSegment")
check("2B has one crossing tick per model x family that crossed, at that row's severity_interp and the threshold height",
      nrow(cross_layer$data) == sum(!thr$never_reached) &&
        isTRUE(all.equal(sort(cross_layer$data$severity_interp), sort(thr$severity_interp[!thr$never_reached]))) &&
        all(cross_layer$data$y == unique(thr$threshold_iou)))
check("2B: no warnings while building and drawing", length(warnings_of(fig2_panel_b(dr, thr))) == 0L)
n_layers <- function(pl, geom) sum(vapply(pl$layers, function(l) class(l$geom)[1] == geom, NA))
y_lim <- function(pl) pl$scales$get_scales("y")$limits
plotted_y <- function(pl) { b <- ggplot_build(pl)$data; unlist(lapply(b, function(x) unlist(x[intersect(names(x), c("y", "ymin", "ymax"))]))) }
level0 <- unique(thr$threshold_iou)
check("2B some rows cross (synthetic): rule line and tick drawn, no 'not reached' footer, one shared y scale from the data (here 0 to 1, since the curves collapse)",
      !all(thr$never_reached) && is.null(pb$labels$caption) && n_layers(pb, "GeomHline") == 1L && n_layers(pb, "GeomText") == 1L &&
        identical(y_lim(pb), c(0, 1)) && identical(y_lim(pb), fig2b_ylim(dr, level0)$limits) &&
        identical(ggplot_build(pb)$layout$facet_params$free, list(x = FALSE, y = FALSE)) &&
        all(layer_of(pb, "GeomHline")$data$family %in% levels(pb$data$family)) &&
        nrow(layer_of(pb, "GeomHline")$data) == 7L && level0 %in% ggplot_build(pb)$layout$panel_scales_y[[1]]$get_breaks())

thr_s <- thr
thr_s$never_reached[thr_s$family %in% c("glare", "noise")] <- TRUE
thr_s$severity_interp[thr_s$never_reached] <- NA
dr_s <- dr
dr_s$median_lo[dr_s$family == "haze" & dr_s$severity == 5] <- NA
dr_s$median_hi[dr_s$family == "haze" & dr_s$severity == 5] <- NA
pb_s <- fig2_panel_b(dr_s, thr_s)
cross_s <- layer_of(pb_s, "GeomSegment")$data
check("2B stress: never_reached rows get no tick (glare and noise absent), NA bands are skipped without warning",
      nrow(cross_s) == sum(!thr_s$never_reached) && !any(cross_s$family %in% c("Specular glare", "Sensor noise")) &&
        length(warnings_of(fig2_panel_b(dr_s, thr_s))) == 0L)
# the reference line is perturbation.rule_first_below_iou (collapse rule), not the gate's failure threshold
txt_b <- vapply(pb$layers[vapply(pb$layers, function(l) class(l$geom)[1] == "GeomText", NA)], function(l) paste(l$data$lab, collapse = " "), "")
lvl <- format(unique(thr$threshold_iou))
lw_breaks <- function(pl) as.character(ggplot_build(pl)$plot$scales$get_scales("linewidth")$get_breaks())
check("2B labels the line 'Collapse rule' (not 'Failure IoU') and the legend 'Median first below IoU <level>'; the level comes from perturb_threshold.csv",
      length(txt_b) == 1L && grepl(paste0("^Collapse rule\\nmedian IoU\\n", lvl, "$"), txt_b) && !any(grepl("ailure", txt_b)) &&
        identical(lw_breaks(pb), paste0("Median first below IoU ", lvl)))
thr_mod <- thr; thr_mod$threshold_iou <- 0.6
check("2B: a different threshold_iou in the CSV changes the printed level and the legend text (nothing typed)",
      grepl("0.6", paste(layer_of(fig2_panel_b(dr, thr_mod), "GeomText")$data$lab), fixed = TRUE) &&
        identical(lw_breaks(fig2_panel_b(dr, thr_mod)), "Median first below IoU 0.6"))
# real-data shape: every model x family stays far above the rule (median IoU 0.88 to 0.98); built by squeezing the synthetic
# curves into 0.85 to 0.98 so the lowest band edge is known
squeeze <- function(x) 0.85 + 0.13 * x
dr_nr <- dr
for (cc in c("median_iou", "median_lo", "median_hi")) dr_nr[[cc]] <- squeeze(dr[[cc]])
thr_nr <- thr; thr_nr$never_reached <- TRUE; thr_nr$first_severity_below <- NA; thr_nr$severity_interp <- NA
pb_nr <- fig2_panel_b(dr_nr, thr_nr)
yl_nr <- fig2b_ylim(dr_nr, NULL)
all_vals <- unlist(dr_nr[c("median_iou", "median_lo", "median_hi")])
check("2B never reached in any row: no rule line, no label, no tick, no linewidth scale (the old 'No shared levels' warning), and no warning at all",
      n_layers(pb_nr, "GeomHline") == 0L && n_layers(pb_nr, "GeomText") == 0L && n_layers(pb_nr, "GeomSegment") == 0L &&
        !pb_nr$scales$has_scale("linewidth") && length(warnings_of(fig2_panel_b(dr_nr, thr_nr))) == 0L)
axis_note <- function(lo) paste0("Vertical axis starts at ", formatC(lo, format = "f", digits = 2))
check("2B never reached: the footer is built from perturb_threshold.csv ('Collapse rule (median IoU <threshold>) not reached in any family') and, on a second line, says where the vertical axis starts (read from the plotted data); both follow the data",
      identical(pb_nr$labels$caption, paste0("Collapse rule (median IoU ", format(unique(thr$threshold_iou)), ") not reached in any family\n", axis_note(yl_nr$limits[1]))) &&
        grepl("median IoU 0.6) not reached", fig2_panel_b(dr_nr, transform(thr_nr, threshold_iou = 0.6))$labels$caption, fixed = TRUE) &&
        grepl(axis_note(0.8), fig2_panel_b(transform(dr_nr, median_lo = 0.8447), thr_nr)$labels$caption, fixed = TRUE) &&
        is.null(pb$labels$caption))
check("2B never reached: shared y range starts above zero at a clean 0.05 tick at or below the lowest median or band edge, ends at 1, clips nothing",
      identical(y_lim(pb_nr), yl_nr$limits) && yl_nr$limits[1] > 0 && yl_nr$limits[1] <= min(all_vals) && min(all_vals) - yl_nr$limits[1] < 0.05 &&
        abs(yl_nr$limits[1] / 0.05 - round(yl_nr$limits[1] / 0.05)) < 1e-9 && yl_nr$limits[2] == 1 &&
        all(plotted_y(pb_nr) >= yl_nr$limits[1] & plotted_y(pb_nr) <= yl_nr$limits[2], na.rm = TRUE) &&
        identical(as.character(ggplot_build(pb_nr)$layout$layout$family), unname(FIG2_FAMILY_LABELS)) &&
        built_rows(pb_nr, "GeomLine") == nrow(dr_nr) && built_rows(pb_nr, "GeomRibbon") == nrow(dr_nr))
check("2B never reached: the axis breaks are the clean ticks from the lower bound to 1 (no 0.75 tick, since no line is drawn)",
      isTRUE(all.equal(ggplot_build(pb_nr)$layout$panel_scales_y[[1]]$get_breaks(), seq(yl_nr$limits[1], 1, 0.05))))
# some rows cross, but only in one family; the rule appears only in the facets that cross it or whose bands span it
dr_m <- dr_nr; thr_m <- thr_nr
hz <- dr_m$model == "unet" & dr_m$family == "haze" & dr_m$severity >= 4
dr_m$median_iou[hz] <- c(0.72, 0.55); dr_m$median_lo[hz] <- c(0.65, NA); dr_m$median_hi[hz] <- c(0.80, NA)
nz <- dr_m$model == "segformer" & dr_m$family == "noise" & dr_m$severity == 5
dr_m$median_lo[nz] <- 0.74                          # band touches the rule, median does not
ix <- thr_m$model == "unet" & thr_m$family == "haze"
thr_m$never_reached[ix] <- FALSE; thr_m$first_severity_below[ix] <- 4; thr_m$severity_interp[ix] <- 3.9
pb_m <- fig2_panel_b(dr_m, thr_m)
yl_m <- fig2b_ylim(dr_m, unique(thr_m$threshold_iou))
check("2B mixed: the rule line is drawn only in the crossing family and in the family whose band spans it; one tick; label in the first of those facets; no 'not reached' footer, only the axis note",
      identical(sort(as.character(layer_of(pb_m, "GeomHline")$data$family)), sort(unname(FIG2_FAMILY_LABELS[c("haze", "noise")]))) &&
        nrow(layer_of(pb_m, "GeomSegment")$data) == 1L && identical(pb_m$labels$caption, axis_note(fig2b_ylim(dr_m, unique(thr_m$threshold_iou))$limits[1])) &&
        as.character(layer_of(pb_m, "GeomText")$data$family) == FIG2_FAMILY_LABELS[["haze"]] && length(warnings_of(fig2_panel_b(dr_m, thr_m))) == 0L)
check("2B mixed: the shared y range now reaches below the lowest median (0.55) at a clean tick, includes the rule level as a tick, still ends at 1",
      identical(y_lim(pb_m), yl_m$limits) && yl_m$limits[1] <= 0.55 && yl_m$limits[2] == 1 && 0.75 %in% ggplot_build(pb_m)$layout$panel_scales_y[[1]]$get_breaks() &&
        all(plotted_y(pb_m) >= yl_m$limits[1] & plotted_y(pb_m) <= 1, na.rm = TRUE))
check("2B y range is the data range, not a constant: lower bound is monotone in the data and always rounded down to a tick",
      fig2b_ylim(dr_nr)$limits[1] > fig2b_ylim(dr_m, 0.75)$limits[1] && fig2b_ylim(dr_m, 0.75)$limits[1] > fig2b_ylim(dr)$limits[1] &&
        identical(fig2b_ylim(transform(dr_nr, median_lo = 0.8447))$limits, c(0.8, 1)))
check("2B refuses an unknown family and a missing column",
      errs(fig2_panel_b(transform(dr, family = ifelse(family == "haze", "smoke", family)), thr), "Unknown perturbation family") &&
        errs(fig2_panel_b(dr[setdiff(names(dr), "median_lo")], thr), "missing columns"))

check("2B and 2C use the family names of Table 2 (PERTURB_FAMILY_LABELS: Defocus, Haze, Exposure (under), Exposure (over), Blood absorption, Specular glare, Sensor noise)",
      identical(unname(FIG2_FAMILY_LABELS), c("Defocus", "Haze", "Exposure (under)", "Exposure (over)", "Blood absorption", "Specular glare", "Sensor noise")) &&
        identical(FIG2_FAMILY_LABELS, PERTURB_FAMILY_LABELS))   # test_tables.R checks Table 2 against the same vector
check("text sizes: heat-map cell values 7 pt, footers 7 pt, headers stay 6 pt because 7 pt 'absorption' is wider than a cell",
      FIG2_CELL_TEXT_PT == 7 && FIG2_NOTE_PT == 7 && FIG2C_HEADER_PT == 6 && FIG2_RHO_WHITE_BELOW == -0.75)

# ---- Figure 2C -------------------------------------------------------------------------------------------------
cat("-- Figure 2C\n")
pc <- fig2_panel_c(rho)
tiles <- pc$layers[vapply(pc$layers, function(l) class(l$geom)[1] == "GeomTile", NA)]
txt <- pc$data
check("2C is a ggplot with 11 metric rows x 7 family columns (77 cells), metrics in config order from the top",
      inherits(pc, "ggplot") && nrow(pc$data) == 77L && nlevels(pc$data$family) == 7L &&
        identical(rev(levels(pc$data$metric)), unname(METRIC_LABELS[OPTICAL_METRICS])))
check("2C outlines exactly the expected_pair cells (12) and nothing else",
      nrow(tiles[[2]]$data) == sum(rho$expected_pair) && sum(rho$expected_pair) == 12L)
check("2C diverging fill is centred on 0 with limits -1 to 1",
      identical(pc$scales$get_scales("fill")$limits, c(-1, 1)) &&
        isTRUE(all.equal(pc$scales$get_scales("fill")$rescaler(0.5, from = c(-1, 1)), 0.75)))
check("2C prints rho at two decimals from the CSV and marks only the known-limitation cell with a dagger",
      sum(grepl("†", txt$cell_text)) == 1L && sum(rho$known_limitation) == 1L &&
        txt$cell_text[txt$family == "Sensor noise" & txt$metric == METRIC_LABELS[["sharp_tenengrad"]]] ==
          paste0(fmt_num(rho$rho[rho$family == "noise" & rho$metric == "sharp_tenengrad"], 2), "†") &&
        !any(grepl("-0\\.00", txt$cell_text)))
check("2C text is white only on the darkest purple (rho below -0.75); the gold side and pale cells print black",
      all(pc$data$cell_ink[pc$data$rho < -0.75] == "white") && all(pc$data$cell_ink[pc$data$rho >= -0.75] == "black") &&
        any(pc$data$cell_ink == "white") && all(pc$data$cell_ink[pc$data$rho > 0] == "black"))
lab_of <- function(hex) grDevices::convertColor(t(grDevices::col2rgb(hex) / 255), "sRGB", "Lab")
delta_e <- function(a, b) sqrt(sum((lab_of(a) - lab_of(b))^2))
check("2C positive hue is far (Lab distance above 30) from the domain D2 vermilion used in panel A and from the other domain colours",
      delta_e(FIG2_RHO_COLOURS[["pos"]], DOMAIN_COLOURS[["D2"]]) > 30 &&
        all(vapply(DOMAIN_COLOURS, function(h) delta_e(FIG2_RHO_COLOURS[["pos"]], h), 0) > 30))
check("2C legend title is plain ASCII 'Spearman rho' (no rho glyph that falls back to another font); the dagger is explained in the figure",
      identical(pc$scales$get_scales("fill")$name, "Spearman rho with severity") && grepl("Known limitation", pc$labels$caption))
rho_s <- rho; rho_s$rho[3] <- NA
check("2C stress: an NA rho cell draws no text and no warning",
      length(warnings_of(fig2_panel_c(rho_s))) == 0L && fig2_panel_c(rho_s)$data$cell_text[3] == "")
check("2C: no warnings while building and drawing", length(warnings_of(fig2_panel_c(rho))) == 0L)
check("2C refuses duplicated cells", errs(fig2_panel_c(rbind(rho, rho[1, ])), "duplicated"))

# ---- Figure 3 --------------------------------------------------------------------------------------------------
cat("-- Figure 3\n")
d <- fig3_data()
pn <- fig3_panels(d)
cv_unet <- read_stat_csv("risk_coverage_curves"); cv_seg <- read_stat_csv("risk_coverage_curves_transfer")
ext <- read_stat_csv("gate_external"); trf <- read_stat_csv("gate_transfer")
check("3A to 3D are ggplot objects", all(vapply(pn, inherits, NA, what = "ggplot")))
check("3 titles: 'Training center, cross-fitted', 'External center, locked gate', 'Transfer to SegFormer, external center', 'Stream flags by video'; no 'Center 1' or 'Center 2' anywhere",
      identical(unname(vapply(pn, function(pl) pl$labels$title, "")), c("Training center, cross-fitted", "External center, locked gate",
                                                                      "Transfer to SegFormer, external center", "Stream flags by video")) &&
        !any(grepl("Center [12]", c(vapply(pn, function(pl) pl$labels$title, ""), pn$d$labels$subtitle))) &&
        grepl("^[0-9.]+% of stream frames flagged \\(training-center reference [0-9.]+%\\)$", pn$d$labels$subtitle))
check("3 text sizes: coverage labels and the per-video shares 7 pt, panel D row labels and the legends of A to D 7 pt",
      FIG3_LABEL_PT == 7 && pn$d$theme$axis.text.y$size == 7 && pn$d$theme$legend.text$size == 7 &&
        all(vapply(pn[c("a", "b", "c")], function(pl) pl$theme$legend.text$size == 7 && pl$theme$legend.title$size == 7, NA)))
check("3 labels: 'Optical gate' is the method name in every legend", all(vapply(pn[c("a", "b", "c")], function(pl) any(startsWith(pl$scales$get_scales("colour")$labels, "Optical gate ")), NA)))
for (nm in c("a", "b", "c")) {
  pl <- pn[[nm]]
  line <- pl$data
  band <- layer_of(pl, "GeomRibbon")$data
  check(sprintf("3%s: seven method curves of 100 coverage points; bands only for gate, entropy and ensemble disagreement", toupper(nm)),
        identical(sort(unique(as.character(line$method))), sort(FIG3_METHODS)) && all(table(line$method) == 100L) &&
          identical(sort(unique(as.character(band$method))), sort(FIG3_BAND_METHODS)))
}
src <- list(a = cv_unet[cv_unet$partition == "internal", ], b = cv_unet[cv_unet$partition == "external", ],
            c = cv_seg[cv_seg$partition == "external", ])
check("3A to 3C plot the CSV curves: A internal U-Net, B external U-Net, C external transfer (SegFormer outcomes)",
      all(vapply(names(src), function(nm) {
        l <- pn[[nm]]$data
        m <- merge(l, src[[nm]], by.x = c("method", "coverage"), by.y = c("method", "coverage"))
        nrow(m) == nrow(src[[nm]]) && isTRUE(all.equal(m$selective_risk.x, m$selective_risk.y))
      }, NA)) && all(src$c$model == "segformer") && all(src$a$model == "unet"))
leg <- function(pl) pl$scales$get_scales("colour")$labels
want <- function(g) paste0(FIG3_METHOD_LABELS[FIG3_METHODS], "  ", fmt_num(g$aurc[match(FIG3_METHODS, g$method)], 3))
check("legend labels carry the AURC read from gate_internal, gate_external and gate_transfer (not typed)",
      identical(unname(leg(pn$a)), want(read_stat_csv("gate_internal"))) &&
        identical(unname(leg(pn$b)), want(ext)) &&
        identical(unname(leg(pn$c)), want(trf[trf$partition == "external", ])))
check("the entropy baseline is labelled 'Pixel entropy' in every legend, never 'softmax'",
      all(vapply(pn[c("a", "b", "c")], function(pl) any(startsWith(pl$scales$get_scales("colour")$labels, "Pixel entropy ")) &&
                   !any(grepl("softmax", pl$scales$get_scales("colour")$labels, ignore.case = TRUE)), NA)))
yr <- lapply(pn[c("a", "b", "c")], function(pl) ggplot_build(pl)$layout$panel_params[[1]]$y.range)
all_cv <- rbind(cv_unet, cv_seg)
top_curves <- max(all_cv$selective_risk)
check("3A to 3C share one y range; its top is the highest curve point over the three panels, not a band (bands are clipped)",
      identical(yr$a, yr$b) && identical(yr$b, yr$c) && yr$a[2] >= top_curves && yr$a[2] < 1.1 * top_curves &&
        isTRUE(all.equal(fig3_ytop(all_cv), top_curves)) && max(all_cv$hi, na.rm = TRUE) > top_curves)
check("the 90% and 80% coverage guides come from the plan", identical(FIG3_COVERAGE_LEVELS, c(0.9, 0.8)) &&
        isTRUE(all.equal(sort(layer_of(pn$a, "GeomVline")$data$xintercept), sort(FIG3_COVERAGE_LEVELS))))
check("method colours and line types are the fixed METHOD_COLOURS and METHOD_LINETYPES in every panel",
      all(vapply(pn[c("a", "b", "c")], function(pl) {
        identical(unname(pl$scales$get_scales("colour")$palette(7)), unname(METHOD_COLOURS[FIG3_METHODS])) &&
          identical(unname(pl$scales$get_scales("linetype")$palette(7)), unname(METHOD_LINETYPES[FIG3_METHODS]))
      }, NA)))

flags <- d$flags; summ <- d$summ
vid <- summ[summ$scope == "video" & summ$statistic == "flagged_share", ]
dl <- pn$d$layers[vapply(pn$d$layers, function(l) class(l$geom)[1] == "GeomSegment", NA)]
check("3D: one row per video, sorted by flagged share (highest on top), one tick per flagged frame",
      identical(unname(pn$d$scales$get_scales("y")$labels), vid$video_patient[order(-vid$est, vid$video_patient)]) &&
        nrow(dl[[1]]$data) == nrow(vid) && nrow(dl[[2]]$data) == sum(flags$flagged))
check("3D: row labels on the right are the per-video shares from stream_summary.csv",
      identical(layer_of(pn$d, "GeomText")$data$lab[order(-layer_of(pn$d, "GeomText")$data$row)],
                paste0(fmt_num(100 * sort(vid$est, decreasing = TRUE), 1), "%")))
check("3D: in-figure key names the grey bar 'Span of analyzed frames' and the tick 'Flagged frame (90% coverage threshold)'; the 'Flagged' header sits above the panel (y = Inf), not on a row",
      identical(as.character(ggplot_build(pn$d)$plot$scales$get_scales("colour")$get_breaks()), c("Span of analyzed frames", "Flagged frame (90% coverage threshold)")) &&
        { hd <- pn$d$layers[vapply(pn$d$layers, function(l) class(l$geom)[1] == "GeomText", NA)][[2]]; identical(hd$aes_params$label, "Flagged") && is.infinite(hd$data$y) })
check("3D: time axis is the file frame index x the plan's frame period, in minutes",
      isTRUE(all.equal(max(dl[[2]]$data$minute), max(flags$frame_idx[flags$flagged]) * CFG$stream$frame_period_s / 60)))
check("3: no warnings while building and drawing the four-panel figure", length(warnings_of(fig3(pn))) == 0L)
check("3: refuses a method that is missing from a curve set and videos that disagree between the two stream files",
      errs(fig3_rc_panel(d$curves[d$curves$method != "oracle", ], d$gate, "internal", "unet", "x"), "lack: oracle") &&
        errs(fig3_panel_d(flags[flags$video_patient != flags$video_patient[1], ], summ), "different videos"))

# stress: 20 videos, an NA band stretch
set.seed(1)
vids <- sprintf("P%d", 69:88)
fl <- do.call(rbind, lapply(vids, function(v) {
  n <- sample(300:900, 1)
  data.frame(video_patient = v, frame_idx = seq_len(n), flagged = runif(n) < runif(1, 0, 0.3))
}))
sm <- data.frame(scope = "video", video_patient = vids, statistic = "flagged_share",
                 est = vapply(vids, function(v) mean(fl$flagged[fl$video_patient == v]), 0), lo = NA, hi = NA, n = NA)
sm <- rbind(summ[summ$scope == "overall", names(sm)], sm)
d20 <- d; d20$flags <- fl; d20$summ <- sm
d20$curves$lo[d20$curves$partition == "external" & d20$curves$model == "unet" & d20$curves$coverage < 0.1] <- NA
check("3 stress: 20 videos and an NA band stretch build and draw without warnings",
      length(warnings_of(fig3(fig3_panels(d20)))) == 0L &&
        length(fig3_panel_d(fl, sm)$scales$get_scales("y")$labels) == 20L)

# ---- files written by the build scripts (subprocess, synthetic root) ---------------------------------------------
cat("-- build scripts\n")
rs <- file.path(R.home("bin"), "Rscript")
run <- function(script) system2(rs, file.path(R_DIR, script), stdout = TRUE, stderr = TRUE, env = "OPTICS_DATA=synthetic")
out <- c(run("11_fig2.R"), run("12_fig3.R"))
check("11_fig2.R and 12_fig3.R run without error output", !any(grepl("^Error|Execution halted", out)))
fdir <- file.path(SYNTH_DIR, "out", "figures")
for (nm in c("Figure2", "Figure3")) {
  paths <- file.path(fdir, paste0(nm, c(".png", ".tif", ".pdf")))
  check(sprintf("%s png, tif and pdf are written, 180 mm wide (%d px at 300 dpi)", nm, EXPECTED_PX),
        all(file.exists(paths)) && png_dim(paths[1])[1] == EXPECTED_PX && abs(pdf_width_mm(paths[3]) - CFG$exhibits$width_mm) < 0.5)
}
check("Figure 2 and 3 heights are the constants of the scripts",
      png_dim(file.path(fdir, "Figure2.png"))[2] == round(FIG2_HEIGHT_MM / 25.4 * 300) &&
        png_dim(file.path(fdir, "Figure3.png"))[2] == round(FIG3_HEIGHT_MM / 25.4 * 300))
if (nzchar(Sys.which("pdffonts"))) {
  fonts_of <- function(nm) { o <- system2("pdffonts", file.path(fdir, paste0(nm, ".pdf")), stdout = TRUE); sub("^[A-Z]+\\+", "", sub(" .*", "", o[-(1:2)])) }
  check("Figure2.pdf embeds Helvetica only (no ArialMT fallback for the rho glyph or the severity tick)",
        length(fonts_of("Figure2")) > 0L && all(grepl("^Helvetica", fonts_of("Figure2"))))
}
check("Figure 2A preview is still written by 11_fig2.R", file.exists(file.path(fdir, "Figure2_panelA_preview.png")))

cat(sprintf("\n%s\n", if (length(failures)) paste("FAILED:", paste(failures, collapse = "; ")) else "all checks passed"))
if (length(failures)) quit(status = 1)
