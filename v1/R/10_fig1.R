# Figure 1: A, data flow (counts from compute_cohort()); B, three domains x Normal/Difficult example frames
# with the FOV outline, the ground-truth contour and two optical metric values.
# One ggplot drawn in millimetres (1 data unit = 1 mm) so boxes, text and tiles sit at fixed positions on the
# 180 mm canvas. Panel B reads real images and masks through work/qc/exhibits/fig1b_tiles.py; it is skipped in
# synthetic mode, where the image paths do not exist.
#   Rscript R/10_fig1.R

if (!exists("compute_cohort", mode = "function")) {
  local({
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    r_dir <- if (exists("V1", envir = globalenv())) file.path(get("V1", envir = globalenv()), "R")
             else if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else "R"
    for (f in c("00_setup.R", "01_load.R", "results_lib.R", "02_cohort.R")) source(file.path(r_dir, f), chdir = FALSE)
  }, envir = globalenv())
}

FIG1_W <- CFG$exhibits$width_mm
FIG1_FS <- 1.25
FIG1_CELLS <- expand.grid(scenario = c("Normal", "Difficult"), domain = names(DOMAIN_COLOURS),
                          stringsAsFactors = FALSE)[, c("domain", "scenario")]
FIG1_CANDIDATES_PER_CELL <- 10      # nearest-to-median frames offered to the pixel helper, in rank order
FIG1_MIN_AREA <- 0.04               # chosen frame: instrument covers 4% to 40% of the FOV disk, so it is clearly visible
FIG1_MAX_AREA <- 0.40
FIG1_TILE_PX <- 600                 # 48 mm at 300 dpi needs 567 px
FIG1_EXHIBITS_DIR <- function() p("work", "qc", "exhibits")

# ---- Panel B: deterministic frame choice --------------------------------------------------------
# Per domain x scenario cell: robust distance (median absolute deviation units) of mean luminance and
# log10 sharpness from the cell medians; the nearest frames are the candidates, ties broken by frame_id.
fig1_candidates <- function(ft, n_per_cell = FIG1_CANDIDATES_PER_CELL) {
  ft <- mutate(ft, log_sharp = log10(sharp_tenengrad))
  out <- lapply(seq_len(nrow(FIG1_CELLS)), function(i) {
    d <- ft[ft$domain == FIG1_CELLS$domain[i] & ft$scenario == FIG1_CELLS$scenario[i], ]
    z <- function(x) (x - median(x)) / max(mad(x), 1e-12)
    d$dist <- sqrt(z(d$lum_mean)^2 + z(d$log_sharp)^2)
    d <- d[order(d$dist, d$frame_id), ][seq_len(min(n_per_cell, nrow(d))), ]
    d$cell <- paste(FIG1_CELLS$domain[i], FIG1_CELLS$scenario[i], sep = "_")
    d$rank <- seq_len(nrow(d))
    d
  })
  bind_rows(out)[c("cell", "rank", "frame_id", "domain", "scenario", "image_path", "mask_path", "fov_cx", "fov_cy",
                   "fov_r", "lum_mean", "sharp_tenengrad", "dist")]
}

# Runs the helper unless the candidate file and the helper are unchanged and its outputs exist.
fig1_tiles <- function(cand) {
  dir <- FIG1_EXHIBITS_DIR()
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  cand_path <- file.path(dir, "fig1b_candidates_in.csv")
  utils::write.csv(cand, cand_path, row.names = FALSE)
  stamp <- file.path(dir, "fig1b_candidates_in.md5")
  helper <- file.path(V1, "work", "qc", "exhibits", "fig1b_tiles.py")
  key <- paste(c(unname(tools::md5sum(c(cand_path, helper))), FIG1_TILE_PX, FIG1_MIN_AREA, FIG1_MAX_AREA), collapse = " ")
  chosen_path <- file.path(dir, "fig1b_chosen.csv")
  fresh <- file.exists(stamp) && file.exists(chosen_path) && identical(readLines(stamp, warn = FALSE), key) &&
    all(file.exists(file.path(dir, paste0("tile_", utils::read.csv(chosen_path)$frame_id, ".png"))))
  if (!fresh) {
    py <- file.path(V1, ".venv", "bin", "python")
    res <- system2(py, c("-I", shQuote(file.path(V1, "work", "qc", "exhibits", "fig1b_tiles.py")), shQuote(cand_path),
                         shQuote(dir), "--tile-px", FIG1_TILE_PX, "--min-area", FIG1_MIN_AREA, "--max-area", FIG1_MAX_AREA),
                   stdout = TRUE, stderr = TRUE)
    status <- attr(res, "status")
    if (!is.null(status) && status != 0) stop("fig1b_tiles.py failed:\n", paste(res, collapse = "\n"), call. = FALSE)
    message(paste(res, collapse = "\n"))
    writeLines(key, stamp)
  }
  chosen <- utils::read.csv(chosen_path, stringsAsFactors = FALSE)
  chosen$png <- file.path(dir, paste0("tile_", chosen$frame_id, ".png"))
  chosen$contour <- file.path(dir, paste0("contours_", chosen$frame_id, ".csv"))
  chosen
}

# ---- drawing helpers (millimetres, y measured downward from the top) ------------------------------
fig1_layers <- function(H) {
  Y <- function(y) H - y
  txt <- function(x, y, label, pt = 7.5, face = "plain", colour = "black", hjust = 0.5, vjust = 0.5, angle = 0) {
    annotate("text", x = x, y = Y(y), label = label, size = pt * FIG1_FS / .pt, fontface = face, colour = colour, hjust = hjust,
             vjust = vjust, angle = angle, family = FONT, lineheight = 0.95)
  }
  box <- function(x0, y0, x1, y1, fill = "white", colour = "grey30", lw = 0.35) {
    annotate("rect", xmin = x0, xmax = x1, ymin = Y(y1), ymax = Y(y0), fill = fill, colour = colour, linewidth = lw)
  }
  arrow_to <- function(x, y, xend, yend, lty = "solid", colour = "grey30") {
    annotate("segment", x = x, y = Y(y), xend = xend, yend = Y(yend), colour = colour, linewidth = 0.35, linetype = lty,
             arrow = grid::arrow(length = grid::unit(1.8, "mm"), type = "closed", angle = 20))
  }
  line <- function(x, y, xend, yend, colour = "grey30", lw = 0.35) {
    annotate("segment", x = x, y = Y(y), xend = xend, yend = Y(yend), colour = colour, linewidth = lw)
  }
  list(Y = Y, txt = txt, box = box, arrow_to = arrow_to, line = line)
}

# Panel A (text enlarged 1.25 times via FIG1_FS for legibility; boxes re-laid out, no content changed)
fig1_panel_a <- function(coh, H) {
  L <- fig1_layers(H)
  Y <- L$Y; txt <- L$txt; box <- L$box; arrow_to <- L$arrow_to
  cc <- coh$counts
  g <- setNames(split(coh$groups, coh$groups$group), coh$groups$group)
  n <- function(x) fmt_num(x, 0)
  x1 <- c(4, 40); x2 <- c(46, 112); x3 <- c(118, 178)
  yc1 <- c(12, 46); yc2 <- c(53, 76); ys <- c(83, 105)
  chip <- function(d, y) annotate("rect", xmin = x2[1] + 2, xmax = x2[1] + 5, ymin = Y(y + 1.7), ymax = Y(y - 1.7),
                                  fill = DOMAIN_COLOURS[[d]], colour = NA)
  dom_row <- function(d, y) list(chip(d, y), txt(x2[1] + 7, y, sprintf("%s: %s patients, %s frames", domain_size_label(d),
                                 n(g[[d]]$n_patients), n(g[[d]]$n_frames)), pt = 7.2, hjust = 0))
  hdr <- function(x, label) list(txt(x[1], 4.4, label, pt = 6.6, face = "bold", colour = "grey35", hjust = 0),
                                 L$line(x[1], 7.6, x[2], 7.6, colour = "grey60", lw = 0.25))
  fold_txt <- sprintf("%s-fold patient-grouped\ncross-fitting", CFG$segmenters$cv_folds)
  list(
    txt(0.5, 2.6, "A", pt = 9, face = "bold", hjust = 0, vjust = 0.5),
    hdr(x1, "SEA DATASET"), hdr(x2, "COHORT AND DOMAINS"), hdr(x3, "ROLE IN THIS STUDY"),
    box(x1[1], yc1[1], x1[2], yc2[2]), box(x1[1], ys[1], x1[2], ys[2]),
    txt(mean(x1), 35, "Annotated frames", pt = 7.2, face = "bold", vjust = 1),
    txt(mean(x1), 40.5, sprintf("%s patients\n%s frames", n(cc$n_patients_labeled), n(cc$n_frames_labeled)), pt = 7.2, vjust = 1),
    txt(mean(x1), 89, "Unannotated frames", pt = 7.2, face = "bold", vjust = 1),
    txt(mean(x1), 94.5, sprintf("%s patients\n%s frames", n(cc$n_patients_stream), n(cc$n_frames_stream)), pt = 7.2, vjust = 1),
    box(x2[1], yc1[1], x2[2], yc1[2]), box(x2[1], yc2[1], x2[2], yc2[2]), box(x2[1], ys[1], x2[2], ys[2]),
    txt(x2[1] + 2, 16.5, domain_role("D1"), pt = 7.4, face = "bold", hjust = 0),
    txt(x2[1] + 2, 22, sprintf("%s patients, %s frames", n(cc$n_patients_c1), n(cc$n_frames_c1)), pt = 7.2, hjust = 0),
    dom_row("D1", 29.5), dom_row("D2", 36),
    txt(x2[1] + 2, 58, domain_role("D3"), pt = 7.4, face = "bold", hjust = 0),
    txt(x2[1] + 2, 63.5, sprintf("%s patients, %s frames", n(cc$n_patients_c2), n(cc$n_frames_c2)), pt = 7.2, hjust = 0),
    dom_row("D3", 70),
    txt(x2[1] + 2, 88, sprintf("%s videos, no masks", sub(" ", "-", domain_role("D1"), fixed = TRUE)), pt = 7.4, face = "bold", hjust = 0),
    chip("D1", 96), txt(x2[1] + 7, 96, sprintf("%s, one frame per second", domain_size_label("D1")), pt = 7.2, hjust = 0),
    box(x3[1], yc1[1], x3[2], yc1[2]), box(x3[1], yc2[1], x3[2], yc2[2]), box(x3[1], ys[1], x3[2], ys[2]),
    txt(x3[1] + 2, 18, fold_txt, pt = 7.4, face = "bold", hjust = 0),
    txt(x3[1] + 2, 28.5, "Out-of-fold masks from two\nsegmenters; optical gate\nfitted on these, then locked", pt = 7.2, hjust = 0, vjust = 0.5),
    txt(x3[1] + 2, 40, sprintf("Analyzed: %s frames\n(%s excluded: no field of view)", n(cc$n_frames_c1_analysed), n(cc$n_excl_fov_c1)),
        pt = 6.8, colour = "grey30", hjust = 0),
    txt(x3[1] + 2, 58, "Locked external test", pt = 7.4, face = "bold", hjust = 0),
    txt(x3[1] + 2, 64.5, "Locked gate; no external-center\noutcome analyzed before the lock", pt = 7.2, hjust = 0),
    txt(x3[1] + 2, 72.5, sprintf("Analyzed: %s frames", n(cc$n_frames_c2_analysed)), pt = 6.8, colour = "grey30", hjust = 0),
    txt(x3[1] + 2, 88, "Deployment stream", pt = 7.4, face = "bold", hjust = 0),
    txt(x3[1] + 2, 93, "Locked gate; flag rate only", pt = 7.2, hjust = 0),
    txt(x3[1] + 2, 99.5, sprintf("Analyzed: %s frames\n(%s excluded: no field of view)", n(cc$n_frames_stream_analysed), n(cc$n_excl_fov_stream)),
        pt = 6.8, colour = "grey30", hjust = 0),
    arrow_to(x1[2], 29, x2[1], 29), arrow_to(x1[2], 64.5, x2[1], 64.5), arrow_to(x1[2], 94, x2[1], 94),
    arrow_to(x2[2], 29, x3[1], 29), arrow_to(x2[2], 64.5, x3[1], 64.5), arrow_to(x2[2], 94, x3[1], 94),
    arrow_to(x3[1] + 8, yc1[2], x3[1] + 8, yc2[1], lty = "dashed"),
    txt(x3[1] + 11, 49.5, "Gate locked before external-center\noutcomes were analyzed", pt = 6.6, face = "italic", colour = "grey30", hjust = 0)
  )
}

# Panel B: tiles
fig1_panel_b <- function(tiles, coh, H, y0) {
  L <- fig1_layers(H)
  Y <- L$Y; txt <- L$txt
  g <- setNames(split(coh$groups, coh$groups$group), coh$groups$group)
  tile_mm <- 38; gap <- 17; x_start <- 21
  doms <- names(DOMAIN_COLOURS)
  y_tile <- c(Normal = y0 + 15, Difficult = y0 + 15 + tile_mm + 12)
  x_tile <- setNames(x_start + (seq_along(doms) - 1) * (tile_mm + gap), doms)
  size_lab <- vapply(doms, domain_size_label, "")
  layers <- list(txt(0.5, y0 + 1, "B", pt = 9, face = "bold", hjust = 0, vjust = 0.5))
  for (d in doms) {
    col <- DOMAIN_COLOURS[[d]]
    centre <- domain_role(d)
    layers <- c(layers, list(
      txt(x_tile[[d]], y0 + 1, sprintf("%s  %s", d, centre), pt = 7.4, face = "bold", colour = col, hjust = 0, vjust = 0.5),
      txt(x_tile[[d]], y0 + 5.2, size_lab[[d]], pt = 7.2, face = "bold", colour = col, hjust = 0, vjust = 0.5),
      txt(x_tile[[d]], y0 + 9.6, sprintf("%s patients, %s frames", fmt_num(g[[d]]$n_patients, 0), fmt_num(g[[d]]$n_frames, 0)),
          pt = 6.8, colour = "grey30", hjust = 0, vjust = 0.5)))
  }
  for (sc in names(y_tile)) {
    layers <- c(layers, list(txt(8.5, y_tile[[sc]] + tile_mm / 2, sc, pt = 7.4, face = "bold", angle = 90)))
  }
  circle_pts <- function(cx, cy, r, n = 181) {
    th <- seq(0, 2 * pi, length.out = n)
    data.frame(x = cx + r * cos(th), y = cy + r * sin(th))
  }
  for (i in seq_len(nrow(tiles))) {
    t <- tiles[i, ]
    d <- sub("_.*", "", t$cell); sc <- sub(".*_", "", t$cell)
    s <- tile_mm / t$tile_px
    xo <- x_tile[[d]]; yo <- y_tile[[sc]]
    img <- png::readPNG(t$png)
    cont <- utils::read.csv(t$contour)
    if (any(cont$x < -1e-6 | cont$x > t$tile_px + 1e-6 | cont$y < -1e-6 | cont$y > t$tile_px + 1e-6)) {
      stop("contour of ", t$frame_id, " leaves the tile: rebuild with fig1b_tiles.py", call. = FALSE)
    }
    fov <- circle_pts(t$tile_cx, t$tile_cy, t$tile_r)
    layers <- c(layers, list(
      annotation_raster(img, xmin = xo, xmax = xo + tile_mm, ymin = Y(yo + tile_mm), ymax = Y(yo), interpolate = TRUE),
      annotate("path", x = xo + fov$x * s, y = Y(yo + fov$y * s), colour = FIG1_FOV_COLOUR, linewidth = 0.4, linetype = "22"),
      geom_path(data = data.frame(x = xo + cont$x * s, y = Y(yo + cont$y * s), g = cont$part), aes(x, y, group = g),
                colour = FIG1_GT_COLOUR, linewidth = 0.35, linejoin = "round", inherit.aes = FALSE),
      annotate("rect", xmin = xo, xmax = xo + tile_mm, ymin = Y(yo + tile_mm), ymax = Y(yo), fill = NA,
               colour = DOMAIN_COLOURS[[d]], linewidth = 0.7),
      txt(xo + tile_mm / 2, yo + tile_mm + 3,
          sprintf("Luminance %.2f", t$lum_mean), pt = 7),
      txt(xo + tile_mm / 2, yo + tile_mm + 6.6,
          sprintf("Sharpness %s", sci_label(t$sharp_tenengrad)), pt = 7)
    ))
  }
  yk <- y_tile[["Difficult"]] + tile_mm + 15
  pick <- sprintf("Frame near its group's median luminance and sharpness,\ninstrument covering %g to %g%% of the field of view",
                  100 * FIG1_MIN_AREA, 100 * FIG1_MAX_AREA)
  values <- "Printed values: mean luminance (Rec. 709\nluma, 0 to 1) and Tenengrad sharpness\n(scientific notation)"
  layers <- c(layers, list(
    annotate("rect", xmin = 4, xmax = FIG1_W - 2, ymin = Y(yk + 16.5), ymax = Y(yk - 3.5), fill = "grey15", colour = NA),
    annotate("segment", x = 8, xend = 17, y = Y(yk), yend = Y(yk), colour = FIG1_GT_COLOUR, linewidth = 0.5),
    txt(19, yk, "Ground-truth instrument contour", pt = 7, colour = "white", hjust = 0),
    annotate("segment", x = 82, xend = 91, y = Y(yk), yend = Y(yk), colour = FIG1_FOV_COLOUR, linewidth = 0.4, linetype = "22"),
    txt(93, yk, "Field-of-view outline", pt = 7, colour = "white", hjust = 0),
    txt(8, yk + 8.5, pick, pt = 7, colour = "white", hjust = 0, vjust = 0.5),
    txt(92, yk + 8.5, values, pt = 7, colour = "white", hjust = 0, vjust = 0.5)
  ))
  layers
}
FIG1_FOV_COLOUR <- "#56B4E9"
FIG1_GT_COLOUR <- "#F0E442"

fig1_height <- function(with_b) if (with_b) 246 else 108

fig1 <- function(coh, tiles = NULL) {
  H <- fig1_height(!is.null(tiles))
  layers <- fig1_panel_a(coh, H)
  if (!is.null(tiles)) layers <- c(layers, fig1_panel_b(tiles, coh, H, y0 = 108))
  ggplot() + layers +
    scale_x_continuous(limits = c(0, FIG1_W), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, H), expand = c(0, 0)) +
    coord_fixed(ratio = 1, clip = "off") +
    theme_void(base_family = FONT) +
    theme(plot.margin = margin(0, 0, 0, 0), plot.background = element_rect(fill = "white", colour = NA))
}

# Frame table of analysed labeled frames, the candidate list and the tiles; NULL when images are unavailable.
fig1_build_tiles <- function() {
  if (SYNTH) { message("synthetic mode: Figure 1B skipped (image paths do not exist)"); return(NULL) }
  ft <- load_frame_table(with_predictions = FALSE)
  cand <- fig1_candidates(ft)
  fig1_tiles(cand) |> left_join(distinct(cand, cell, rank, lum_mean, sharp_tenengrad), by = c("cell", "rank"))
}

if (sys.nframe() == 0L) {
  coh <- compute_cohort()
  tiles <- fig1_build_tiles()
  plot <- fig1(coh, tiles)
  paths <- save_fig(plot, "Figure1", height_mm = fig1_height(!is.null(tiles)))
  cat("Figure 1 written:", paste(basename(paths), collapse = ", "), "\n")
  if (!is.null(tiles)) print(as.data.frame(tiles[c("cell", "frame_id", "rank", "gt_area_share", "lum_mean", "sharp_tenengrad")]), row.names = FALSE)
}
