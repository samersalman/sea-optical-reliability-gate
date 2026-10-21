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
    annotate("text", x = x, y = Y(y), label = label, size = pt / .pt, fontface = face, colour = colour, hjust = hjust,
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

# Panel A: top 78 mm of the canvas
fig1_panel_a <- function(coh, H) {
  L <- fig1_layers(H)
  Y <- L$Y; txt <- L$txt; box <- L$box; arrow_to <- L$arrow_to
  cc <- coh$counts
  g <- setNames(split(coh$groups, coh$groups$group), coh$groups$group)
  n <- function(x) fmt_num(x, 0)
  x1 <- c(8, 40); x2 <- c(48, 104); x3 <- c(112, 178)
  yc1 <- c(11, 30); yc2 <- c(38, 54); ys <- c(59, 75)
  chip <- function(d, y) annotate("rect", xmin = x2[1] + 2, xmax = x2[1] + 5, ymin = Y(y + 1.5), ymax = Y(y - 1.5),
                                  fill = DOMAIN_COLOURS[[d]], colour = NA)
  dom_row <- function(d, y) list(chip(d, y), txt(x2[1] + 6.5, y, sprintf("%s: %s patients, %s frames", domain_size_label(d),
                                 n(g[[d]]$n_patients), n(g[[d]]$n_frames)), hjust = 0))
  hdr <- function(x, label) list(txt(x[1], 4.2, label, pt = 7, face = "bold", colour = "grey35", hjust = 0),
                                 L$line(x[1], 7, x[2], 7, colour = "grey60", lw = 0.25))
  fold_txt <- sprintf("%s-fold patient-grouped cross-fitting", CFG$segmenters$cv_folds)

  list(
    txt(0.5, 2.2, "A", pt = 10, face = "bold", hjust = 0, vjust = 0.5),
    hdr(x1, "SEA DATASET"), hdr(x2, "COHORT AND DOMAINS"), hdr(x3, "ROLE IN THIS STUDY"),
    # column 1: sources
    box(x1[1], yc1[1], x1[2], yc2[2]), box(x1[1], ys[1], x1[2], ys[2]),
    txt(mean(x1), 29, "Annotated frames", pt = 8, face = "bold", vjust = 1),
    txt(mean(x1), 34, sprintf("%s patients\n%s frames", n(cc$n_patients_labeled), n(cc$n_frames_labeled)), vjust = 1),
    txt(mean(x1), 61.4, "Unannotated frames", pt = 8, face = "bold", vjust = 1),
    txt(mean(x1), 66.6, sprintf("%s patients\n%s frames", n(cc$n_patients_stream), n(cc$n_frames_stream)), vjust = 1),
    # column 2: cohorts
    box(x2[1], yc1[1], x2[2], yc1[2]), box(x2[1], yc2[1], x2[2], yc2[2]), box(x2[1], ys[1], x2[2], ys[2]),
    txt(x2[1] + 2, 14, sprintf("%s: %s patients, %s frames", domain_role("D1"), n(cc$n_patients_c1), n(cc$n_frames_c1)),
        face = "bold", hjust = 0),
    dom_row("D1", 19.5), dom_row("D2", 24.5),
    txt(x2[1] + 2, 41, sprintf("%s: %s patients, %s frames", domain_role("D3"), n(cc$n_patients_c2), n(cc$n_frames_c2)),
        face = "bold", hjust = 0),
    dom_row("D3", 46.5),
    txt(x2[1] + 2, 63.5, sprintf("%s videos, no masks", sub(" ", "-", domain_role("D1"), fixed = TRUE)), face = "bold", hjust = 0),
    chip("D1", 69), txt(x2[1] + 6.5, 69, sprintf("%s, one frame per second", domain_size_label("D1")), hjust = 0),
    # column 3: roles
    box(x3[1], yc1[1], x3[2], yc1[2]), box(x3[1], yc2[1], x3[2], yc2[2]), box(x3[1], ys[1], x3[2], ys[2]),
    txt(x3[1] + 2, 14, fold_txt, face = "bold", hjust = 0),
    txt(x3[1] + 2, 19.2, "Out-of-fold masks from two segmenters;\noptical gate fitted on these, then locked", hjust = 0, vjust = 0.5),
    txt(x3[1] + 2, 26.6, sprintf("Analyzed: %s frames (%s excluded: no field of view)", n(cc$n_frames_c1_analysed), n(cc$n_excl_fov_c1)),
        pt = 7, colour = "grey30", hjust = 0),
    txt(x3[1] + 2, 41, "Locked external test", face = "bold", hjust = 0),
    # the lock covers the gate objects and the frozen settings, not the segmenter weights, so the box says only that
    txt(x3[1] + 2, 46.4, "Locked gate; no external-center outcome\nanalyzed before the lock", hjust = 0),
    txt(x3[1] + 2, 52, sprintf("Analyzed: %s frames", n(cc$n_frames_c2_analysed)), pt = 7, colour = "grey30", hjust = 0),
    txt(x3[1] + 2, 63.5, "Deployment stream", face = "bold", hjust = 0),
    txt(x3[1] + 2, 67.7, "Locked gate; flag rate only", hjust = 0),
    txt(x3[1] + 2, 71.6, sprintf("Analyzed: %s frames (%s excluded: no field of view)", n(cc$n_frames_stream_analysed), n(cc$n_excl_fov_stream)),
        pt = 7, colour = "grey30", hjust = 0),
    # arrows
    arrow_to(x1[2], 20.5, x2[1], 20.5), arrow_to(x1[2], 45, x2[1], 45), arrow_to(x1[2], 67, x2[1], 67),
    arrow_to(x2[2], 20.5, x3[1], 20.5), arrow_to(x2[2], 45, x3[1], 45), arrow_to(x2[2], 67, x3[1], 67),
    arrow_to(x3[1] + 10, yc1[2], x3[1] + 10, yc2[1], lty = "dashed"),
    txt(x3[1] + 13, 34, "Gate locked before external-center\noutcomes were analyzed", pt = 7, face = "italic", colour = "grey30", hjust = 0)
  )
}

# Panel B: tiles
fig1_panel_b <- function(tiles, coh, H, y0) {
  L <- fig1_layers(H)
  Y <- L$Y; txt <- L$txt
  g <- setNames(split(coh$groups, coh$groups$group), coh$groups$group)
  tile_mm <- 48; gap <- 10; x_start <- 12
  doms <- names(DOMAIN_COLOURS)
  y_tile <- c(Normal = y0 + 9, Difficult = y0 + 9 + tile_mm + 13)
  x_tile <- setNames(x_start + (seq_along(doms) - 1) * (tile_mm + gap), doms)
  size_lab <- vapply(doms, domain_size_label, "")

  layers <- list(txt(0.5, y0 + 1, "B", pt = 10, face = "bold", hjust = 0, vjust = 0.5))
  for (d in doms) {
    col <- DOMAIN_COLOURS[[d]]
    centre <- domain_role(d)
    layers <- c(layers, list(
      txt(x_tile[[d]], y0 + 1, sprintf("%s  %s, %s", d, centre, size_lab[[d]]), pt = 8, face = "bold", colour = col, hjust = 0, vjust = 0.5),
      txt(x_tile[[d]], y0 + 5, sprintf("%s patients, %s frames", fmt_num(g[[d]]$n_patients, 0), fmt_num(g[[d]]$n_frames, 0)),
          pt = 7, colour = "grey30", hjust = 0, vjust = 0.5)))
  }
  for (sc in names(y_tile)) {
    layers <- c(layers, list(txt(6.5, y_tile[[sc]] + tile_mm / 2, sc, pt = 8, face = "bold", angle = 90)))
  }

  circle_pts <- function(cx, cy, r, n = 181) {
    th <- seq(0, 2 * pi, length.out = n)
    data.frame(x = cx + r * cos(th), y = cy + r * sin(th))
  }
  for (i in seq_len(nrow(tiles))) {
    t <- tiles[i, ]
    d <- sub("_.*", "", t$cell); sc <- sub(".*_", "", t$cell)
    s <- tile_mm / t$tile_px                     # tile pixel -> mm
    xo <- x_tile[[d]]; yo <- y_tile[[sc]]
    img <- png::readPNG(t$png)
    cont <- utils::read.csv(t$contour)
    # the helper clips the outline to the tile square (an instrument can extend past the inscribed-circle crop)
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
          sprintf("Luminance %.2f   Sharpness %s", t$lum_mean, sci_label(t$sharp_tenengrad)), pt = 7)
    ))
  }
  # key: line 1 explains the overlays, line 2 how each frame was chosen
  yk <- y_tile[["Difficult"]] + tile_mm + 9.5
  pick <- sprintf("Frame near its group's median luminance and sharpness, instrument covering %g to %g%% of the field of view",
                  100 * FIG1_MIN_AREA, 100 * FIG1_MAX_AREA)
  values <- "Printed values: mean luminance (Rec. 709 luma, 0 to 1) and Tenengrad sharpness (scientific notation)"
  layers <- c(layers, list(
    annotate("rect", xmin = x_start, xmax = FIG1_W - 4, ymin = Y(yk + 12.2), ymax = Y(yk - 3), fill = "grey15", colour = NA),
    annotate("segment", x = x_start + 3, xend = x_start + 11, y = Y(yk), yend = Y(yk), colour = FIG1_GT_COLOUR, linewidth = 0.5),
    txt(x_start + 13, yk, "Ground-truth instrument contour", pt = 7, colour = "white", hjust = 0),
    annotate("segment", x = x_start + 62, xend = x_start + 70, y = Y(yk), yend = Y(yk), colour = FIG1_FOV_COLOUR, linewidth = 0.4, linetype = "22"),
    txt(x_start + 72, yk, "Field-of-view outline", pt = 7, colour = "white", hjust = 0),
    txt(x_start + 3, yk + 4.6, pick, pt = 7, colour = "white", hjust = 0),
    txt(x_start + 3, yk + 9.2, values, pt = 7, colour = "white", hjust = 0)
  ))
  layers
}
FIG1_FOV_COLOUR <- "#56B4E9"
FIG1_GT_COLOUR <- "#F0E442"

fig1_height <- function(with_b) if (with_b) 224 else 80

fig1 <- function(coh, tiles = NULL) {
  H <- fig1_height(!is.null(tiles))
  layers <- fig1_panel_a(coh, H)
  if (!is.null(tiles)) layers <- c(layers, fig1_panel_b(tiles, coh, H, y0 = 82))
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
