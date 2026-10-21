# Tests for 00_setup.R, 01_load.R and the synthetic data.
#   OPTICS_DATA=synthetic Rscript v1/R/tests/make_synthetic.R && OPTICS_DATA=synthetic Rscript v1/R/tests/test_setup.R
# The contract below is restated from config/CONTRACTS.md on purpose, independent of LOAD_SPEC in 01_load.R.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
source(file.path(dirname(.here), "00_setup.R"))
source(file.path(dirname(.here), "01_load.R"))

failures <- character()
check <- function(name, expr) {
  res <- tryCatch(if (isTRUE(expr)) "ok" else "FALSE", error = function(e) paste("error:", conditionMessage(e)))
  cat(sprintf("%-4s %s\n", if (res == "ok") "ok" else "FAIL", name))
  if (res != "ok") { cat("     ", res, "\n"); failures <<- c(failures, name) }
}

M11 <- c("lum_mean", "contrast_rms", "contrast_local", "sharp_tenengrad", "under_frac", "sat_frac",
         "spec_frac", "haze_dcp", "red_dom", "illum_slope", "lum_entropy")
types <- function(...) { x <- c(...); structure(unname(x), names = names(x)) }
dbl <- function(nm) setNames(rep("double", length(nm)), nm)
CONTRACT <- list(
  manifest = c(frame_id = "character", partition = "character", patient_id = "character", center = "integer",
               domain = "character", width = "integer", height = "integer", channel = "character",
               region = "character", scenario = "character", instrument = "character", file_format = "character",
               video_idx = "integer", frame_idx = "integer", image_path = "character", mask_path = "character",
               labeled = "logical"),
  fov = c(frame_id = "character", fov_found = "logical", fov_cx = "double", fov_cy = "double", fov_r = "double",
          fov_method = "character", fov_resid = "double", crop_x0 = "integer", crop_y0 = "integer",
          crop_side = "integer"),
  optics = c(frame_id = "character", dbl(M11)),
  folds = c(patient_id = "character", center = "integer", domain = "character", n_frames = "integer",
            role = "character", fold = "integer"),
  predictions = c(frame_id = "character", patient_id = "character", model = "character", source = "character",
                  partition = "character", fold = "integer", iou = "double", dice = "double",
                  pred_area_frac = "double", gt_area_frac = "double", entropy_fov = "double",
                  entropy_roi = "double", ens_dis = "double", dbl(paste0("unet_bg_", M11))),
  stream = c(frame_id = "character", patient_id = "character", model = "character", source = "character",
             partition = "character", fold = "integer", pred_area_frac = "double", entropy_fov = "double",
             entropy_roi = "double", ens_dis = "double", dbl(paste0("unet_bg_", M11)),
             video_patient = "character", t_index = "integer"),
  perturb = c(frame_id = "character", patient_id = "character", domain = "character", scenario = "character",
              instrument = "character", model = "character", family = "character", severity = "integer",
              param = "double", iou = "double", dice = "double", pred_area_frac = "double",
              entropy_fov = "double", dbl(M11))
)
real_before <- list(figures = list.files(p0 <- file.path(V1, "figures")), tables = list.files(file.path(V1, "tables")))

# ---- setup -------------------------------------------------------------------
cat("-- setup\n")
check("synthetic mode on", SYNTH && SYNTH_DIR == file.path(V1, "work", "synthetic"))
check("config metrics equal the 11 in the contract", identical(OPTICAL_METRICS, M11))
check("p() sends derived reads to work/synthetic", p("derived", "manifest.parquet") == file.path(SYNTH_DIR, "manifest.parquet"))
check("p() sends figures/tables/results/lock to work/synthetic/out",
      all(startsWith(c(p("figures", "a.png"), p("tables", "a.csv"), p("results", "results.json"),
                       p("config", "analysis_lock.yaml")), file.path(SYNTH_DIR, "out"))))
check("p() keeps plan, R and python real",
      p("config", "analysis_plan.yaml") == file.path(V1, "config", "analysis_plan.yaml") &&
        p("R", "00_setup.R") == file.path(V1, "R", "00_setup.R"))
real_p <- system2(file.path(R.home("bin"), "Rscript"),
                  c("-e", shQuote(sprintf("source('%s'); cat(p('derived', 'x'), SYNTH)", file.path(V1, "R", "00_setup.R")))),
                  stdout = TRUE, stderr = FALSE, env = "OPTICS_DATA=")
check("without OPTICS_DATA, p() is the real path", identical(real_p, paste(file.path(V1, "derived", "x"), "FALSE")))
check("setup is idempotent", { source(file.path(V1, "R", "00_setup.R")); TRUE })
check("set_seeds is reproducible", { set_seeds(); a <- runif(3); set_seeds(); identical(a, runif(3)) })
check("domain colours come from the yaml",
      identical(names(DOMAIN_COLOURS), c("D1", "D2", "D3")) && all(DOMAIN_COLOURS == c("#0072B2", "#D55E00", "#009E73")))
check("method colours: 7 named, distinct, none equal a domain colour",
      identical(names(METHOD_COLOURS), c("random", "predicted_area", "entropy", "ens_disagreement", "optical_gate",
                                         "gate_plus_ensemble", "oracle")) &&
        !anyDuplicated(METHOD_COLOURS) && !any(toupper(METHOD_COLOURS) %in% toupper(DOMAIN_COLOURS)) &&
        identical(names(METHOD_LINETYPES), names(METHOD_COLOURS)) && identical(names(METHOD_LABELS), names(METHOD_COLOURS)))
check("fmt_num", identical(fmt_num(c(10662, -0.001, NA), c(0)), c("10,662", "0", "NA")) && fmt_num(-0.001, 2) == "0.00" &&
        fmt_num(1234.567, 1) == "1,234.6")
check("fmt_ci", fmt_ci(0.123, 0.081, 0.171) == "0.12 (0.08 to 0.17)" && fmt_ci(-0.5, -0.9, -0.1, 1) == "-0.5 (-0.9 to -0.1)")
check("tbl_* aliases equal fmt_*", identical(tbl_ci, fmt_ci) && identical(tbl_p, fmt_p) && identical(tbl_num, fmt_num))
check("fmt_p", identical(fmt_p(c(0.0004, 0.034, 0.5, NA)), c("<0.001", "0.034", "0.500", "NA")))

# ---- fonts and figures -------------------------------------------------------
cat("-- figures\n")
hf <- systemfonts::match_fonts(FONT)
check("Helvetica resolved by systemfonts", nzchar(hf$path) && grepl("Helvetica", hf$path))

png_dim <- function(path) { b <- readBin(path, "raw", 24); c(readBin(b[17:20], "integer", size = 4, endian = "big"), readBin(b[21:24], "integer", size = 4, endian = "big")) }
tiff_tags <- function(path) {
  b <- readBin(path, "raw", file.size(path))
  en <- if (rawToChar(b[1:2]) == "II") "little" else "big"
  u <- function(i, size) readBin(b[i:(i + size - 1)], "integer", size = size, signed = (size == 4), endian = en)
  ifd <- u(5, 4) + 1
  n <- u(ifd, 2)
  out <- list()
  for (k in seq_len(n)) {
    e <- ifd + 2 + 12 * (k - 1)
    out[[as.character(u(e, 2))]] <- if (u(e + 2, 2) == 3) u(e + 8, 2) else u(e + 8, 4)
  }
  out
}
g <- ggplot(mtcars, aes(wt, mpg, colour = factor(cyl))) + geom_point() + geom_text(aes(label = cyl)) +
  labs(title = "Helvetica check") + theme_optics()
fig_paths <- save_fig(g, "zz_test_fig", height_mm = 100)
check("save_fig writes png, tif, pdf under work/synthetic/out/figures",
      all(file.exists(fig_paths)) && all(startsWith(fig_paths, file.path(SYNTH_DIR, "out", "figures"))) &&
        identical(names(fig_paths), c("png", "tif", "pdf")))
check("png is 2126 x 1181 px (180 x 100 mm at 300 dpi)", identical(png_dim(fig_paths[["png"]]), c(2126L, 1181L)))
tg <- tiff_tags(fig_paths[["tif"]])
check("tif is 2126 x 1181 px, LZW (tag 259 = 5)", tg[["256"]] == 2126 && tg[["257"]] == 1181 && tg[["259"]] == 5)
pdf_bytes <- readBin(fig_paths[["pdf"]], "raw", file.size(fig_paths[["pdf"]]))
pdf_txt <- rawToChar(replace(pdf_bytes, as.integer(pdf_bytes) == 0 | as.integer(pdf_bytes) > 127, as.raw(32)))
media <- as.numeric(strsplit(gsub("/MediaBox *\\[|\\]", "", regmatches(pdf_txt, regexpr("/MediaBox *\\[[^]]+\\]", pdf_txt))), " +")[[1]])
check("pdf page is 180 mm wide", abs(media[3] - 180 / 25.4 * 72) < 0.5)
check("pdf embeds Helvetica", grepl("Helvetica", pdf_txt, useBytes = TRUE) && grepl("/FontFile", pdf_txt, useBytes = TRUE))
check("save_fig needs height_mm", inherits(try(save_fig(g, "zz_test_fig2"), silent = TRUE), "try-error"))
unlink(fig_paths)

# ---- tables ------------------------------------------------------------------
cat("-- tables\n")
tdf <- data.frame(Variable = c("Panel A", "Patients", "Frames"), Value = c("", "99", "10,662"), stringsAsFactors = FALSE)
tft <- flex_three_line(tdf, labels = c(Variable = "Characteristic"), bold_rows = 1, indent_rows = 2:3, rule_after = 3,
                       footnote = "Note.")
tbl_paths <- save_table(tft, tdf, "zz_test_tbl")
check("save_table writes docx and csv under work/synthetic/out/tables",
      all(file.exists(tbl_paths)) && all(startsWith(tbl_paths, file.path(SYNTH_DIR, "out", "tables"))))
xml <- { d <- tempfile(); unzip(tbl_paths[["docx"]], "word/document.xml", exdir = d); paste(readLines(file.path(d, "word/document.xml"), warn = FALSE), collapse = "") }
check("docx carries the cell text, Helvetica and the header label", grepl("10,662", xml) && grepl('w:ascii="Helvetica"', xml, fixed = TRUE) && grepl("Characteristic", xml))
check("docx has a top and bottom rule and no vertical rules", grepl("<w:top ", xml) && grepl("<w:bottom ", xml) && !grepl("<w:left [^>]*w:val=\"single\"", xml))
check("csv round-trips", identical(utils::read.csv(tbl_paths[["csv"]], stringsAsFactors = FALSE, na.strings = NULL), tdf))
unlink(tbl_paths)

# ---- loaders vs contract -----------------------------------------------------
cat("-- loaders\n")
loaded <- list(manifest = load_manifest(), fov = load_fov(), optics = load_optics(), folds = load_folds(),
               predictions = load_predictions(), stream = load_stream(), perturb = load_perturb())
for (nm in names(loaded)) {
  got <- vapply(loaded[[nm]], typeof, character(1))
  check(sprintf("%s: columns and types equal the contract", nm), identical(got, CONTRACT[[nm]]))
}
# the files on disk, read raw, must also match (loaders must not be hiding a mismatch)
for (nm in c("manifest", "fov", "optics", "predictions", "stream", "perturb")) {
  raw_names <- names(arrow::read_parquet(file.path(SYNTH_DIR, paste0(nm, ".parquet"))))
  check(sprintf("%s.parquet column names on disk equal the contract", nm), identical(raw_names, names(CONTRACT[[nm]])))
}
check("folds.csv header on disk equals the contract", identical(names(utils::read.csv(file.path(SYNTH_DIR, "folds.csv"))), names(CONTRACT$folds)))

tmp_missing <- file.path(tempdir(), "no_such_file.parquet")
msg <- tryCatch(.read_contract(tmp_missing, LOAD_SPEC$manifest), error = function(e) conditionMessage(e))
check("missing file error names the file", grepl("no_such_file.parquet", msg, fixed = TRUE))
short <- file.path(tempdir(), "short.parquet")
arrow::write_parquet(loaded$manifest[setdiff(names(loaded$manifest), "domain")], short)
msg <- tryCatch(.read_contract(short, LOAD_SPEC$manifest), error = function(e) conditionMessage(e))
check("missing column error names the column and file", grepl("domain", msg) && grepl("short.parquet", msg, fixed = TRUE))

# ---- synthetic data invariants ----------------------------------------------
cat("-- synthetic data\n")
man <- loaded$manifest; fol <- loaded$folds; pr <- loaded$predictions; st <- loaded$stream; pt <- loaded$perturb
lab <- filter(man, labeled)
pats <- distinct(lab, patient_id, center, domain)
check("99 labeled patients: 61 Center 1 (41 D1 + 20 D2), 38 Center 2 (D3)",
      nrow(pats) == 99 && sum(pats$center == 1) == 61 && sum(pats$domain == "D1") == 41 &&
        sum(pats$domain == "D2") == 20 && sum(pats$center == 2) == 38 && all(pats$domain[pats$center == 2] == "D3"))
check("about 1,500 labeled frames, 4 stream patients", nrow(lab) == 1500 && n_distinct(man$patient_id[!man$labeled]) == 4)
check("frame_id unique; manifest, fov, optics have the same frames", !anyDuplicated(man$frame_id) &&
        identical(man$frame_id, loaded$fov$frame_id) && identical(man$frame_id, loaded$optics$frame_id))
check("folds.csv: 99 + 4 patients; roles cv/external/stream; folds 0-4 / -1 / -2",
      nrow(fol) == 103 && identical(sort(unique(fol$role)), c("cv", "external", "stream")) &&
        all(fol$fold[fol$role == "cv"] %in% 0:4) && all(fol$fold[fol$role == "external"] == -1) &&
        all(fol$fold[fol$role == "stream"] == -2) && sum(fol$role == "cv") == 61)
check("every fold has patients of both Center 1 domains",
      all(table(fol$fold[fol$role == "cv"], fol$domain[fol$role == "cv"]) > 0))
check("n_frames in folds.csv matches the manifest", all(fol$n_frames == as.integer(table(man$patient_id)[fol$patient_id])))
check("no patient in two folds (predictions)", {
  x <- distinct(filter(pr, source == "oof"), patient_id, fold); !anyDuplicated(x$patient_id) })
check("predictions: 2 rows per labeled frame; oof for Center 1, final for Center 2",
      nrow(pr) == 2 * nrow(lab) && all(pr$source[pr$partition == "center1_labeled"] == "oof") &&
        all(pr$source[pr$partition == "center2_labeled"] == "final") && all(pr$fold[pr$partition == "center2_labeled"] == -1))
check("ens_dis and unet_bg_* identical on both model rows", {
  w <- pr |> group_by(frame_id) |> summarise(a = n_distinct(ens_dis), b = n_distinct(unet_bg_lum_mean), .groups = "drop")
  all(w$a == 1 & w$b == 1) })
check("dice = 2 iou / (1 + iou); values in range", all(abs(pr$dice - 2 * pr$iou / (1 + pr$iou)) < 1e-9) &&
        all(pr$iou >= 0 & pr$iou <= 1) && all(pr$entropy_fov >= 0 & pr$entropy_fov <= 1))
check("stream: 480 frames x 2 models, consecutive t_index from 0", nrow(st) == 960 &&
        all(st |> filter(model == "unet") |> group_by(video_patient) |> summarise(ok = all(diff(t_index) == 1) && min(t_index) == 0) |> pull(ok)))
check("stream rows carry no labels", !any(c("iou", "dice", "gt_area_frac") %in% names(st)))
check("perturb: 7 families x severities 0-5, baseline rows equal the prediction rows", {
  base <- filter(pt, severity == 0) |> inner_join(pr, by = c("frame_id", "model"), suffix = c("", ".p"))
  identical(sort(unique(pt$family)), sort(c("defocus", "haze", "exposure_under", "exposure_over", "blood_absorption", "glare", "noise"))) &&
    identical(sort(unique(pt$severity)), 0:5) && nrow(base) == sum(pt$severity == 0) && all(abs(base$iou - base$iou.p) < 1e-12) })
check("perturb frames were unet IoU >= 0.90 out-of-fold frames", {
  u0 <- filter(pr, model == "unet", frame_id %in% pt$frame_id); all(u0$iou >= 0.90) && all(u0$source == "oof") })
check("perturb dose-response: mean IoU falls with severity in every family",
      all(pt |> group_by(family, severity) |> summarise(m = mean(iou), .groups = "drop") |>
            group_by(family) |> summarise(ok = m[severity == 5] < m[severity == 0] - 0.05) |> pull(ok)))
rho <- function(fam, metric) { d <- filter(pt, family == fam); suppressWarnings(cor(d$severity, d[[metric]], method = "spearman")) }
check("perturb metrics move as designed (defocus -sharp, noise +sharp, blood +red, glare +spec, under +under_frac)",
      rho("defocus", "sharp_tenengrad") < -0.5 && rho("noise", "sharp_tenengrad") > 0.5 && rho("blood_absorption", "red_dom") > 0.5 &&
        rho("glare", "spec_frac") > 0.5 && rho("exposure_under", "under_frac") > 0.5 && rho("haze", "haze_dcp") > 0.5)

man0 <- load_manifest()
fov0 <- load_fov()
n_nf <- sum(!fov0$fov_found[match(man0$frame_id[man0$labeled], fov0$frame_id)])
ft_all <- load_frame_table(exclude_no_fov = FALSE)
ft <- load_frame_table()   # default: fov.exclude_not_found is true in the plan, so frames without a detected FOV are dropped
check("load_frame_table(exclude_no_fov = FALSE): one row per labeled frame with unet_ and segformer_ columns",
      nrow(ft_all) == 1500 && !anyDuplicated(ft_all$frame_id) && attr(ft_all, "n_excluded_fov") == 0L &&
        all(c("unet_iou", "segformer_iou", "unet_entropy_fov", "segformer_entropy_roi", "ens_dis", "gt_area_frac",
              "unet_bg_haze_dcp", "role", "fold", "fov_r", M11) %in% names(ft_all)) && !anyNA(ft_all$unet_iou) && !anyNA(ft_all$segformer_iou))
check("load_frame_table drops frames without a detected FOV by default and reports the count (synthetic: some are flagged)",
      isTRUE(CFG$fov$exclude_not_found) && n_nf > 0L && attr(ft, "n_excluded_fov") == n_nf && nrow(ft) == 1500 - n_nf &&
        all(ft$fov_found) && identical(ft$frame_id, ft_all$frame_id[ft_all$fov_found]) &&
        attr(load_frame_table(FALSE), "n_excluded_fov") == n_nf)
check("load_frame_table(with_predictions = FALSE) drops predictions", !("unet_iou" %in% names(load_frame_table(FALSE))))

# planted structure, as documented in planted_effects.json
int <- filter(ft, center == 1); ext <- filter(ft, center == 2)
rk <- function(x, y) cor(x, y, method = "spearman")
check("planted: risk rises with under_frac, falls with sharpness, rises with haze (Center 1)",
      rk(1 - int$unet_iou, int$under_frac) > 0.25 && rk(1 - int$unet_iou, int$sharp_tenengrad) < -0.25 &&
        rk(1 - int$unet_iou, int$haze_dcp) > 0.25)
check("planted: entropy and ensemble disagreement track risk", rk(1 - int$unet_iou, int$unet_entropy_fov) > 0.3 &&
        rk(1 - int$unet_iou, int$ens_dis) > 0.45)
check("planted: failures (IoU < 0.75) rarer inside than outside, enough external events",
      mean(int$unet_iou < 0.75) < 0.25 && mean(ext$unet_iou < 0.75) > 0.3 &&
        sum(ext$unet_iou < 0.75) >= CFG$endpoints$min_external_failure_events &&
        n_distinct(ext$patient_id[ext$unet_iou < 0.75]) >= CFG$endpoints$min_external_patients_with_events)
check("planted: Center 2 metrics shifted (darker, more under-exposed)", median(ext$lum_mean) < median(int$lum_mean) &&
        median(ext$under_frac) > median(int$under_frac))

check("real figures/ and tables/ untouched by the tests",
      identical(list.files(file.path(V1, "figures")), real_before$figures) && identical(list.files(file.path(V1, "tables")), real_before$tables))

cat(sprintf("\n%s\n", if (length(failures)) paste(length(failures), "FAILED:", paste(failures, collapse = "; ")) else "all checks passed"))
if (length(failures)) quit(status = 1)
