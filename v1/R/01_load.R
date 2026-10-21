# Loaders for the derived tables in config/CONTRACTS.md. Each returns a tibble with exactly the
# contract columns, in contract order and type; a missing file or column is a hard error that names it.
# Source 00_setup.R first.

if (!exists("p", mode = "function") || !exists("OPTICAL_METRICS")) stop("source 00_setup.R before 01_load.R")

.metric_spec <- function(prefix = "") setNames(rep("dbl", length(OPTICAL_METRICS)), paste0(prefix, OPTICAL_METRICS))

.pred_core <- c(frame_id = "chr", patient_id = "chr", model = "chr", source = "chr",
                partition = "chr", fold = "int")

LOAD_SPEC <- list(
  manifest = c(frame_id = "chr", partition = "chr", patient_id = "chr", center = "int", domain = "chr",
               width = "int", height = "int", channel = "chr", region = "chr", scenario = "chr",
               instrument = "chr", file_format = "chr", video_idx = "int", frame_idx = "int",
               image_path = "chr", mask_path = "chr", labeled = "lgl"),
  fov = c(frame_id = "chr", fov_found = "lgl", fov_cx = "dbl", fov_cy = "dbl", fov_r = "dbl",
          fov_method = "chr", fov_resid = "dbl", crop_x0 = "int", crop_y0 = "int", crop_side = "int"),
  optics = c(frame_id = "chr", .metric_spec()),
  folds = c(patient_id = "chr", center = "int", domain = "chr", n_frames = "int", role = "chr", fold = "int"),
  predictions = c(.pred_core, iou = "dbl", dice = "dbl", pred_area_frac = "dbl", gt_area_frac = "dbl",
                  entropy_fov = "dbl", entropy_roi = "dbl", ens_dis = "dbl", .metric_spec("unet_bg_")),
  stream = c(.pred_core, pred_area_frac = "dbl", entropy_fov = "dbl", entropy_roi = "dbl", ens_dis = "dbl",
             .metric_spec("unet_bg_"), video_patient = "chr", t_index = "int"),
  perturb = c(frame_id = "chr", patient_id = "chr", domain = "chr", scenario = "chr", instrument = "chr",
              model = "chr", family = "chr", severity = "int", param = "dbl", iou = "dbl", dice = "dbl",
              pred_area_frac = "dbl", entropy_fov = "dbl", .metric_spec())
)

.coerce <- c(chr = as.character, int = as.integer, dbl = as.numeric, lgl = as.logical)

.read_contract <- function(path, spec, unique_id = NULL) {
  if (!file.exists(path)) {
    stop("File not found: ", path, "\n  (produced by an upstream task; use OPTICS_DATA=synthetic for development)",
         call. = FALSE)
  }
  df <- if (grepl("\\.parquet$", path)) arrow::read_parquet(path) else utils::read.csv(path, stringsAsFactors = FALSE)
  df <- tibble::as_tibble(df)
  missing_cols <- setdiff(names(spec), names(df))
  if (length(missing_cols)) {
    stop(path, " is missing contract columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  extra <- setdiff(names(df), names(spec))
  if (length(extra)) warning(path, " has columns outside the contract (kept at the end): ", paste(extra, collapse = ", "), call. = FALSE)
  for (col in names(spec)) df[[col]] <- .coerce[[spec[[col]]]](df[[col]])
  df <- df[c(names(spec), extra)]
  if (!is.null(unique_id) && anyDuplicated(df[unique_id])) {
    stop(path, " has duplicated ", paste(unique_id, collapse = " + "), call. = FALSE)
  }
  df
}

load_manifest <- function() .read_contract(p("derived", "manifest.parquet"), LOAD_SPEC$manifest, "frame_id")
load_fov <- function() .read_contract(p("derived", "fov.parquet"), LOAD_SPEC$fov, "frame_id")
load_optics <- function() .read_contract(p("derived", "optics.parquet"), LOAD_SPEC$optics, "frame_id")
load_folds <- function() .read_contract(p("derived", "folds.csv"), LOAD_SPEC$folds, "patient_id")
load_predictions <- function() .read_contract(p("derived", "predictions.parquet"), LOAD_SPEC$predictions, c("frame_id", "model"))
load_stream <- function() .read_contract(p("derived", "stream.parquet"), LOAD_SPEC$stream, c("frame_id", "model"))
load_perturb <- function() .read_contract(p("derived", "perturb.parquet"), LOAD_SPEC$perturb)

# One row per labeled frame: manifest + FOV + the 11 metrics + patient role/fold. Predictions, when the
# file exists, are widened to <model>_{source,iou,dice,pred_area_frac,entropy_fov,entropy_roi} columns
# (models: unet, segformer). gt_area_frac, ens_dis and unet_bg_* are identical on both model rows
# of a frame, so they appear once, unprefixed.
# exclude_no_fov (default: fov.exclude_not_found in the plan) drops the labeled frames whose field of view was not
# found (fov_found == FALSE); they have NaN optics and no predictions. The count is attr(, "n_excluded_fov"), and the
# missing-value check on the optics then runs on the remaining frames only.
load_frame_table <- function(with_predictions = file.exists(p("derived", "predictions.parquet")),
                             exclude_no_fov = isTRUE(CFG$fov$exclude_not_found)) {
  ft <- load_manifest() |>
    filter(labeled) |>
    left_join(load_fov(), by = "frame_id") |>
    left_join(load_optics(), by = "frame_id") |>
    left_join(select(load_folds(), patient_id, role, fold), by = "patient_id")
  if (anyNA(ft$role) || anyNA(ft$fov_found)) {
    stop("Labeled frames without a folds or fov row: the derived tables disagree", call. = FALSE)
  }
  n_excluded <- 0L
  if (exclude_no_fov) {
    n_excluded <- sum(!ft$fov_found)
    ft <- filter(ft, fov_found)
  }
  if (anyNA(ft[[OPTICAL_METRICS[1]]])) {
    stop("Labeled frames without an optics value: the derived tables disagree", call. = FALSE)
  }
  if (with_predictions) ft <- .add_predictions(ft)
  attr(ft, "n_excluded_fov") <- n_excluded
  ft
}

.add_predictions <- function(ft) {
  pr <- load_predictions()
  shared <- distinct(pr, frame_id, gt_area_frac, ens_dis, across(starts_with("unet_bg_")))
  if (anyDuplicated(shared$frame_id)) {
    stop("gt_area_frac, ens_dis or unet_bg_* differ between the model rows of a frame", call. = FALSE)
  }
  ft <- left_join(ft, shared, by = "frame_id")
  per_model <- c("source", "iou", "dice", "pred_area_frac", "entropy_fov", "entropy_roi")
  for (m in sort(unique(pr$model), decreasing = TRUE)) {   # unet before segformer
    one <- pr[pr$model == m, c("frame_id", per_model)]
    names(one)[-1] <- paste0(m, "_", per_model)
    ft <- left_join(ft, one, by = "frame_id")
  }
  ft
}
