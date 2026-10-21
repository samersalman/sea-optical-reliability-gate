# Writes small synthetic stand-ins for every derived table (schemas exactly as CONTRACTS.md) to
# work/synthetic/, plus work/synthetic/planted_effects.json describing what was planted.
#   Rscript v1/R/tests/make_synthetic.R
# Always runs in synthetic mode so it can never overwrite real derived/ files.
# Deterministic: everything is drawn from seeds$global in analysis_plan.yaml.

Sys.setenv(OPTICS_DATA = "synthetic")
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
source(file.path(dirname(.here), "00_setup.R"))
set_seeds("global")

clip <- function(x, lo, hi) pmin(pmax(x, lo), hi)
shift_logit <- function(x, by) plogis(qlogis(clip(x, 1e-6, 1 - 1e-6)) + by)
dir.create(SYNTH_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- patients ----------------------------------------------------------------
pat <- tibble(
  patient_id = c(paste0("P", 1:61), paste0("T", 101:138)),
  center = c(rep(1L, 61), rep(2L, 38)),
  domain = c(rep("D1", 41), rep("D2", 20), rep("D3", 38))
)
frames_per_domain <- c(D1 = 820, D2 = 240, D3 = 440)   # ~1,500 labeled frames
pat$n_frames <- as.integer(unlist(lapply(names(frames_per_domain), function(d) {
  w <- rlnorm(sum(pat$domain == d), 0, 0.7)
  n <- pmax(4, round(frames_per_domain[[d]] * w / sum(w)))
  n[which.max(n)] <- n[which.max(n)] + frames_per_domain[[d]] - sum(n)
  n
})))
pat$channel <- "small"
pat$channel[c(sample(which(pat$domain == "D1"), 7), sample(which(pat$domain == "D2"), 3))] <- "big"
pat$region <- sample(c("cervical", "lumbar"), nrow(pat), replace = TRUE, prob = c(0.4, 0.6))
LATENTS <- c("L", "H", "B", "R", "G")   # light level, haze, blur, blood, glare
pat_re <- matrix(rnorm(nrow(pat) * 5, 0, 0.5), ncol = 5, dimnames = list(NULL, LATENTS))
pat$u_eta <- rnorm(nrow(pat), 0, 0.35)
pat$u_area <- rnorm(nrow(pat), 0, 0.2)

# 5-fold patient-grouped folds for Center 1: snake order within domain, largest patients first
pat$fold <- NA_integer_
for (d in c("D1", "D2")) {
  i <- which(pat$domain == d)
  i <- i[order(-pat$n_frames[i], pat$patient_id[i])]
  pat$fold[i] <- rep(c(0:4, 4:0), length.out = length(i))
}
pat$role <- c(rep("cv", 61), rep("external", 38))
pat$fold[pat$role == "external"] <- -1L

# ---- labeled frames ----------------------------------------------------------
pidx <- rep(seq_len(nrow(pat)), pat$n_frames)
fr <- pat[pidx, c("patient_id", "center", "domain", "channel", "region")]
n <- nrow(fr)
fr$scenario <- ifelse(runif(n) < c(D1 = 0.72, D2 = 0.92, D3 = 0.70)[fr$domain], "Difficult", "Normal")
INSTR_P <- c(forceps = 4918, bipolar = 2933, drill = 1283, punch = 880, scissor = 407, dissector = 241)
INSTR_AREA <- c(forceps = 1, bipolar = 0.9, drill = 1.2, punch = 1.1, scissor = 0.7, dissector = 0.6)
fr$instrument <- sample(names(INSTR_P), n, replace = TRUE, prob = INSTR_P)
nv <- ifelse(pat$domain == "D3", 2L, sample(1:3, nrow(pat), replace = TRUE))
fr$vcode <- vapply(pidx, function(i) sample(1:nv[i], 1), integer(1))
fr$vcode[fr$domain == "D3" & runif(n) < 0.25] <- 0L   # Center 2 names without a video index
fr <- fr |>
  group_by(patient_id, vcode) |>
  mutate(frame_idx = sort(sample(seq_len(n() * 6), n()))) |>
  ungroup()
fr$video_idx <- ifelse(fr$vcode == 0L, NA_integer_, fr$vcode)
fr$frame_idx <- as.integer(fr$frame_idx)
fr$frame_id <- ifelse(is.na(fr$video_idx), sprintf("%s-%04d", fr$patient_id, fr$frame_idx),
                      sprintf("%s-%d-%04d", fr$patient_id, fr$video_idx, fr$frame_idx))
stopifnot(!anyDuplicated(fr$frame_id))

# ---- optical metrics ---------------------------------------------------------
DOM_BASE <- data.frame(
  row.names = c("D1", "D2", "D3"),
  lum = c(0.42, 0.38, 0.30), under = c(-3.2, -2.7, -1.8), sharp = c(0.030, 0.022, 0.014),
  haze = c(0.10, 0.12, 0.20), crms = c(0.20, 0.18, 0.15), cloc = c(0.060, 0.052, 0.040),
  sat = c(-4.2, -3.9, -4.8), spec = c(-4.6, -4.4, -4.9), red = c(0.43, 0.44, 0.50),
  slope = c(-0.25, -0.22, -0.12), ent = c(5.0, 4.85, 4.5)
)

gen_metrics <- function(domain, lat) {
  b <- DOM_BASE[domain, ]
  m <- nrow(lat)
  with(as.data.frame(lat), tibble(
    lum_mean = clip(b$lum + 0.06 * L - 0.02 * R + 0.015 * H + rnorm(m, 0, 0.012), 0.03, 0.85),
    contrast_rms = exp(log(b$crms) + 0.15 * L - 0.20 * H - 0.05 * R + rnorm(m, 0, 0.08)),
    contrast_local = exp(log(b$cloc) - 0.20 * B - 0.20 * H + 0.05 * L + rnorm(m, 0, 0.10)),
    sharp_tenengrad = exp(log(b$sharp) - 0.40 * B - 0.25 * H + rnorm(m, 0, 0.15)),
    under_frac = plogis(b$under - 0.9 * L + 0.25 * R + 0.2 * H + rnorm(m, 0, 0.3)),
    sat_frac = plogis(b$sat + 0.5 * G + 0.2 * L + rnorm(m, 0, 0.4)),
    spec_frac = plogis(b$spec + 0.8 * G + rnorm(m, 0, 0.4)),
    haze_dcp = clip(b$haze + 0.045 * H + 0.012 * R - 0.015 * L + rnorm(m, 0, 0.012), 0.01, 0.7),
    red_dom = clip(b$red + 0.025 * R - 0.008 * L + rnorm(m, 0, 0.010), 0.3, 0.7),
    illum_slope = b$slope + 0.04 * L - 0.02 * H + rnorm(m, 0, 0.035),
    lum_entropy = clip(b$ent + 0.25 * L - 0.20 * H + rnorm(m, 0, 0.12), 2, 6)
  ))[, OPTICAL_METRICS]
}

difficult <- fr$scenario == "Difficult"
lat <- sapply(LATENTS, function(k) pat_re[pidx, k] + rnorm(n, 0, 0.8) +
                c(L = -0.35, H = 0.45, B = 0.35, R = 0.30, G = 0.20)[[k]] * difficult)
met <- gen_metrics(fr$domain, lat)
gt_area <- clip(exp(log(0.08) + log(INSTR_AREA[fr$instrument]) + pat$u_area[pidx] + rnorm(n, 0, 0.4)), 0.004, 0.5)

# Reference means/sds fix the scale of the planted terms (same constants for the stream).
REF <- list(
  under = c(mean(log(met$under_frac)), sd(log(met$under_frac))),
  sharp = c(mean(log(met$sharp_tenengrad)), sd(log(met$sharp_tenengrad))),
  haze = c(mean(met$haze_dcp), sd(met$haze_dcp)),
  area = c(mean(log(gt_area)), sd(log(gt_area))),
  metric_sd = vapply(met, sd, numeric(1))
)
z <- function(x, ref) (x - ref[1]) / ref[2]

# ---- planted outcome model ---------------------------------------------------
# risk = 1 - IoU = plogis(intercept[domain] + under + sharp + haze + size + patient + noise + model offset + noise)
COEF <- list(under_linear = 0.45, under_quadratic = 0.15, sharp = -0.55, haze = 0.45, log_gt_area = -0.30)
INTERCEPT <- c(D1 = -1.9, D2 = -2.1, D3 = -2.8)
MODEL_OFFSET <- c(unet = 0, segformer = 0.30)
BG_BOUNDS <- list(lum_mean = c(0, 1), contrast_rms = c(0, 1), contrast_local = c(0, 1),
                  sharp_tenengrad = c(0, Inf), under_frac = c(0, 1), sat_frac = c(0, 1),
                  spec_frac = c(0, 1), haze_dcp = c(0, 1), red_dom = c(0, 1),
                  illum_slope = c(-Inf, Inf), lum_entropy = c(0, 6))

gen_outcomes <- function(domain, met, gt_area, u_eta) {
  m <- length(domain)
  zu <- z(log(met$under_frac), REF$under)
  eta_optics <- COEF$under_linear * zu + COEF$under_quadratic * pmax(zu, 0)^2 +
    COEF$sharp * z(log(met$sharp_tenengrad), REF$sharp) + COEF$haze * z(met$haze_dcp, REF$haze) +
    COEF$log_gt_area * z(log(gt_area), REF$area)
  eta_common <- unname(INTERCEPT[domain]) + eta_optics + u_eta + rnorm(m, 0, 0.35)
  out <- list(eta_optics = unname(eta_optics))
  for (mod in names(MODEL_OFFSET)) {
    risk <- plogis(eta_common + MODEL_OFFSET[[mod]] + rnorm(m, 0, 0.45))
    empty <- runif(m) < 0.008 + 0.03 * (domain == "D3")   # empty prediction: IoU 0
    risk[empty] <- 1
    out[[mod]] <- tibble(
      risk = risk,
      pred_area_frac = ifelse(empty, 0, clip(gt_area * (1 - 0.6 * risk) * exp(rnorm(m, 0, 0.15)), 0.001, 0.6)),
      entropy_fov = clip(exp(log(0.025) + 1.6 * risk + 0.3 * (domain == "D3") + rnorm(m, 0, 0.35)), 0, 1),
      entropy_roi = clip(exp(log(0.12) + 1.4 * risk + 0.25 * (domain == "D3") + rnorm(m, 0, 0.45)), 0, 1)
    )
  }
  rbar <- (out$unet$risk + out$segformer$risk) / 2
  out$ens_dis <- plogis(-3.3 + 5.2 * rbar + rnorm(m, 0, 0.7))
  zpa <- as.numeric(scale(log(pmax(out$unet$pred_area_frac, 0.001))))
  bg <- lapply(OPTICAL_METRICS, function(k) {
    v <- met[[k]] + REF$metric_sd[[k]] * (0.15 * rnorm(m) + 0.10 * zpa)
    clip(v, BG_BOUNDS[[k]][1], BG_BOUNDS[[k]][2])
  })
  names(bg) <- paste0("unet_bg_", OPTICAL_METRICS)
  out$bg <- as_tibble(bg)
  out
}
oc <- gen_outcomes(fr$domain, met, gt_area, pat$u_eta[pidx])

# ---- predictions (long, one row per frame x model) ---------------------------
pred_rows <- function(mod) {
  r <- oc[[mod]]
  iou <- 1 - r$risk
  bind_cols(
    tibble(frame_id = fr$frame_id, patient_id = fr$patient_id, model = mod,
           source = ifelse(fr$center == 1, "oof", "final"),
           partition = ifelse(fr$center == 1, "center1_labeled", "center2_labeled"),
           fold = pat$fold[pidx], iou = iou, dice = 2 * iou / (1 + iou),
           pred_area_frac = r$pred_area_frac, gt_area_frac = gt_area,
           entropy_fov = r$entropy_fov, entropy_roi = r$entropy_roi, ens_dis = oc$ens_dis),
    oc$bg
  )
}
predictions <- bind_rows(pred_rows("unet"), pred_rows("segformer"))

# ---- stream: 4 videos, autocorrelated latents with smoke / blood episodes ---
STREAM_N <- 120L
stream_pat <- tibble(patient_id = paste0("P", 69:72), domain = rep("D1", 4))   # real stream frames are all D1 (1920x1080)
ar1 <- function(len, phi = 0.97) {
  x <- numeric(len); x[1] <- rnorm(1)
  for (t in 2:len) x[t] <- phi * x[t - 1] + sqrt(1 - phi^2) * rnorm(1)
  x
}
episode <- function(len, centre, width) pmax(0, 0.5 * (1 + cos(pi * (seq_len(len) - centre) / width))) * (abs(seq_len(len) - centre) < width)
s_lat <- do.call(rbind, lapply(seq_len(nrow(stream_pat)), function(i) {
  bump <- episode(STREAM_N, sample(25:95, 1), 18)
  cbind(L = ar1(STREAM_N) * 0.8 - 0.9 * bump, H = ar1(STREAM_N) * 0.8 + 2.5 * bump,
        B = ar1(STREAM_N) * 0.8 + 1.0 * bump, R = ar1(STREAM_N) * 0.8, G = ar1(STREAM_N) * 0.8)
}))
s_dom <- rep(stream_pat$domain, each = STREAM_N)
s_met <- gen_metrics(s_dom, s_lat)
s_area <- clip(exp(log(0.08) + rnorm(nrow(s_lat), 0, 0.4)), 0.004, 0.5)
s_u <- rep(rnorm(nrow(stream_pat), 0, 0.35), each = STREAM_N)
s_oc <- gen_outcomes(s_dom, s_met, s_area, s_u)
s_id <- rep(stream_pat$patient_id, each = STREAM_N)
s_t <- rep(seq_len(STREAM_N) - 1L, times = nrow(stream_pat))
s_frame_id <- sprintf("%s-1-%04d", s_id, s_t + 1L)
stream <- bind_rows(lapply(c("unet", "segformer"), function(mod) {
  bind_cols(
    tibble(frame_id = s_frame_id, patient_id = s_id, model = mod, source = "final", partition = "stream",
           fold = -2L, pred_area_frac = s_oc[[mod]]$pred_area_frac, entropy_fov = s_oc[[mod]]$entropy_fov,
           entropy_roi = s_oc[[mod]]$entropy_roi, ens_dis = s_oc$ens_dis),
    s_oc$bg,
    tibble(video_patient = s_id, t_index = s_t)
  )
}))

# ---- manifest, fov, optics, folds -------------------------------------------
SIZE <- list(D1 = c(1920L, 1080L), D2 = c(1280L, 720L), D3 = c(720L, 576L))
FOV_R <- c(D1 = 406, D2 = 271, D3 = 255)
img_path <- function(center, channel, region, patient, scenario, instrument, stem, ext) {
  file.path("Spine endoscopic atlas", "classified", paste(tools::toTitleCase(channel), "channel"), region,
            patient, paste(scenario, "scenario"), instrument, paste0(stem, ".", ext))
}
fmt <- ifelse(fr$domain == "D3", "png", "jpg")
man_lab <- tibble(
  frame_id = fr$frame_id,
  partition = ifelse(fr$center == 1, "center1_labeled", "center2_labeled"),
  patient_id = fr$patient_id, center = fr$center, domain = fr$domain,
  width = vapply(fr$domain, function(d) SIZE[[d]][1], integer(1)),
  height = vapply(fr$domain, function(d) SIZE[[d]][2], integer(1)),
  channel = fr$channel, region = fr$region, scenario = fr$scenario, instrument = fr$instrument,
  file_format = fmt, video_idx = fr$video_idx, frame_idx = fr$frame_idx,
  image_path = img_path(fr$center, fr$channel, fr$region, fr$patient_id, fr$scenario, fr$instrument, fr$frame_id, fmt),
  mask_path = img_path(fr$center, fr$channel, fr$region, fr$patient_id, fr$scenario, fr$instrument, fr$frame_id, "nrrd"),
  labeled = TRUE
)
man_stream <- tibble(
  frame_id = s_frame_id, partition = "stream", patient_id = s_id, center = 1L, domain = s_dom,
  width = vapply(s_dom, function(d) SIZE[[d]][1], integer(1)),
  height = vapply(s_dom, function(d) SIZE[[d]][2], integer(1)),
  channel = NA_character_, region = NA_character_, scenario = NA_character_, instrument = NA_character_,
  file_format = "jpg", video_idx = 1L, frame_idx = s_t + 1L,
  image_path = file.path("Spine endoscopic atlas", "unclassified", s_id, paste0(s_frame_id, ".jpg")),
  mask_path = NA_character_, labeled = FALSE
)
manifest <- bind_rows(man_lab, man_stream)

fov <- tibble(frame_id = manifest$frame_id) |>
  mutate(
    fov_cx = manifest$width / 2 + rnorm(n(), 0, 6), fov_cy = manifest$height / 2 + rnorm(n(), 0, 5),
    fov_r = FOV_R[manifest$domain] + rnorm(n(), 0, 5),
    fov_method = sample(c("ray", "mask", "fallback"), n(), replace = TRUE, prob = c(0.85, 0.12, 0.03)),
    fov_found = fov_method != "fallback",   # "fallback" is fov.py's not-found geometry
    fov_resid = 0.3 + abs(rnorm(n(), 0, 1.2)),
    crop_side = as.integer(round(2 * fov_r)),
    crop_x0 = as.integer(round(fov_cx - fov_r)), crop_y0 = as.integer(round(fov_cy - fov_r))
  ) |>
  select(frame_id, fov_found, fov_cx, fov_cy, fov_r, fov_method, fov_resid, crop_x0, crop_y0, crop_side)

optics <- bind_cols(tibble(frame_id = manifest$frame_id), bind_rows(met, s_met))

folds <- bind_rows(
  pat |> transmute(patient_id, center, domain, n_frames, role, fold),
  tibble(patient_id = stream_pat$patient_id, center = 1L, domain = stream_pat$domain,
         n_frames = STREAM_N, role = "stream", fold = -2L)
)

# ---- perturbation experiment -------------------------------------------------
# The synthetic grids below are placeholders. They differ from python/perturb_ops.py CONDITIONS,
# which is authoritative for the real experiment.
FAMILIES <- c("defocus", "haze", "exposure_under", "exposure_over", "blood_absorption", "glare", "noise")
PARAM <- list(
  defocus = c(0, 1, 2, 3, 4.5, 6.5), haze = c(0, 0.1, 0.2, 0.35, 0.55, 0.8),
  exposure_under = c(1, 0.8, 0.62, 0.48, 0.35, 0.25), exposure_over = c(1, 1.3, 1.7, 2.2, 2.9, 3.8),
  blood_absorption = c(0, 0.4, 0.8, 1.3, 1.9, 2.6), glare = c(0, 0.01, 0.02, 0.04, 0.07, 0.1),
  noise = c(0, 0.01, 0.02, 0.035, 0.055, 0.08)
)
IOU_DROP <- c(defocus = 0.25, haze = 0.55, exposure_under = 0.60, exposure_over = 0.35,
              blood_absorption = 0.30, glare = 0.20, noise = 0.40)
IOU_POWER <- c(defocus = 1.6, haze = 1.4, exposure_under = 1.3, exposure_over = 1.5,
               blood_absorption = 1.2, glare = 1.1, noise = 1.5)

ft_unet <- predictions[predictions$model == "unet", ]
cand <- which(ft_unet$source == "oof" & ft_unet$iou >= 0.90)
dom_c <- fr$domain[cand]
pick <- c(cand[dom_c == "D1"][seq_len(min(36, sum(dom_c == "D1")))],
          cand[dom_c == "D2"][seq_len(min(24, sum(dom_c == "D2")))])
pick <- sample(pick)   # frames are in file order; shuffle so the pick is not patient-ordered
nsel <- length(pick)

grid <- tidyr::expand_grid(i = seq_len(nsel), model = c("unet", "segformer"), family = FAMILIES, severity = 0:5)
fi <- pick[grid$i]
w_iou <- exp(rnorm(nsel * length(FAMILIES), 0, 0.3))[(grid$i - 1) * length(FAMILIES) + match(grid$family, FAMILIES)]
w_met <- exp(rnorm(nrow(grid), 0, 0.25))
mod_scale <- ifelse(grid$model == "segformer", 1.15, 1)
base <- lapply(c("unet", "segformer"), function(mod) {
  pr <- predictions[predictions$model == mod, ]
  list(iou = pr$iou, pa = pr$pred_area_frac, ent = pr$entropy_fov)
})
names(base) <- c("unet", "segformer")
iou0 <- ifelse(grid$model == "unet", base$unet$iou[fi], base$segformer$iou[fi])
pa0 <- ifelse(grid$model == "unet", base$unet$pa[fi], base$segformer$pa[fi])
ent0 <- ifelse(grid$model == "unet", base$unet$ent[fi], base$segformer$ent[fi])
dose <- grid$severity / 5
iou_s <- clip(iou0 * (1 - IOU_DROP[grid$family] * w_iou * mod_scale * dose^IOU_POWER[grid$family]) +
                ifelse(grid$severity > 0, rnorm(nrow(grid), 0, 0.01), 0), 0, 1)
lost <- ifelse(iou0 > 0, 1 - iou_s / iou0, 0)
perturb <- grid |>
  mutate(
    frame_id = fr$frame_id[fi], patient_id = fr$patient_id[fi], domain = fr$domain[fi],
    scenario = fr$scenario[fi], instrument = fr$instrument[fi],
    param = unname(unlist(Map(function(f, s) PARAM[[f]][s + 1], family, severity))),
    iou = iou_s, dice = 2 * iou_s / (1 + iou_s),
    pred_area_frac = pa0 * (1 - 0.85 * lost), entropy_fov = clip(ent0 * exp(2.2 * lost), 0, 1)
  )
m0 <- met[fi, ]
t_eff <- dose * w_met
nz <- function(len, sd) rnorm(len, 0, sd)   # small measurement noise on perturbed rows
pm <- m0
f <- grid$family
fam <- function(k) f == k
pm$sharp_tenengrad <- m0$sharp_tenengrad * exp(
  -1.2 * t_eff * fam("defocus") - 0.6 * t_eff * fam("haze") - 1.2 * t_eff * fam("exposure_under") +
    1.0 * t_eff * fam("noise"))
pm$contrast_local <- m0$contrast_local * exp(
  -0.5 * t_eff * fam("defocus") - 0.9 * t_eff * fam("haze") + 0.6 * t_eff * fam("noise"))
pm$contrast_rms <- m0$contrast_rms * exp(
  -0.1 * t_eff * fam("defocus") - 0.8 * t_eff * fam("haze") - 0.8 * t_eff * fam("exposure_under"))
pm$haze_dcp <- clip(m0$haze_dcp + 0.35 * t_eff * fam("haze") + 0.05 * t_eff * fam("exposure_over") -
                      0.8 * m0$haze_dcp * t_eff * fam("blood_absorption"), 0, 1)
pm$lum_mean <- clip(m0$lum_mean + 0.10 * t_eff * fam("haze") - 0.7 * m0$lum_mean * t_eff * fam("exposure_under") +
                      (1 - m0$lum_mean) * 0.7 * t_eff * fam("exposure_over") -
                      0.35 * m0$lum_mean * t_eff * fam("blood_absorption") + 0.05 * t_eff * fam("glare"), 0, 1)
pm$under_frac <- shift_logit(m0$under_frac, 5 * t_eff * fam("exposure_under") - 1.5 * t_eff * fam("haze") -
                               2 * t_eff * fam("exposure_over") + 0.6 * t_eff * fam("blood_absorption"))
pm$sat_frac <- shift_logit(m0$sat_frac, 5 * t_eff * fam("exposure_over") + 3 * t_eff * fam("glare"))
pm$spec_frac <- shift_logit(m0$spec_frac, 4 * t_eff * fam("glare") + 0.8 * t_eff * fam("exposure_over"))
pm$red_dom <- clip(m0$red_dom + 0.12 * t_eff * fam("blood_absorption"), 0, 1)
pm$lum_entropy <- clip(m0$lum_entropy - 0.8 * t_eff * fam("haze") - 1.5 * t_eff * fam("exposure_under") -
                         0.8 * t_eff * fam("exposure_over") + 0.3 * t_eff * fam("noise"), 0, 6)
pm$illum_slope <- m0$illum_slope - 0.05 * t_eff * (fam("haze") | fam("exposure_under")) +
  0.04 * t_eff * fam("exposure_over")
changed <- grid$severity > 0
for (k in OPTICAL_METRICS) {   # unchanged baseline; tiny noise on perturbed rows
  noisy <- clip(pm[[k]] + nz(nrow(pm), 0.01 * REF$metric_sd[[k]]), BG_BOUNDS[[k]][1], BG_BOUNDS[[k]][2])
  pm[[k]] <- ifelse(changed, noisy, m0[[k]])
}
perturb <- bind_cols(perturb, pm) |>
  select(frame_id, patient_id, domain, scenario, instrument, model, family, severity, param,
         iou, dice, pred_area_frac, entropy_fov, all_of(OPTICAL_METRICS))

# ---- write -------------------------------------------------------------------
w <- function(df, file) arrow::write_parquet(df, p("derived", file))
w(manifest, "manifest.parquet")
w(fov, "fov.parquet")
w(optics, "optics.parquet")
w(predictions, "predictions.parquet")
w(stream, "stream.parquet")
w(perturb, "perturb.parquet")
utils::write.csv(folds, p("derived", "folds.csv"), row.names = FALSE)

# ---- diagnostics: what the planted structure implies --------------------------
auroc <- function(score, y) {
  r <- rank(score); n1 <- sum(y); n0 <- sum(!y)
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
aurc <- function(risk, score) mean(cumsum(risk[order(score)]) / seq_along(risk))
ext <- fr$center == 2
u <- predictions[predictions$model == "unet", ]
rows <- list(
  list("optics (true planted score)", oc$eta_optics), list("predicted area (inverse)", -u$pred_area_frac),
  list("entropy_fov", u$entropy_fov), list("entropy_roi", u$entropy_roi), list("ens_dis", u$ens_dis)
)
diag_tab <- bind_rows(lapply(rows, function(r) tibble(
  ranking = r[[1]],
  auroc_internal = auroc(r[[2]][!ext], u$iou[!ext] < 0.75), auroc_external = auroc(r[[2]][ext], u$iou[ext] < 0.75),
  aurc_internal = aurc(1 - u$iou[!ext], r[[2]][!ext]), aurc_external = aurc(1 - u$iou[ext], r[[2]][ext])
)))
cmb <- data.frame(fail = u$iou < 0.75, ens = qlogis(u$ens_dis), optics = oc$eta_optics)
combo <- predict(glm(fail ~ ens + optics, family = binomial, data = cmb[!ext, ]), cmb)
diag_tab <- bind_rows(diag_tab, tibble(
  ranking = "ens_dis + optics score (logistic, fit on Center 1)",
  auroc_internal = auroc(combo[!ext], u$iou[!ext] < 0.75), auroc_external = auroc(combo[ext], u$iou[ext] < 0.75),
  aurc_internal = aurc(1 - u$iou[!ext], combo[!ext]), aurc_external = aurc(1 - u$iou[ext], combo[ext])
))
ref_rows <- tibble(
  ranking = c("random", "oracle"), auroc_internal = NA_real_, auroc_external = NA_real_,
  aurc_internal = c(mean(1 - u$iou[!ext]), aurc(1 - u$iou[!ext], 1 - u$iou[!ext])),
  aurc_external = c(mean(1 - u$iou[ext]), aurc(1 - u$iou[ext], 1 - u$iou[ext]))
)
diag_tab <- bind_rows(diag_tab, ref_rows)
rates <- predictions |>
  mutate(domain = fr$domain[match(frame_id, fr$frame_id)]) |>
  group_by(model, domain) |>
  summarise(frames = n(), median_iou = median(iou), failure_rate = mean(iou < 0.75),
            patients_with_failure = n_distinct(patient_id[iou < 0.75]), .groups = "drop")

planted <- list(
  note = "Synthetic data. risk = 1 - IoU = plogis(eta); eta = intercept[domain] + optics terms + patient + noise + model offset + noise.",
  seed = CFG$seeds$global,
  optics_terms = list(
    under_frac = "0.45 * z(log under_frac) + 0.15 * max(z, 0)^2",
    sharp_tenengrad = "-0.55 * z(log sharp_tenengrad)",
    haze_dcp = "0.45 * z(haze_dcp)",
    instrument_size = "-0.30 * z(log gt_area_frac); predicted area carries it noisily",
    null_metrics_given_the_rest = c("lum_mean", "contrast_rms", "contrast_local", "sat_frac", "spec_frac",
                                    "red_dom", "illum_slope", "lum_entropy")
  ),
  coefficients = COEF, domain_intercept = as.list(INTERCEPT), segformer_offset = MODEL_OFFSET[["segformer"]],
  domain_shift = "D2/D3 metrics shifted (D3 darker, hazier, redder, less sharp), so D3 failures are more frequent",
  uncertainty = "entropy_fov, entropy_roi ~ exp(a + b * risk + noise), ens_dis ~ plogis(-3.3 + 5.2 * mean model risk + noise)",
  difficult_label = "Difficult frames shift the latent light, haze, blur, blood and glare factors",
  perturb = "IoU falls with severity^power per family; metrics move by family as in perturb_ops (sharpness down for defocus, up for noise; red_dom up for blood; spec_frac up for glare; ...)",
  perturb_baseline = "severity 0 rows are repeated once per family and equal the prediction row",
  stream = "4 videos x 120 frames, AR(1) latents with one smoke episode each",
  counts = list(labeled_patients = nrow(pat), labeled_frames = n, stream_frames = nrow(man_stream),
                perturb_frames = nsel, perturb_rows = nrow(perturb)),
  diagnostics = list(rankings = diag_tab, rates = rates)
)
jsonlite::write_json(planted, file.path(SYNTH_DIR, "planted_effects.json"), pretty = TRUE,
                     auto_unbox = TRUE, digits = 4, na = "null")

cat("synthetic files in", SYNTH_DIR, "\n")
cat(sprintf("labeled patients %d (C1 %d, C2 %d), labeled frames %d, stream frames %d, perturb rows %d\n",
            nrow(pat), sum(pat$center == 1), sum(pat$center == 2), n, nrow(man_stream), nrow(perturb)))
print(as.data.frame(rates), digits = 3)
print(as.data.frame(diag_tab), digits = 3)
