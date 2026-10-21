# Tests for T18b: 20_tables.R (Table 1 panel B and the combined Table 1, Table 2, Table 3).
#   Rscript v1/R/tests/test_tables.R        (re-runs itself with OPTICS_DATA=synthetic)
# Builds every table from the synthetic data in work/synthetic/ (outputs under work/synthetic/out/), recomputes panel B
# straight from the parquet files, compares each csv cell with the CSV it was formatted from, reads every .docx back, and
# stress-tests the branches the synthetic stats do not reach (descriptive-only AUROC, no margin, ridge odds ratios, no VIF
# flag, a covariate with the largest VIF) on mutated copies under work/synthetic/out/work/tests/tables, and checks the
# real-mode lock gate of panel B on a scratch real-like root (SYNTH and p() are mocked and restored, predictions.parquet
# there is not a parquet file). Thresholds, counts and the margin are read from the synthetic lock, data and CSVs, so a
# regenerated synthetic lock (for instance at 0.90, no margin) does not break it. No real exhibit is touched.

if (!identical(Sys.getenv("OPTICS_DATA"), "synthetic")) {
  .f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  quit(status = system2(file.path(R.home("bin"), "Rscript"), normalizePath(.f[1]), env = "OPTICS_DATA=synthetic"))
}
.f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
.here <- if (length(.f)) dirname(normalizePath(.f[1])) else file.path(getwd(), "v1", "R", "tests")
R_DIR <- dirname(.here)
for (f in c("00_setup.R", "01_load.R", "results_lib.R", "02_cohort.R", "20_tables.R")) source(file.path(R_DIR, f))

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
quiet <- function(expr) suppressMessages(suppressWarnings(expr))

# only the real files this script could clobber (other tasks write into the same folders while it runs)
real_owned <- function() { f <- list.files(file.path(V1, c("tables", file.path("results", "parts"))), full.names = TRUE)
  f <- f[grepl("^Table[123]|^20_tables\\.json$", basename(f))]; setNames(file.mtime(f), basename(f)) }
real_before <- real_owned()

stats_dir <- p("results", "stats")
out_tables <- p("tables")
scratch <- p("work", "tests", "tables")
unlink(scratch, recursive = TRUE)
dir.create(scratch, recursive = TRUE)

# ---- build ------------------------------------------------------------------------------------------------------
coh <- quiet(compute_cohort())
unlink(p("results", "parts", "20_tables.json"))
built <- quiet(capture.output(res <- build_tables(coh)))
check("build_tables wrote Table1 (panels A and B), Table1_panelA|B, Table2 and Table3 as docx and csv",
      all(file.exists(file.path(out_tables, paste0(rep(c("Table1", "Table1_panelA", "Table1_panelB", "Table2", "Table3"), each = 2), c(".docx", ".csv"))))))
check("every output path stays under work/synthetic/out", all(startsWith(normalizePath(unlist(res)), normalizePath(file.path(SYNTH_DIR, "out")))))

rd <- function(path) utils::read.csv(path, colClasses = "character", check.names = FALSE, na.strings = character())
t1 <- rd(file.path(out_tables, "Table1.csv")); t1a <- rd(file.path(out_tables, "Table1_panelA.csv"))
t1b <- rd(file.path(out_tables, "Table1_panelB.csv")); t2 <- rd(file.path(out_tables, "Table2.csv")); t3 <- rd(file.path(out_tables, "Table3.csv"))
no_na <- function(df) !any(grepl("(^|[^A-Za-z])NA([^A-Za-z]|$)|NaN|Inf", unlist(df))) && !anyNA(df)
num <- function(x) as.numeric(gsub(",", "", x))
P_FAIL <- "Patients with at least one failure, n"
fmt3 <- function(x) formatC(round(x, 3), format = "f", digits = 3, big.mark = ",")

# ---- Table 1 panel B: recomputed from the parquet files ---------------------------------------------------------
pr <- arrow::read_parquet(file.path(SYNTH_DIR, "predictions.parquet"))
fv <- arrow::read_parquet(file.path(SYNTH_DIR, "fov.parquet"))
fo <- utils::read.csv(file.path(SYNTH_DIR, "folds.csv"), stringsAsFactors = FALSE)
pr <- pr[pr$frame_id %in% fv$frame_id[fv$fov_found], ]
pr$domain <- fo$domain[match(pr$patient_id, fo$patient_id)]
lock_syn <- yaml::read_yaml(p("config", "analysis_lock.yaml"))
THR <- lock_failure_threshold(lock_syn)
MARGIN <- lock_syn$noninferiority_margin_aurc
thr_txt <- fmt_num(THR, 2)
expect_b <- function(model, rows) {
  d <- pr[pr$model == model & rows(pr), ]
  q <- function(x) stats::quantile(x, c(0.5, 0.25, 0.75), names = FALSE, type = 7)
  f <- function(x) sprintf("%s [%s to %s]", fmt3(q(x)[1]), fmt3(q(x)[2]), fmt3(q(x)[3]))
  c(format(nrow(d), big.mark = ","), as.character(length(unique(d$patient_id))), f(d$iou), f(d$dice),
    sprintf("%s (%s)", format(sum(d$iou < THR), big.mark = ","), formatC(100 * mean(d$iou < THR), format = "f", digits = 1)),
    as.character(length(unique(d$patient_id[d$iou < THR]))))
}
parts_rows <- list(function(d) d$partition == "center1_labeled", function(d) d$partition == "center1_labeled" & d$domain == "D1",
                   function(d) d$partition == "center1_labeled" & d$domain == "D2", function(d) d$partition == "center2_labeled")
want <- rbind(do.call(rbind, lapply(parts_rows, function(r) expect_b("unet", r))), do.call(rbind, lapply(parts_rows, function(r) expect_b("segformer", r))))
body_rows <- which(t1b$Partition != "" & t1b$`Frames, n` != "")
check("Table 1B has 7 columns and 1 heading + 2 model rows + 8 data rows (2 models x 4 partitions)",
      ncol(t1b) == 7 && nrow(t1b) == 11 && length(body_rows) == 8)
check("Table 1B: frames, patients, median [IQR] of IoU and Dice, failures n (%), patients with a failure equal the recomputation from predictions.parquet",
      identical(unname(as.matrix(t1b[body_rows, -1])), unname(want)))
check("Table 1B rows: training center (all) / D1 / D2 / external center for U-Net then SegFormer, under bold model rows; columns spell out the interquartile range and 'at least one failure'",
      identical(t1b$Partition[c(2, 7)], c("U-Net", "SegFormer")) && all(grepl("^Training center, all \\(D1 and D2\\)$", t1b$Partition[c(3, 8)])) &&
        all(grepl("^D1", t1b$Partition[c(4, 9)])) && all(grepl("^D2", t1b$Partition[c(5, 10)])) && all(grepl("^External center \\(D3, ", t1b$Partition[c(6, 11)])) &&
        identical(names(t1b), c("Partition", "Frames, n", "Patients, n", "Median IoU [interquartile range]", "Median Dice [interquartile range]",
                                "Failures, n (%)", P_FAIL)))
check("Table 1B patients: D1 + D2 patients add up to the Center 1 row; Center 1 frames add up too",
      all(num(t1b$`Patients, n`[c(4, 9)]) + num(t1b$`Patients, n`[c(5, 10)]) == num(t1b$`Patients, n`[c(3, 8)])) &&
        all(num(t1b$`Frames, n`[c(4, 9)]) + num(t1b$`Frames, n`[c(5, 10)]) == num(t1b$`Frames, n`[c(3, 8)])))
check("Table 1B analysed frames equal the cohort's analysed frames (Table 1A / 02_cohort exclusions)",
      num(t1b$`Frames, n`[3]) == coh$counts$n_frames_c1_analysed && num(t1b$`Frames, n`[6]) == coh$counts$n_frames_c2_analysed &&
        num(t1b$`Patients, n`[3]) + num(t1b$`Patients, n`[6]) == coh$counts$n_patients_labeled)
gi <- rd(file.path(stats_dir, "gate_internal.csv")); ge <- rd(file.path(stats_dir, "gate_external.csv")); gt <- rd(file.path(stats_dir, "gate_transfer.csv"))
ev <- function(tab, part) c(tab$n_events[tab$method == "random" & tab$partition == part], tab$n_patients_with_events[tab$method == "random" & tab$partition == part])
cell_fail <- function(i) c(num(sub(" .*", "", t1b$`Failures, n (%)`[i])), num(t1b[[P_FAIL]][i]))
check("Table 1B failure events and patients with a failure equal the gate CSVs' counts (U-Net: gate_internal/external, SegFormer: gate_transfer)",
      identical(cell_fail(3), num(ev(gi, "internal"))) && identical(cell_fail(6), num(ev(ge, "external"))) &&
        identical(cell_fail(8), num(ev(gt, "internal"))) && identical(cell_fail(11), num(ev(gt, "external"))))
check("Table 1B has no NA, NaN or Inf in any cell or footnote", no_na(t1b) && no_na(as.list(table1_panel_b(coh)$footnote)))

tb <- quiet(table1_panel_b(coh))
mf <- arrow::read_parquet(file.path(SYNTH_DIR, "manifest.parquet"))
mf <- mf[mf$labeled, ]; mf$found <- fv$fov_found[match(mf$frame_id, fv$frame_id)]
big <- function(x) format(x, big.mark = ",", trim = TRUE)
excl_txt <- sprintf("%s of %s training-center frames and %s of %s external-center frames (%s of %s annotated frames analyzed)",
                    big(sum(mf$center == 1 & !mf$found)), big(sum(mf$center == 1)), big(sum(mf$center == 2 & !mf$found)), big(sum(mf$center == 2)),
                    big(sum(mf$found)), big(nrow(mf)))
check("table1_panel_b() returns df, bold/indent rows, footnote and the numeric summary (8 rows)",
      is.data.frame(tb$df) && identical(tb$bold_rows, c(1L, 2L, 7L)) && length(tb$indent_rows) == 8L && nrow(tb$summary) == 8L &&
        any(grepl(excl_txt, tb$footnote, fixed = TRUE)) && any(grepl(paste0("below ", thr_txt), tb$footnote, fixed = TRUE)))
check("Table 1B footnote calls the threshold 'the frozen failure threshold' when the lock carries one, and names the plan when it does not (fallback); it also defines Dice, says patients with at least one failure are counted without a rate, and has no IQR abbreviation",
      any(grepl(paste0("below ", thr_txt, " (the frozen failure threshold)"), tb$footnote, fixed = TRUE)) && !any(grepl("analysis plan|analysis lock", tb$footnote)) &&
        any(grepl("Dice overlap = twice the shared area over the summed areas", tb$footnote, fixed = TRUE)) &&
        any(grepl("Patients with at least one failure are counted; no per-patient rate is given.", tb$footnote, fixed = TRUE)) &&
        !any(grepl("IQR|Center [12]", tb$footnote)) &&
        { tp <- quiet(table1_panel_b(coh, lock = list())); tp$threshold == CFG$endpoints$failure_iou_threshold &&
            any(grepl("the failure threshold in the analysis plan; the analysis lock carries none", tp$footnote, fixed = TRUE)) &&
            !any(grepl("frozen", tp$footnote)) })
ftab <- quiet(load_frame_table(with_predictions = TRUE))
check("tbl_check_seg_frames accepts the synthetic prediction table", is.data.frame(quiet(tbl_check_seg_frames(ftab))))
bad_src <- ftab; bad_src$unet_source[which(bad_src$center == 1L)[1]] <- "final"
check("tbl_check_seg_frames refuses Center 1 rows that are not out-of-fold", errs(tbl_check_seg_frames(bad_src), "out-of-fold"))
bad_src2 <- ftab; bad_src2$segformer_source[which(bad_src2$center == 2L)[1]] <- "oof"
check("tbl_check_seg_frames refuses Center 2 rows that are not from the final model", errs(tbl_check_seg_frames(bad_src2), "final model"))
bad_na <- ftab; bad_na$segformer_dice[5] <- NA
check("tbl_check_seg_frames refuses an analysed frame without a prediction (NA Dice)", errs(tbl_check_seg_frames(bad_na), "without a prediction row: 1"))
check("panel B refuses a frame count that disagrees with the cohort",
      errs(quiet(table1_panel_b({ c2 <- coh; c2$counts$n_frames_c1_analysed <- c2$counts$n_frames_c1_analysed + 1L; c2 })), "analysed frames"))
check("threshold: the lock's failure_iou_threshold is what the footnote and the counts use; a different threshold changes both",
      { tb2 <- quiet(table1_panel_b(coh, thr = 0.5)); tb2$summary$fail_n[1] < tb$summary$fail_n[1] && any(grepl("below 0.50", tb2$footnote)) })

# ---- real-mode lock gate of panel B ---------------------------------------------------------------------------------------
# On real data panel B (which opens predictions.parquet, i.e. Center 2 outcomes) must wait for a sealed lock. This mocks
# SYNTH = FALSE and p() onto a scratch real-like root, so no real path is read or written, and replaces every loader of
# predictions with a counter; predictions.parquet there is not a parquet file, so any real read would also fail.
real_like <- file.path(scratch, "real_like")
for (d in c("config", "derived", "tables", "work/models")) dir.create(file.path(real_like, d), recursive = TRUE)
writeLines("not a parquet file", file.path(real_like, "derived", "predictions.parquet"))
invisible(file.copy(list.files(p("work", "models"), full.names = TRUE), file.path(real_like, "work", "models")))
write_lock <- function(lock) yaml::write_yaml(lock, file.path(real_like, "config", "analysis_lock.yaml"))
opened <- 0L
with_real_like <- function(expr) {
  keep <- mget(c("SYNTH", "p", "tbl_seg_frames", "load_frame_table", "load_predictions"), envir = globalenv())
  on.exit(list2env(keep, envir = globalenv()))
  assign("SYNTH", FALSE, envir = globalenv())
  assign("p", function(...) file.path(real_like, ...), envir = globalenv())
  for (fn in c("tbl_seg_frames", "load_frame_table", "load_predictions")) {
    assign(fn, function(...) { opened <<- opened + 1L; stop("mock: predictions were opened", call. = FALSE) }, envir = globalenv())
  }
  force(expr)
}
msgs_of <- function(expr) { m <- character(); withCallingHandlers(suppressWarnings(capture.output(force(expr))), message = function(e) { m <<- c(m, conditionMessage(e)); invokeRestart("muffleMessage") }); m }
unsealed <- lock_syn; unsealed$lock_fields_sha256 <- NULL
write_lock(unsealed)
msgs <- with_real_like(msgs_of(build_tables(coh)))
check("real mode, lock without lock_fields_sha256: panel B is skipped with a message that names lock_fields_sha256, predictions.parquet is never opened",
      any(grepl("table1_panel_b: skipped, the analysis lock has no lock_fields_sha256", msgs, fixed = TRUE)) && opened == 0L)
check("real mode, unsealed lock: only Table 1 panel A is written (no Table1.docx, no panel B files, no 20_tables part)",
      file.exists(file.path(real_like, "tables", "Table1_panelA.docx")) && !any(file.exists(file.path(real_like, "tables", c("Table1.docx", "Table1_panelB.docx", "Table1_panelB.csv")))) &&
        !file.exists(file.path(real_like, "results", "parts", "20_tables.json")))
check("real mode, unsealed lock: table1_panel_b() itself refuses before reading any frame",
      errs(with_real_like(table1_panel_b(coh)), "not built: the analysis lock has no lock_fields_sha256") && opened == 0L)
check("real mode, no lock file at all: also refused, nothing opened",
      { unlink(file.path(real_like, "config", "analysis_lock.yaml")); errs(with_real_like(table1_panel_b(coh)), "lock_fields_sha256") && opened == 0L })
bad <- lock_syn; bad$lock_fields_sha256 <- "deadbeef"; write_lock(bad)
check("real mode, lock_fields_sha256 present but wrong: verify_lock_hashes stops it before any frame is read",
      errs(with_real_like(table1_panel_b(coh)), "lock_fields_sha256 mismatch") && opened == 0L)
good <- lock_syn
for (o in GATE_LOCK_OBJECTS) good[[paste0(o, "_sha256")]] <- unname(tools::sha256sum(file.path(real_like, good[[paste0(o, "_path")]])))
good$lock_fields_sha256 <- lock_fields_hash(good); write_lock(good)
check("real mode, sealed and verified lock: panel B proceeds to load the frames (positive control: the mocked loader is reached)",
      errs(with_real_like(table1_panel_b(coh)), "mock: predictions were opened") && opened == 1L &&
        is.null(with_real_like(tbl_panel_b_blocked(good))))
check("synthetic lock without lock_fields_sha256 is still accepted in synthetic mode (panel B built above from the synthetic predictions)",
      is.null(tbl_panel_b_blocked(unsealed)) && is.data.frame(tb$df))
unlink(real_like, recursive = TRUE)

# ---- registered keys --------------------------------------------------------------------------------------------
part_path <- p("results", "parts", "20_tables.json")
check("results/parts/20_tables.json written under work/synthetic/out", file.exists(part_path))
pj <- read_part(part_path)$entries
keys <- names(pj)
check("88 keys: 2 models x 4 partitions x 11 statistics, all seg_<partition>_<model>_<stat>, all unique",
      length(keys) == 88L && !anyDuplicated(keys) && all(grepl("^seg_(c1|c1_d1|c1_d2|c2)_(unet|segformer)_[a-z0-9_]+$", keys)))
other <- unlist(lapply(setdiff(list.files(p("results", "parts"), pattern = "\\.json$", full.names = TRUE), part_path), function(f) names(read_part(f)$entries)))
check("no key collides with any other results part", length(intersect(keys, other)) == 0L)
disp <- function(k) pj[[k]]$display
check("registered displays equal the table cells (U-Net Center 2: median IoU, Dice, failures n and %, patients)",
      disp("seg_c2_unet_iou_median") == sub(" .*", "", t1b$`Median IoU [interquartile range]`[6]) &&
        sprintf("%s [%s to %s]", disp("seg_c2_unet_dice_median"), disp("seg_c2_unet_dice_q1"), disp("seg_c2_unet_dice_q3")) == t1b$`Median Dice [interquartile range]`[6] &&
        sprintf("%s (%s)", disp("seg_c2_unet_fail_n"), sub("%$", "", disp("seg_c2_unet_fail_share"))) == t1b$`Failures, n (%)`[6] &&
        disp("seg_c2_unet_fail_patients") == t1b[[P_FAIL]][6] && disp("seg_c2_unet_n_frames") == t1b$`Frames, n`[6])
check("registered displays equal the cells for every model x partition x statistic",
      all(vapply(seq_len(nrow(tb$summary)), function(i) {
        r <- tb$summary[i, ]; row <- t1b[body_rows[i], ]
        k <- function(s) disp(paste("seg", r$part, r$model, s, sep = "_"))
        k("n_frames") == row$`Frames, n` && k("n_patients") == row$`Patients, n` &&
          sprintf("%s [%s to %s]", k("iou_median"), k("iou_q1"), k("iou_q3")) == row$`Median IoU [interquartile range]` &&
          sprintf("%s [%s to %s]", k("dice_median"), k("dice_q1"), k("dice_q3")) == row$`Median Dice [interquartile range]` &&
          sprintf("%s (%s)", k("fail_n"), sub("%$", "", k("fail_share"))) == row$`Failures, n (%)` && k("fail_patients") == row[[P_FAIL]]
      }, NA)))
check("IoU and Dice keys have 3 digits, shares are percent with 1 digit", grepl("^[0-9]\\.[0-9]{3}$", disp("seg_c1_unet_iou_q1")) && grepl("^[0-9.]+%$", disp("seg_c1_unet_fail_share")) &&
        grepl("^[0-9]+\\.[0-9]%$", disp("seg_c1_unet_fail_share")))
check("a second registration of the same keys is refused (reg duplicates)", errs({ reg("seg_c1_unet_n_frames", 1L); reg("seg_c1_unet_n_frames", 2L) }, "duplicate"))
reg_reset()

# ---- combined Table 1 -------------------------------------------------------------------------------------------
check("Table1.csv = panel A rows, then panel B column names as a row, then panel B rows, padded to the wider panel",
      ncol(t1) == 7 && nrow(t1) == nrow(t1a) + 1L + nrow(t1b) &&
        identical(unname(as.matrix(t1[seq_len(nrow(t1a)), 1:5])), unname(as.matrix(t1a))) &&
        identical(unname(unlist(t1[nrow(t1a) + 1L, ])), names(t1b)) &&
        identical(unname(as.matrix(t1[nrow(t1a) + 1L + seq_len(nrow(t1b)), ])), unname(as.matrix(t1b))))

# ---- docx read-back -----------------------------------------------------------------------------------------------
docx_rows <- function(path) {
  s <- officer::docx_summary(officer::read_docx(path))
  tc <- s[s$content_type == "table cell", ]
  lapply(split(tc, tc$row_id), function(r) r$text[order(r$cell_id)])
}
norm <- function(x) trimws(gsub("[[:space:];]+", " ", x))
# first row whose first cell is `first` -> body rows (csv row i) follow; merged heading rows carry only their first cell
rows_equal <- function(drows, df, first) {
  at <- which(vapply(drows, function(r) identical(r[1], first), NA))[1]
  if (is.na(at)) return(FALSE)
  all(vapply(seq_len(nrow(df)), function(i) {
    got <- drows[[at + i]]; exp <- unname(unlist(df[i, ]))
    if (length(got) == 1L && all(exp[-1] == "")) norm(got) == norm(exp[1]) else length(got) == length(exp) && identical(norm(got), norm(exp))
  }, NA))
}
d1 <- docx_rows(file.path(out_tables, "Table1.docx"))
check("Table1.docx holds panel A and panel B: A's first body row and B's header are both there, B rows read back equal Table1_panelB.csv",
      rows_equal(d1, t1a, "Characteristic") && rows_equal(d1, t1b, "Partition"))
check("Table1.docx: panel B starts on a new page (page break between the two tables)",
      { x <- tempfile(tmpdir = scratch); dir.create(x); unzip(file.path(out_tables, "Table1.docx"), exdir = x)
        grepl('w:br w:type="page"', paste(readLines(file.path(x, "word", "document.xml"), warn = FALSE), collapse = "")) })
check("Table1_panelB.docx read back equals Table1_panelB.csv", rows_equal(docx_rows(file.path(out_tables, "Table1_panelB.docx")), t1b, "Partition"))

# ---- Table 2 --------------------------------------------------------------------------------------------------------
a2 <- rd(file.path(stats_dir, "assoc_table2.csv")); rho <- utils::read.csv(file.path(stats_dir, "perturb_rho.csv"), stringsAsFactors = FALSE)
check("Table 2: 11 metric rows and 8 columns, in the plan's metric order", nrow(t2) == 11L && ncol(t2) == 8L && identical(t2$Metric, unname(METRIC_LABELS[OPTICAL_METRICS])))
check("Table 2: dictionary text has no digits and American spelling", !any(grepl("[0-9]", t2[[2]])) && !any(grepl("colour|centre|towards", t2[[2]])))
check("Table 2: headers spell out 'interquartile range'; family names are those of Figure 2 (PERTURB_FAMILY_LABELS)",
      identical(names(t2)[3:4], c("Normal, median [interquartile range]", "Difficult, median [interquartile range]")) &&
        identical(TBL_FAMILY_LABELS, PERTURB_FAMILY_LABELS))
# independent "4.1e-4" for the sharpness row: "4.11e-04 (2.76e-04 to 5.97e-04)" -> "4.1e-4 [2.8e-4 to 6.0e-4]"
sci1 <- function(z) sub("e-0", "e-", z, fixed = TRUE)
sci_fix <- function(z) vapply(z, function(v) { m <- gregexpr("[0-9.]+e-[0-9]+", v); regmatches(v, m) <- lapply(regmatches(v, m), function(t) sci1(formatC(as.numeric(t), format = "e", digits = 1))); v }, "", USE.NAMES = FALSE)
brk <- function(z) sub("^([^ ]+) \\((.+)\\)$", "\\1 [\\2]", z)
ix <- match(OPTICAL_METRICS, a2$metric)
check("Table 2: AUROC cells are assoc_table2.csv's auroc_text; median [interquartile range] cells are its strings with brackets, sharpness in the 4.1e-4 notation of Figure 1B and the Figure 2A axis",
      identical(t2[[5]], a2$auroc_text[ix]) &&
        identical(t2[[3]], sci_fix(brk(a2$normal_median_iqr[ix]))) && identical(t2[[4]], sci_fix(brk(a2$difficult_median_iqr[ix]))) &&
        grepl("^[0-9]\\.[0-9]e-[0-9] \\[[0-9]\\.[0-9]e-[0-9] to [0-9]\\.[0-9]e-[0-9]\\]$", t2[[3]][4]) && !any(grepl("e-0", unlist(t2))))
or_cell <- t2[[6]]; names(or_cell) <- OPTICAL_METRICS
ridge_m <- a2$metric[a2$or_method != "cluster-robust logistic"][order(match(a2$metric[a2$or_method != "cluster-robust logistic"], OPTICAL_METRICS))]
check("Table 2: OR cells are or_text; only sharpness carries the log-scale mark and the ridge mark is on exactly the rows whose or_method is not cluster-robust",
      identical(unname(sub("[‡¶]+$", "", or_cell)), a2$or_text[match(OPTICAL_METRICS, a2$metric)]) &&
        identical(names(or_cell)[grepl("‡", or_cell)], "sharp_tenengrad") && identical(names(or_cell)[grepl("¶", or_cell)], ridge_m))
rho_txt <- function(f, m) { r <- rho[rho$family == f & rho$metric == m, ]; sprintf("%s (%s to %s)", fmt_num(r$rho, 2), fmt_num(r$rho_lo, 2), fmt_num(r$rho_hi, 2)) }
fam_lab <- function(f) TBL_FAMILY_LABELS[[f]]
exp_pairs <- rho[rho$expected_pair, ]
ok_rho <- vapply(seq_along(OPTICAL_METRICS), function(i) {
  m <- OPTICAL_METRICS[i]; e <- exp_pairs[exp_pairs$metric == m, ]
  if (nrow(e)) {
    e <- e[order(match(e$family, names(TBL_FAMILY_LABELS))), ]
    identical(strsplit(t2[[8]][i], "; ")[[1]], vapply(e$family, function(f) rho_txt(f, m), "", USE.NAMES = FALSE)) &&
      identical(strsplit(t2[[7]][i], "; ")[[1]], paste0(vapply(e$family, fam_lab, ""), ifelse(e$known_limitation, "†", "")))
  } else {
    r <- rho[rho$metric == m, ]; top <- r[which.max(abs(r$rho)), ]
    identical(t2[[8]][i], rho_txt(top$family, m)) && identical(t2[[7]][i], paste0(fam_lab(top$family), "§"))
  }
}, NA)
check("Table 2: perturbation cells are the expected families' rho (95% CI) from perturb_rho.csv, the largest |rho| family with a section mark where none is expected, the noise-sharpness cell with the dagger (the mark Figure 2C uses for the known limitation)",
      all(ok_rho))
check("Table 2: two families for luminance (under, over), sharpness (defocus, noise), saturated fraction (over, glare); none expected for illumination and entropy",
      identical(lengths(strsplit(t2[[7]], "; ")), c(2L, 1L, 1L, 2L, 1L, 2L, 1L, 1L, 1L, 1L, 1L)) && all(grepl("§", t2[[7]][c(10, 11)])) && !any(grepl("§", t2[[7]][-c(10, 11)])) && sum(grepl("†", t2[[7]])) == 1L)
t2obj <- quiet(table2())
vf <- utils::read.csv(file.path(stats_dir, "assoc_vif.csv"), stringsAsFactors = FALSE)
vf_m <- vf[vf$scope == "center1_oof" & vf$term %in% OPTICAL_METRICS, ]; vf_top <- vf_m[which.max(vf_m$vif), ]
bd <- utils::read.csv(file.path(stats_dir, "assoc_by_domain.csv"), stringsAsFactors = FALSE)
bd2 <- bd[bd$scope == "center1_oof" & bd$domain == "D2" & bd$metric == OPTICAL_METRICS[1], ]
dg_all <- utils::read.csv(file.path(stats_dir, "assoc_diagnostics.csv"), stringsAsFactors = FALSE)
aliased_txt <- paste(unique(stats::na.omit(dg_all$aliased_terms[dg_all$scope == "center1_oof" & dg_all$model == "univariable_adjusted"])), collapse = "; ")
n_ev <- dg_all$events[dg_all$scope == "center1_oof" & dg_all$model == "univariable_adjusted"][1]
a2n <- a2[1, ]
check("Table 2 footnote: training-center counts and failures from the CSVs, adjustment set, perturbation frames, dagger (known limitation) / double dagger (log scale) / section mark (largest |rho| fallback) in that order, ridge footnote only with ridge rows, Supplementary Table S1 for within-domain values, OR, SD and rho spelled out",
      any(grepl(sprintf("training-center out-of-fold frames: %s Normal and %s Difficult frames, with %s U-Net failures (IoU below %s, the frozen failure threshold)", big(as.numeric(a2n$n_normal)), big(as.numeric(a2n$n_difficult)), big(n_ev), thr_txt), t2obj$footnote, fixed = TRUE)) &&
        any(grepl("within-domain AUROC values are given in Supplementary Table S1", t2obj$footnote, fixed = TRUE)) && !any(grepl("analysis files|IQR", t2obj$footnote)) &&
        any(grepl("OR, odds ratio; SD, standard deviation; rho, Spearman rank correlation coefficient", t2obj$footnote, fixed = TRUE)) &&
        identical(substr(t2obj$footnote[grepl("^[†‡§]", t2obj$footnote)], 1, 1), c("†", "‡", "§")) &&
        any(grepl("domain, channel, region, image format, instrument class", t2obj$footnote)) &&
        any(grepl(sprintf("%s frames from %s patients", big(max(rho$n_frames)), big(max(rho$n_patients))), t2obj$footnote, fixed = TRUE)) &&
        any(startsWith(t2obj$footnote, "†")) && any(startsWith(t2obj$footnote, "‡")) && any(startsWith(t2obj$footnote, "§")) &&
        identical(any(startsWith(t2obj$footnote, "¶")), length(ridge_m) > 0L) &&
        (!nzchar(aliased_txt) || any(grepl(aliased_txt, t2obj$footnote, fixed = TRUE))))
check("Table 2 footnote: the Difficult share of D2 comes from the analysed frames in assoc_by_domain.csv, not from the cohort that still counts the excluded frames",
      nrow(bd2) == 1L && any(grepl(sprintf("(%s%% of D2 frames are Difficult)", fmt_num(100 * bd2$n_difficult / (bd2$n_normal + bd2$n_difficult), 1)), t2obj$footnote, fixed = TRUE)) &&
        { g2 <- coh$groups[coh$groups$group == "D2", ]; bd2$n_normal + bd2$n_difficult <= g2$n_frames })
check("Table 2 footnote: VIF caveat uses the largest VIF among the optical metrics only, with the flag level from the plan (statistics.vif_flag)",
      (vf_top$vif > CFG$statistics$vif_flag) == any(grepl("variance inflation", t2obj$footnote)) &&
        (vf_top$vif <= CFG$statistics$vif_flag || any(grepl(sprintf("among the optical metrics was %s (%s; flag level %s)", fmt_num(vf_top$vif, 1), METRIC_LABELS[[vf_top$term]], fmt_num(CFG$statistics$vif_flag, 0)), t2obj$footnote, fixed = TRUE))))
check("Table 2 footnote: CI wording (AUROC patient-clustered bootstrap percentile interval; odds ratios cluster-robust; rho patient-clustered bootstrap)",
      any(grepl("The AUROC interval is a patient-clustered bootstrap percentile interval", t2obj$footnote, fixed = TRUE)) &&
        any(grepl("odds-ratio intervals are cluster-robust by patient", t2obj$footnote, fixed = TRUE)) &&
        any(grepl("rho intervals are patient-clustered bootstrap intervals", t2obj$footnote, fixed = TRUE)))
check("Table 2 has no NA, NaN or Inf in cells or footnote", no_na(t2) && no_na(as.list(t2obj$footnote)))
check("Table2.docx read back: header names (line breaks as spaces) and 11 rows equal Table2.csv",
      identical(norm(docx_rows(file.path(out_tables, "Table2.docx"))[[1]]), norm(names(t2))) && rows_equal(docx_rows(file.path(out_tables, "Table2.docx")), t2, "Metric"))

# stress copy: ridge ORs, no VIF above the flag, no known limitation, one metric not estimable
copy_stats <- function(tag) {
  d <- file.path(scratch, tag); dir.create(d, recursive = TRUE)
  file.copy(list.files(stats_dir, full.names = TRUE), d)
  d
}
sd2 <- copy_stats("t2")
x <- utils::read.csv(file.path(sd2, "assoc_table2.csv"), stringsAsFactors = FALSE, check.names = FALSE)
x$or_method[c(2, 5)] <- "ridge logistic"; x$or_text[7] <- "not estimable"
utils::write.csv(x, file.path(sd2, "assoc_table2.csv"), row.names = FALSE)
v <- utils::read.csv(file.path(sd2, "assoc_vif.csv")); v$vif <- 2; utils::write.csv(v, file.path(sd2, "assoc_vif.csv"), row.names = FALSE)
r <- utils::read.csv(file.path(sd2, "perturb_rho.csv")); r$known_limitation <- FALSE; utils::write.csv(r, file.path(sd2, "perturb_rho.csv"), row.names = FALSE)
t2s <- quiet(table2(stats_dir = sd2))
check("Table 2 stress: ridge rows carry the pilcrow and its footnote, 'not estimable' is kept, no VIF line below the flag, no known-limitation (dagger) mark",
      sum(grepl("¶", t2s$df[[6]])) == sum(x$or_method[match(OPTICAL_METRICS, x$metric)] != "cluster-robust logistic") &&
        all(grepl("¶", t2s$df[[6]][c(2, 5)])) && t2s$df[[6]][7] == "not estimable" && any(startsWith(t2s$footnote, "¶")) &&
        !any(grepl("variance inflation", t2s$footnote)) && !any(grepl("†", t2s$df[[7]])) && !any(startsWith(t2s$footnote, "†")) && any(startsWith(t2s$footnote, "§")))
check("Table 2 refuses a missing input with a message that names the file", errs(quiet(table2(stats_dir = file.path(scratch, "nowhere"))), "missing input .*assoc_table2.csv"))

# a covariate (not an optical metric) holding the largest VIF must not set the caveat; a metric above the flag must
sdv <- copy_stats("vif")
vv <- utils::read.csv(file.path(sdv, "assoc_vif.csv"), stringsAsFactors = FALSE)
vv$vif[vv$term %in% OPTICAL_METRICS] <- 2; vv$vif[vv$term == "domainD2"] <- 50
utils::write.csv(vv, file.path(sdv, "assoc_vif.csv"), row.names = FALSE)
check("Table 2 VIF: a covariate with the largest VIF (domainD2 = 50, every metric = 2) raises no caveat",
      !any(grepl("variance inflation", quiet(table2(stats_dir = sdv))$footnote)))
vv$vif[vv$term == "lum_mean"] <- 7.04
utils::write.csv(vv, file.path(sdv, "assoc_vif.csv"), row.names = FALSE)
check("Table 2 VIF: with a metric above the flag and a covariate far above it, the caveat names the metric and its VIF, not the covariate",
      { fnv <- quiet(table2(stats_dir = sdv))$footnote; any(grepl("among the optical metrics was 7.0 (Mean luminance; flag level", fnv, fixed = TRUE)) && !any(grepl("50.0|domainD2", fnv)) })
vv$scope[vv$term == "lum_mean"] <- "center2_external"
utils::write.csv(vv, file.path(sdv, "assoc_vif.csv"), row.names = FALSE)
check("Table 2 VIF: only the center1_oof scope counts (a larger metric VIF in another scope is ignored)",
      !any(grepl("variance inflation", quiet(table2(stats_dir = sdv))$footnote)))
flag_old <- CFG$statistics$vif_flag
with_flag <- function(v, expr) { CFG$statistics$vif_flag <<- v; on.exit(CFG$statistics$vif_flag <<- flag_old); force(expr) }
fn_high <- with_flag(ceiling(vf_top$vif) + 1, quiet(table2())$footnote)
fn_low <- with_flag(1, quiet(table2())$footnote)
check("Table 2 VIF: the cut-off is read from config statistics.vif_flag (above the largest metric VIF: no caveat; set to 1: the caveat says flag level 1)",
      !any(grepl("variance inflation", fn_high)) && any(grepl("flag level 1)", fn_low, fixed = TRUE)) &&
        identical(CFG$statistics$vif_flag, flag_old) && !exists("TBL_VIF_FLAG"))

# ---- Table 3 --------------------------------------------------------------------------------------------------------
pu <- rd(file.path(stats_dir, "gate_paired_diffs.csv")); ps <- rd(file.path(stats_dir, "gate_transfer_paired_diffs.csv"))
sn <- rd(file.path(stats_dir, "sensitivity.csv"))
sn_row <- sn[sn$variant == "locked" & sn$method == "entropy", ]
check("Table 3: 11 columns (method + 5 statistics x training/external center, selective risk at 90% and at 80% coverage side by side); 27 rows = 2 models x (section + 6 methods + sub-heading + 5 differences) + the pixel entropy vs ensemble row",
      ncol(t3) == 11L && nrow(t3) == 27L &&
        identical(names(t3)[2:6], paste("Training center", c("AURC", "AUROC", "Selective risk at 90% coverage", "Selective risk at 80% coverage", "Calibration slope"))) &&
        identical(names(t3)[7:11], paste("External center", c("AURC", "AUROC", "Selective risk at 90% coverage", "Selective risk at 80% coverage", "Calibration slope"))))
ci_txt <- function(r, s, d) {
  e <- as.numeric(r[[s]]); lo <- as.numeric(r[[paste0(s, "_lo")]]); hi <- as.numeric(r[[paste0(s, "_hi")]])
  if (is.na(e)) "" else if (is.na(lo) || is.na(hi)) fmt_num(e, d) else sprintf("%s (%s to %s)", fmt_num(e, d), fmt_num(lo, d), fmt_num(hi, d))
}
dg <- c(aurc = 3, auroc = 2, sel_risk_90 = 3, sel_risk_80 = 3, cal_slope = 2)
tabs <- list(unet = rbind(gi, ge), segformer = gt)
rows_of <- function(m) if (m == "unet") 2:7 else 16:21
ok_cells <- vapply(names(tabs), function(m) {
  all(vapply(seq_along(TBL3_METHODS), function(i) {
    got <- unname(unlist(t3[rows_of(m)[i], -1]))
    exp <- unlist(lapply(c("internal", "external"), function(pt) {
      r <- tabs[[m]][tabs[[m]]$partition == pt & tabs[[m]]$method == TBL3_METHODS[i], ]
      vapply(names(dg), function(s) ci_txt(r, s, dg[[s]]), "")
    }))
    identical(got, unname(exp)) && identical(t3$Method[rows_of(m)[i]], TBL_METHOD_LABELS[[TBL3_METHODS[i]]])
  }, NA))
}, NA)
check("Table 3: every method row equals gate_internal/gate_external (U-Net) and gate_transfer (SegFormer) formatted est (lo to hi), AURC and both selective risks at 3 digits, AUROC and slope at 2", all(ok_cells))
check("Table 3: calibration slope is blank for random, pixel entropy and ensemble disagreement; random AUROC has no interval; selective risk at 80% is filled for every method",
      all(t3[c(2, 4, 5, 16, 18, 19), c(6, 11)] == "") && t3[2, 3] == "0.50" && t3[16, 3] == "0.50" && all(t3[c(3, 6, 7, 17, 20, 21), c(6, 11)] != "") &&
        all(t3[c(2:7, 16:21), c(5, 10)] != ""))
cmp_ids <- GATE_COMPARISONS$comparison
ok_diff <- vapply(c("unet", "segformer"), function(m) {
  dif <- if (m == "unet") pu else ps
  base <- if (m == "unet") 9L else 23L
  all(vapply(seq_along(cmp_ids), function(i) {
    got <- unname(unlist(t3[base + i - 1L, -1]))
    exp <- unlist(lapply(c("internal", "external"), function(pt) vapply(names(dg), function(s) {
      if (s %in% c("cal_slope", "sel_risk_80")) return("")
      r <- dif[dif$partition == pt & dif$comparison == cmp_ids[i] & dif$metric == s, ]
      sprintf("%s (%s to %s)", fmt_num(as.numeric(r$diff), dg[[s]]), fmt_num(as.numeric(r$lo), dg[[s]]), fmt_num(as.numeric(r$hi), dg[[s]]))
    }, "")))
    identical(got, unname(exp))
  }, NA))
}, NA)
check("Table 3: the five paired-difference rows per model equal gate_paired_diffs.csv and gate_transfer_paired_diffs.csv (AURC, AUROC, selective risk at 90%), selective risk at 80% and slope blank", all(ok_diff))
check("Table 3: difference row labels read 'first vs second' with 'Optical gate' as the method name",
      identical(t3$Method[9:13], c("Optical gate vs random", "Optical gate vs predicted area", "Optical gate vs pixel entropy",
        "Optical gate vs ensemble disagreement", "Gate + ensemble vs ensemble disagreement")) && identical(t3$Method[23:27], t3$Method[9:13]) &&
        identical(TBL_METHOD_LABELS[["optical_gate"]], "Optical gate") && !any(grepl("^Gate vs", t3$Method)))
check("Table 3: the 'Pixel entropy vs ensemble disagreement' row (U-Net block only) holds the external AURC difference from sensitivity.csv (locked variant, entropy row) and nothing else; its resamples are the table's (the locked entropy AURC equals gate_external.csv)",
      identical(t3$Method[14], "Pixel entropy vs ensemble disagreement") && nrow(sn_row) == 1L &&
        t3[14, 7] == sprintf("%s (%s to %s)", fmt_num(as.numeric(sn_row$diff_vs_ens), 3), fmt_num(as.numeric(sn_row$diff_vs_ens_lo), 3), fmt_num(as.numeric(sn_row$diff_vs_ens_hi), 3)) &&
        all(t3[14, -c(1, 7)] == "") && !any(grepl("ensemble disagreement$", t3$Method[22:27]) & grepl("^Pixel", t3$Method[22:27])) &&
        abs(as.numeric(sn_row$aurc) - as.numeric(ge$aurc[ge$method == "entropy"])) < 1e-9)
# registered gate keys must equal the table cells
p04b <- read_part(p("results", "parts", "04b_external_eval.json"))$entries
strip_ci <- function(x) sub("95% CI ", "", x, fixed = TRUE)
check("Table 3 cells equal the registered gate_ext_*/gate_int_* strings (AURC of the gate and of entropy, AUROC, selective risk at 90% and 80%, calibration slope)",
      strip_ci(p04b$gate_ext_aurc_optical_gate_ci$display) == t3[6, 7] && strip_ci(p04b$gate_int_aurc_entropy_ci$display) == t3[4, 2] &&
        strip_ci(p04b$gate_ext_auroc_gate_plus_ensemble_ci$display) == t3[7, 8] && strip_ci(p04b$gate_ext_sel90_predicted_area_ci$display) == t3[3, 9] &&
        strip_ci(p04b$gate_ext_sel80_predicted_area_ci$display) == t3[3, 10] && strip_ci(p04b$gate_int_sel80_optical_gate_ci$display) == t3[6, 5] &&
        strip_ci(p04b$gate_ext_calslope_optical_gate_ci$display) == t3[6, 11] &&
        strip_ci(p04b$gate_ext_diff_aurc_gate_vs_ens_ci$display) == t3[12, 7])
t3obj <- quiet(table3())
rnd <- function(tab, part) tab[tab$partition == part & tab$method == "random", ]
orc <- function(tab, part) fmt_num(as.numeric(tab$aurc[tab$partition == part & tab$method == "oracle"]), 3)
cnt_txt <- function(part, label) {
  u <- rnd(rbind(gi, ge), part); sf <- rnd(gt, part)
  sprintf("%s: %s frames from %s patients; U-Net %s failures in %s patients, SegFormer %s failures in %s patients", label,
          big(as.numeric(u$n_frames)), big(as.numeric(u$n_patients)), big(as.numeric(u$n_events)), big(as.numeric(u$n_patients_with_events)),
          big(as.numeric(sf$n_events)), big(as.numeric(sf$n_patients_with_events)))
}
ent_txt <- c(entropy_fov = "the whole field of view", entropy_roi = "the region around the predicted mask")[[lock_syn$entropy_baseline_choice]]
check("Table 3 footnote: failure threshold from the lock (frozen), bootstrap B from the plan, frames, patients and failures per partition and model from the CSVs, oracle AURC, entropy variant, AURC and AUROC spelled out, both selective risks defined, the entropy-vs-ensemble row explained",
      any(grepl(paste0("IoU below ", thr_txt, ", the frozen failure threshold"), t3obj$footnote, fixed = TRUE)) &&
        any(grepl("AURC, area under the risk-coverage curve", t3obj$footnote, fixed = TRUE)) && any(grepl("AUROC, area under the receiver operating characteristic curve", t3obj$footnote, fixed = TRUE)) &&
        any(grepl("Selective risk at 90% and 80% coverage, the mean risk of the 90% and 80% of frames ranked most reliable", t3obj$footnote, fixed = TRUE)) &&
        any(grepl("Paired differences cover AURC, AUROC and selective risk at 90% coverage (not at 80%).", t3obj$footnote, fixed = TRUE)) &&
        any(grepl("The pixel entropy vs ensemble disagreement row is for the external center and AURC only", t3obj$footnote, fixed = TRUE)) &&
        any(grepl(paste0(big(CFG$statistics$bootstrap_resamples), " resamples"), t3obj$footnote, fixed = TRUE)) &&
        any(grepl("Training-center scores are cross-fitted (out-of-fold); external-center frames are scored by the locked gate.", t3obj$footnote, fixed = TRUE)) &&
        any(grepl(cnt_txt("internal", "Training center"), t3obj$footnote, fixed = TRUE)) && any(grepl(cnt_txt("external", "External center"), t3obj$footnote, fixed = TRUE)) &&
        any(grepl(sprintf("Pixel entropy, the mean binary entropy of the predicted foreground probability, is averaged over %s", ent_txt), t3obj$footnote, fixed = TRUE)) &&
        any(grepl(sprintf("an AURC of %s internal and %s external", orc(gi, "internal"), orc(ge, "external")), t3obj$footnote, fixed = TRUE)))
check("Table 3 footnote: the intervals are 'patient-clustered bootstrap, 95% percentile intervals'",
      any(grepl("Intervals are patient-clustered bootstrap, 95% percentile intervals (", t3obj$footnote, fixed = TRUE)) && !any(grepl("95% CIs from", t3obj$footnote)))
verdict_txt <- function(cmpid, part) { r <- pu[pu$comparison == cmpid & pu$partition == part & pu$metric == "aurc", ]; if (as.logical(r$noninferior)) "non-inferior" else "not shown non-inferior" }
if (!is.null(MARGIN)) {
  check("Table 3 footnote: the lock's margin gives the non-inferiority line, verdicts from the paired-differences CSV",
        any(grepl(sprintf("Non-inferiority margin %s", fmt_num(MARGIN, 3)), t3obj$footnote, fixed = TRUE)) &&
          any(grepl(sprintf("Optical gate vs pixel entropy: %s in the training center, %s in the external center", verdict_txt("gate_vs_entropy", "internal"), verdict_txt("gate_vs_entropy", "external")), t3obj$footnote, fixed = TRUE)))
} else {
  check("Table 3 footnote: the lock has no margin, so the footnote says the comparisons are descriptive and names pixel entropy",
        any(grepl("No non-inferiority margin was retained in the analysis lock, so the comparisons with pixel entropy and ensemble disagreement are descriptive", t3obj$footnote, fixed = TRUE)) &&
          !any(grepl("Non-inferiority margin [0-9]", t3obj$footnote)))
}
all_text <- c(unlist(t1a), names(t1a), unlist(t1b), names(t1b), unlist(t2), names(t2), unlist(t3), names(t3), tb$footnote, t2obj$footnote, t3obj$footnote,
              quiet(table1_panel_a(coh))$footnote)
check("no exhibit says 'softmax' (the network has one logit: pixel entropy): Table 1A, 1B, 2, 3 cells, headers and footnotes",
      !any(grepl("softmax", all_text, ignore.case = TRUE)) && "Pixel entropy" %in% t3$Method)
check("terminology and spelling in every table string: no 'Center 1' / 'Center 2' / 'Unlabeled' / 'All labeled' / 'IQR' / 'FOV diameter', no British spellings (analysed, colour, centre, labelled, favour, normalis-, penalis-, licence)",
      !any(grepl("Center [12]|nlabeled|All labeled|IQR|FOV diameter|\\banalys(e|ed|ing)\\b|colour|centre|labelled|favour|normalis|penalis|licence", all_text)))
check("Table 3 has no NA, NaN or Inf in cells or footnote", no_na(t3) && no_na(as.list(t3obj$footnote)))
d3 <- docx_rows(file.path(out_tables, "Table3.docx"))
d3_head <- identical(d3[[1]][2:3], c("Training center, cross-fitted", "External center, locked gate"))
at3 <- which(vapply(d3, function(r) identical(r[1], "U-Net outcomes (primary)"), NA))[1]
check("Table3.docx: spanner header row (training center, external center) then body rows (after the two header rows) equal Table3.csv, merged section rows have one cell",
      d3_head && !is.na(at3) && at3 == 3L && length(d3[[at3]]) == 1L && length(d3[[at3 + 1L]]) == 11L &&
        all(vapply(2:nrow(t3), function(i) { got <- d3[[at3 + i - 1L]]; exp <- unname(unlist(t3[i, ]))
          if (length(got) == 1L) all(exp[-1] == "") && norm(got) == norm(exp[1]) else identical(norm(got), norm(exp)) }, NA)))

# stress copy: descriptive-only external, no margin, no entropy choice in the lock
sd3 <- copy_stats("t3")
ge2 <- utils::read.csv(file.path(sd3, "gate_external.csv"), check.names = FALSE); ge2$binary_descriptive <- TRUE
utils::write.csv(ge2, file.path(sd3, "gate_external.csv"), row.names = FALSE)
for (f in c("gate_paired_diffs.csv", "gate_transfer_paired_diffs.csv")) {
  z <- utils::read.csv(file.path(sd3, f), check.names = FALSE); z$margin <- NA_real_; z$noninferior <- NA
  utils::write.csv(z, file.path(sd3, f), row.names = FALSE)
}
t3s <- quiet(table3(lock = list(), stats_dir = sd3))
check("Table 3 stress: descriptive-only external rows carry the dagger on AUROC and slope (U-Net external only), internal rows and SegFormer do not",
      all(grepl("†$", t3s$df[c(3, 6, 7), 8])) && all(grepl("†$", t3s$df[c(3, 6, 7), 11])) && !any(grepl("†", unlist(t3s$df[, 2:6]))) &&
        !any(grepl("†", unlist(t3s$df[15:27, ]))) && grepl("†$", t3s$df[13, 8]) && grepl("†$", t3s$df[9, 8]) &&
        any(startsWith(t3s$footnote, "†")))
sd3b <- copy_stats("t3b"); unlink(file.path(sd3b, "sensitivity.csv"))
t3n <- quiet(table3(stats_dir = sd3b))
check("Table 3: without sensitivity.csv the pixel entropy vs ensemble row is left out (26 rows) and the footnote does not mention it",
      nrow(t3n$df) == 26L && !any(grepl("^Pixel entropy vs", t3n$df$Method)) && !any(grepl("entropy vs ensemble disagreement row", t3n$footnote)))
check("Table 3 stress: no margin in the lock or CSVs gives the descriptive-comparison line, an empty lock omits the entropy variant",
      any(grepl("No non-inferiority margin", t3s$footnote)) && !any(grepl("Non-inferiority margin [0-9]", t3s$footnote)) && !any(grepl("is averaged over", t3s$footnote)))
check("Table 3 refuses a missing input with a message that names the file", errs(quiet(table3(stats_dir = file.path(scratch, "nowhere"))), "missing input .*gate_internal.csv"))

# ---- build skips missing inputs; nothing real was written -------------------------------------------------------------
check("tbl_inputs names the files each table needs; with them present nothing is skipped", all(file.exists(unlist(tbl_inputs()))))
check("no real Table1* to Table3* file under v1/tables and no results/parts/20_tables.json was created or modified", identical(real_before, real_owned()))
unlink(scratch, recursive = TRUE)

cat(sprintf("\n%s\n", if (length(failures)) paste("FAILED:", paste(failures, collapse = "; ")) else "all checks passed"))
if (length(failures)) quit(status = 1)
