# Merge results/parts/*.json into results/results.json and results/numbers-ledger.md.
# Output is deterministic (sorted keys, no timestamps), so a re-run on the same parts is
# byte-identical. Usage: Rscript R/30_bundle.R [root]

if (!exists("write_results_part", mode = "function")) {
  local({
    args <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    cand <- c(if (length(args)) dirname(sub("^--file=", "", args[1])),
              tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL), "R", "v1/R", ".")
    hit <- cand[file.exists(file.path(cand, "results_lib.R"))]
    if (!length(hit)) stop("results_lib.R not found next to 30_bundle.R")
    source(file.path(hit[1], "results_lib.R"), chdir = FALSE)
  })
}

sha256_file <- function(path) unname(tools::sha256sum(path))

ledger_cell <- function(x) gsub("|", "\\|", paste(x, collapse = ", "), fixed = TRUE)

raw_text <- function(raw) {
  if (is.numeric(raw)) paste(trimws(formatC(raw, digits = 15, format = "g")), collapse = ", ") else as.character(raw)
}

bundle_results <- function(root = results_root()) {
  parts_dir <- file.path(root, "results", "parts")
  files <- sort(list.files(parts_dir, pattern = "\\.json$", full.names = TRUE), method = "radix")
  if (!length(files)) stop("bundle: no result parts in ", parts_dir)

  entries <- list()
  for (f in files) {
    part <- read_part(f)
    for (k in names(part$entries)) {
      if (!is.null(entries[[k]])) {
        stop("bundle: duplicate key '", k, "' in ", part$script, " and ", entries[[k]]$script)
      }
      entries[[k]] <- c(part$entries[[k]], list(script = part$script))
    }
  }
  keys <- sort(names(entries), method = "radix")

  display <- lapply(keys, function(k) entries[[k]]$display)
  raw <- lapply(keys, function(k) entries[[k]]$raw)
  names(display) <- keys
  names(raw) <- paste0("raw.", keys)
  json_path <- file.path(root, "results", "results.json")
  write_json_file(c(display, raw), json_path)

  rows <- vapply(keys, function(k) {
    e <- entries[[k]]
    paste0("| ", paste(ledger_cell(k), ledger_cell(e$display), ledger_cell(raw_text(e$raw)),
                       ledger_cell(e$script), ledger_cell(e$note), sep = " | "), " |")
  }, character(1))
  ledger <- c(
    "# Numbers ledger", "",
    "Built by R/30_bundle.R from results/parts/*.json. Do not edit by hand.", "",
    "| key | display | raw | source script | note |",
    "|---|---|---|---|---|",
    unname(rows)
  )
  ledger_path <- file.path(root, "results", "numbers-ledger.md")
  writeLines(enc2utf8(ledger), ledger_path, useBytes = TRUE)

  invisible(list(json = json_path, ledger = ledger_path, n_keys = length(keys), n_parts = length(files)))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(TRUE)
  out <- bundle_results(if (length(args)) args[1] else results_root())
  cat(sprintf("bundled %d keys from %d parts\n%s  %s\n%s  %s\n",
              out$n_keys, out$n_parts, sha256_file(out$json), out$json,
              sha256_file(out$ledger), out$ledger))
}
