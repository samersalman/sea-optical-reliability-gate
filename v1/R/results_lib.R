# Results registry: every number that reaches the manuscript passes through reg().
#
# Convention for every analysis script (T13, T14, T16-T19):
#
#   source(file.path(V1, "R", "results_lib.R"))  # after 00_setup.R, in either order; V1 is the v1 dir
#   reg("n_patients_c1", 61L)                    # integer-valued -> "61"; 10662 -> "10,662"
#   reg("ext_aurc_gate", 0.1234, digits = 3)     # "0.123"; always pass digits for non-integers
#   reg("ext_aurc_diff_null", 0.0100, digits = 3)  # nulls print at the same precision as effects
#   reg("ext_gate_p", 0.0004, fmt = "p")         # "<0.001"
#   reg("ext_auroc_ci", c(0.71, 0.64, 0.78), digits = 2, fmt = "ci")  # "0.71 (95% CI 0.64 to 0.78)"
#   reg("share_difficult", 0.752, fmt = "percent")  # proportion in, "75.2%" out
#   reg("software_r", "4.6.0")                   # strings are kept as given (versions, licence names)
#   write_results_part("04b_external_eval")      # LAST line of the script
#
# Rules
# - Keys are lowercase snake_case: ^[a-z][a-z0-9_]*$. Duplicate keys (same script or another
#   part) are an error. Keys are unique across the whole project.
# - reg() stores the display string (what {{key}} prints) and the raw value. Display is fixed
#   precision (trailing zeros kept), comma thousands separators, ASCII minus, no dashes.
# - write_results_part() writes results/parts/<script>.json (sorted keys, no timestamps) and
#   then clears the registry. R/30_bundle.R merges the parts into results/results.json.
# - Downstream scripts read the bundle with results_get(key) / results_get(key, "raw").
# - With OPTICS_DATA=synthetic (see 00_setup.R) the default root is work/synthetic/out, so
#   development runs never write into the real results/.
# - Numbers that are not results (software versions, licence names) still go through reg()
#   as strings, because prose templates may not contain digits (see manuscript/digit_allowlist.txt).
#
# Formatting is R round() then fixed-decimal formatC(), the same as fmt_num() in 00_setup.R.
# Years and identifiers must be registered as strings: a numeric 2025 would print as "2,025".
# fmt_num/fmt_ci/fmt_p below are defined only if 00_setup.R has not already defined them (its
# versions are call-compatible); reg() itself never depends on them.

if (!exists(".reg_state", inherits = FALSE)) {
  .reg_state <- new.env(parent = emptyenv())
  .reg_state$entries <- list()
}

KEY_RE <- "^[a-z][a-z0-9_]*$"
EM_DASH <- "\u2014"

synthetic_mode <- function() identical(Sys.getenv("OPTICS_DATA", ""), "synthetic")

# The real v1 directory: OPTICS_V1, else the nearest ancestor of this script or the working
# directory that holds config/analysis_plan.yaml (directly or in a v1/ child).
v1_dir <- function() {
  env <- Sys.getenv("OPTICS_V1", "")
  if (nzchar(env)) return(normalizePath(env, mustWork = TRUE))
  file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  for (d in c(if (length(file_arg)) dirname(normalizePath(file_arg[1], mustWork = FALSE)), getwd())) {
    repeat {
      for (cand in c(d, file.path(d, "v1"))) {
        if (file.exists(file.path(cand, "config", "analysis_plan.yaml"))) return(normalizePath(cand))
      }
      parent <- dirname(d)
      if (parent == d) break
      d <- parent
    }
  }
  stop("cannot locate v1/config/analysis_plan.yaml from ", getwd(), "; set OPTICS_V1")
}

# Default root for results/ and manuscript/: v1, or work/synthetic/out in synthetic mode.
results_root <- function() {
  if (synthetic_mode()) file.path(v1_dir(), "work", "synthetic", "out") else v1_dir()
}

# ---- formatters ---------------------------------------------------------------------------

.rl_num <- function(x, digits = 2, big.mark = ",", na = "NA") {
  x <- round(x, digits) + 0   # + 0 turns -0 into 0
  out <- formatC(x, format = "f", digits = digits, big.mark = big.mark)
  out[is.na(x)] <- na
  unname(out)
}

.rl_ci <- function(est, lo, hi, digits = 2, sep = " to ", level = NULL) {
  lab <- if (is.null(level)) "" else paste0(level, "% CI ")
  paste0(.rl_num(est, digits), " (", lab, .rl_num(lo, digits), sep, .rl_num(hi, digits), ")")
}

# "<0.001" below the display floor; ">0.999" when rounding would print 1; prefix = TRUE gives "P < 0.001"
.rl_p <- function(p, digits = 3, prefix = FALSE) {
  floor_p <- 10^-digits
  op <- ifelse(p < floor_p, "<", ifelse(p > 1 - floor_p, ">", "="))
  num <- .rl_num(ifelse(p < floor_p, floor_p, ifelse(p > 1 - floor_p, 1 - floor_p, p)), digits)
  out <- if (prefix) paste("P", op, num) else ifelse(op == "=", num, paste0(op, num))
  out[is.na(p)] <- "NA"
  unname(out)
}

if (!exists("fmt_num", mode = "function")) fmt_num <- .rl_num
if (!exists("fmt_ci", mode = "function")) fmt_ci <- .rl_ci
if (!exists("fmt_p", mode = "function")) fmt_p <- .rl_p

# x is a proportion in [0, 1]; use reg(..., unit = "%") for a value already on the percent scale
fmt_pct <- function(x, digits = 1) paste0(.rl_num(100 * x, digits), "%")

fmt_ratio <- function(x, digits = 2) .rl_num(x, digits)

.formatters <- list(
  num = function(x, digits) .rl_num(x, digits),
  percent = function(x, digits) fmt_pct(x, digits),
  ratio = function(x, digits) fmt_ratio(x, digits),
  ci = function(x, digits) .rl_ci(x[1], x[2], x[3], digits, level = 95),
  p = function(x, digits) .rl_p(x, digits),
  p_prefixed = function(x, digits) .rl_p(x, digits, prefix = TRUE)
)
.default_digits <- c(percent = 1, ratio = 2, ci = 2, p = 3, p_prefixed = 3)

# ---- registry -----------------------------------------------------------------------------

reg_reset <- function() {
  .reg_state$entries <- list()
  invisible(NULL)
}

reg <- function(key, value, digits = NULL, fmt = NULL, unit = NULL, note = NULL) {
  if (!is.character(key) || length(key) != 1L || is.na(key) || !grepl(KEY_RE, key)) {
    stop("reg: key must be lowercase snake_case, got ", deparse(key))
  }
  if (!is.null(.reg_state$entries[[key]])) stop("reg: duplicate key '", key, "'")
  if (!is.null(digits) && (length(digits) != 1L || !is.numeric(digits) || digits < 0 || digits != round(digits))) {
    stop("reg: digits must be a single non-negative integer (key '", key, "')")
  }

  if (is.character(value)) {
    if (length(value) != 1L || is.na(value) || !nzchar(value)) stop("reg: string value must be one non-empty string (key '", key, "')")
    if (!is.null(digits) || !is.null(fmt)) stop("reg: digits/fmt do not apply to a string value (key '", key, "')")
    display <- value
    raw <- value
  } else if (is.numeric(value)) {
    if (!all(is.finite(value))) stop("reg: value must be finite (key '", key, "')")
    fmt_name <- if (is.null(fmt)) "num" else fmt
    if (is.function(fmt)) {
      if (is.null(digits)) stop("reg: pass digits with a custom fmt function (key '", key, "')")
      display <- fmt(value, digits)
    } else {
      if (!is.character(fmt_name) || length(fmt_name) != 1L || is.null(.formatters[[fmt_name]])) {
        stop("reg: unknown fmt '", paste(fmt_name, collapse = ","), "' (key '", key, "'); use ",
             paste(names(.formatters), collapse = ", "), " or a function(value, digits)")
      }
      need <- if (fmt_name == "ci") 3L else 1L
      if (length(value) != need) stop("reg: fmt '", fmt_name, "' needs ", need, " value(s) (key '", key, "')")
      if (is.null(digits)) {
        digits <- if (fmt_name == "num") (if (all(value == round(value))) 0 else 2) else .default_digits[[fmt_name]]
      }
      display <- .formatters[[fmt_name]](value, digits)
    }
    raw <- as.numeric(value)
  } else {
    stop("reg: value must be numeric or a string (key '", key, "')")
  }

  if (!is.null(unit)) display <- paste0(display, if (startsWith(unit, "%")) "" else " ", unit)
  note <- if (is.null(note)) "" else as.character(note)
  for (s in c(display, note)) {
    if (grepl(EM_DASH, s, fixed = TRUE)) stop("reg: em-dash in value or note (key '", key, "')")
    if (grepl("[\r\n]", s)) stop("reg: line break in value or note (key '", key, "')")
  }
  .reg_state$entries[[key]] <- list(display = display, raw = raw, note = note)
  invisible(display)
}

# ---- JSON and part files ------------------------------------------------------------------

write_json_file <- function(obj, path) {
  txt <- jsonlite::toJSON(obj, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  writeLines(enc2utf8(as.character(txt)), path, useBytes = TRUE)
}

read_part <- function(path) {
  p <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!is.list(p) || is.null(p$script) || !is.list(p$entries)) stop("malformed results part: ", path)
  for (k in names(p$entries)) {
    e <- p$entries[[k]]
    if (!is.character(e$display) || is.null(e$raw)) stop("malformed entry '", k, "' in ", path)
    p$entries[[k]]$raw <- unlist(e$raw)
  }
  p
}

write_results_part <- function(script_name, root = results_root()) {
  name <- sub("\\.[Rr]$", "", basename(script_name))
  entries <- .reg_state$entries
  if (!length(entries)) stop("write_results_part: nothing registered for '", name, "'")

  parts_dir <- file.path(root, "results", "parts")
  dir.create(parts_dir, recursive = TRUE, showWarnings = FALSE)
  own <- file.path(parts_dir, paste0(name, ".json"))
  for (f in setdiff(list.files(parts_dir, pattern = "\\.json$", full.names = TRUE), own)) {
    clash <- intersect(names(entries), names(read_part(f)$entries))
    if (length(clash)) {
      stop("write_results_part: key(s) already registered in ", basename(f), ": ", paste(clash, collapse = ", "))
    }
  }

  entries <- entries[order(names(entries), method = "radix")]
  write_json_file(list(script = paste0(name, ".R"), entries = entries), own)
  reg_reset()
  invisible(own)
}

# ---- reading the bundle -------------------------------------------------------------------

read_results <- function(root = results_root()) {
  path <- file.path(root, "results", "results.json")
  if (!file.exists(path)) stop("results.json not found; run R/30_bundle.R first: ", path)
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

# what = "display" returns the publication string, "raw" the numeric (or string) value
results_get <- function(key, what = c("display", "raw"), root = results_root()) {
  what <- match.arg(what)
  res <- read_results(root)
  k <- if (what == "raw") paste0("raw.", key) else key
  if (is.null(res[[k]])) stop("results_get: unknown key '", key, "'")
  if (what == "raw") unlist(res[[k]]) else res[[k]]
}
