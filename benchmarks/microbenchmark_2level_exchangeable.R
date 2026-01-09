#!/usr/bin/env Rscript

SCRIPT_DIR <- normalizePath(dirname(sys.frame(1)$ofile), winslash = "/")
RESULTS_DIR <- file.path(SCRIPT_DIR, "results")
FIGURES_DIR <- file.path(SCRIPT_DIR, "figures")

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# microbenchmark_2level_exchangeable.R
#
# Benchmarks (median runtime) for 2-level exchangeable GEE:
#   (1) networkGEE::ngee      (corstr = "nested-exchangeable")
#   (2) geepack::geese        (corstr = "exchangeable")
#   (3) gee::gee              (corstr = "exchangeable")
#   (4) geeM::geem            (corstr = "exchangeable")
#
# Grid: I in {10,30,100} clusters and J in {10,30,100} cluster sizes.
# Output: a gt() table saved to benchmarks_gt.html (and printed).
# ============================================================

suppressPackageStartupMessages({
  library(microbenchmark)
  library(dplyr)
  library(tidyr)
  library(gt)
})

# ---- Dependency checks (fail fast with helpful messages) ----
need_pkgs <- c("networkGEE", "geepack", "gee", "geeM", "microbenchmark", "dplyr", "tidyr", "purrr", "gt")
missing <- need_pkgs[!vapply(need_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(
    "Missing packages: ", paste(missing, collapse = ", "), "\n",
    "Install them, then rerun.\n",
    "For networkGEE (if needed): devtools::install_github('tomchen00/networkGEE')"
  )
}

# ---- Simulation (2-level clustered binary outcome) ----
simulate_2level_bin <- function(I, J, seed = 1729) {
  set.seed(seed)

  N <- I * J
  dat <- data.frame(
    cluster = rep(seq_len(I), each = J),
    x1      = rnorm(N),
    x2      = rbinom(N, 1, 0.5)
  )

  # cluster random intercept to induce within-cluster dependence
  b_i <- rnorm(I, sd = 0.7)
  eta <- -0.5 + 0.8 * dat$x1 - 0.4 * dat$x2 + b_i[dat$cluster]
  p   <- plogis(eta)
  dat$Y <- rbinom(N, size = 1, prob = p)

  dat
}

# ---- Fit wrappers (each returns a fitted object) ----
fit_ngee <- function(dat) {
  networkGEE::ngee(
    formula    = Y ~ x1 + x2,
    id         = "cluster",
    dat        = dat,
    family     = "binomial",
    corstr     = "nested-exchangeable",
    se_adjust  = "unadjusted",
    optim      = "deterministic"
  )
}

fit_geese <- function(dat) {
  geepack::geese(
    formula   = Y ~ x1 + x2,
    id        = cluster,
    data      = dat,
    family    = binomial,
    corstr    = "exchangeable",
    scale.fix = TRUE
  )
}

fit_gee <- function(dat) {
  gee::gee(
    formula = Y ~ x1 + x2,
    id      = cluster,
    data    = dat,
    family  = binomial(link = "logit"),
    corstr  = "exchangeable",
    scale.fix = TRUE
  )
}

fit_geem <- function(dat) {
  geeM::geem(
    formula = Y ~ x1 + x2,
    id      = cluster,
    data    = dat,
    family  = binomial(link = "logit"),
    corstr  = "exchangeable",
    scale.fix = TRUE
  )
}

# ---- Benchmark one (I,J) cell ----
bench_one <- function(I, J, times = 5, seed = 1729) {
  dat <- simulate_2level_bin(I, J, seed = seed)

  # microbenchmark will abort if any expression errors.
  # So we wrap each fit in tryCatch() and return NULL on failure.
  mb <- microbenchmark::microbenchmark(
    ngee  = tryCatch(fit_ngee(dat),  error = function(e) NULL),
    geese = tryCatch(fit_geese(dat), error = function(e) NULL),
    gee   = tryCatch(fit_gee(dat),   error = function(e) NULL),
    geem  = tryCatch(fit_geem(dat),  error = function(e) NULL),
    times = times
  )

  # Summarize median time (ms) per method
  out <- as.data.frame(mb) %>%
    mutate(
      I = I,
      J = J,
      method = as.character(expr),
      time_ms = as.numeric(time) / 1e6
    ) %>%
    group_by(I, J, method) %>%
    summarise(
      median_ms = median(time_ms, na.rm = TRUE),
      .groups = "drop"
    )

  # Mark failures: if median is NA (all NULL due to errors)
  out %>%
    mutate(status = ifelse(is.finite(median_ms), "ok", "failed"))
}

# ---- Run full grid ----
grid_I <- c(10, 30, 100)
grid_J <- c(10, 30, 100)

# Adjust times upward if you want more stable estimates (at cost of runtime)
MB_TIMES <- 5
grid <- expand.grid(I = grid_I, J = grid_J)

results <- purrr::pmap_dfr(
  grid,
  function(I, J) bench_one(I = I, J = J, times = MB_TIMES, seed = 1729)
)

# ---- Make a gt table (rows=method; columns=I×J) ----
tab <- results %>%
  mutate(
    cell = paste0("I=", I, ",\n J=", J),
    value = ifelse(status == "ok", median_ms, NA_real_)
  ) %>%
  select(method, cell, value) %>%
  tidyr::pivot_wider(names_from = cell, values_from = value) %>%
  arrange(factor(method, levels = c("ngee", "geese", "gee", "geem")))

gt_tbl <- tab %>%
  gt(rowname_col = "method") %>%
  tab_header(
    title = "2-level Exchangeable GEE Runtime (median milliseconds)",
    subtitle = paste0("microbenchmark times = ", MB_TIMES, " per method per (I,J)")
  ) %>%
  fmt_number(columns = where(is.numeric), decimals = 2) %>%
  cols_label(.list = setNames(names(tab)[-1], names(tab)[-1])) %>%
  tab_source_note(source_note = "Cells are median wall-clock times (ms).")

print(gt_tbl)

# ---- Save outputs ----
gt::gtsave(gt_tbl, "benchmarks_gt.html")
png_file <- file.path(FIGURES_DIR, "timing_2level_exchangeable.png")
gt_grob <- gt::as_gtable(gt_tbl)
png(
  filename = png_file,
  width  = 1600,
  height = 900,
  res    = 150
)
grid::grid.newpage()
grid::grid.draw(gt_grob)
dev.off()

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

rds_file <- file.path(RESULTS_DIR, "timing_2level_exchangeable.rds")
csv_file <- file.path(RESULTS_DIR, "timing_2level_exchangeable.csv")
png_file <- file.path(FIGURES_DIR, "timing_2level_exchangeable.png")

saveRDS(results, rds_file)
write.csv(results, csv_file, row.names = FALSE)

# Optional: save pretty table as image for README embedding
# Requires: install.packages("webshot2")
message("Saved:")
message("  ", rds_file)
message("  ", csv_file)
message("  ", png_file)
