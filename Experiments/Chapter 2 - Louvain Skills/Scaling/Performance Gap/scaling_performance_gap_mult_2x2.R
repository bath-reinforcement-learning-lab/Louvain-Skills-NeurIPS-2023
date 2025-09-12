# ---- Packages & setup ----
library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

# Input/output
input_file  <- "scaling_results_mult.json"
output_file <- "scaling_performance_gap_mult.pdf"

# Plot options
dosave <- TRUE
par_family <- "serif"

# Colors
col_primitive <- "grey30"
col_louvain   <- "red"
col_nodebet   <- "navy"
col_labelprop <- "forestgreen"
col_eig       <- "orange"

# Thresholds + titles
thresholds <- c("0.2", "0.4", "0.6", "0.8")
titles     <- c("20%", "40%", "60%", "80%")

# ---- Load JSON ----
dataj <- fromJSON(file = input_file)

# Helper to extract and sort (x=states, y=steps)
get_xy <- function(method_name, thr) {
  raw <- dataj[[method_name]][[thr]]
  if (is.null(raw) || length(raw) == 0) return(NULL)
  m <- do.call(rbind, raw)
  colnames(m) <- c("states", "steps")
  m[order(m[, "states"]), , drop = FALSE]
}

# Methods
methods <- list(
  list(key = "primitive",        col = col_primitive, lab = "Primitive"),
  list(key = "louvain",          col = col_louvain,   lab = "Louvain"),
  list(key = "node_betweenness", col = col_nodebet,   lab = "Node Betweeness"),
  list(key = "label_prop",       col = col_labelprop, lab = "Label Propagation"),
  list(key = "eigenoptions",     col = col_eig,       lab = "Eigenoptions")
)

# ---- Global axis limits (shared across all panels) ----
all_states <- c(); all_steps <- c()
for (thr in thresholds) for (m in methods) {
  xy <- get_xy(m$key, thr)
  if (!is.null(xy)) { all_states <- c(all_states, xy[, "states"]); all_steps <- c(all_steps, xy[, "steps"]) }
}
stopifnot(length(all_states) > 0)

# X: start at 0, gentle right padding
xmax <- max(all_states)
xlim <- c(0, xmax * 1.05)

# Y: pad to previous & next full decades
ymin_raw <- min(all_steps[all_steps > 0]); ymax_raw <- max(all_steps)
y_lower_decade <- 10^floor(log10(ymin_raw))
y_upper_decade <- 10^ceiling(log10(ymax_raw))
ylim <- c(y_lower_decade, y_upper_decade)

# ---- Tick helpers ----
mk_pow10_ticks <- function(rng) {
  pmin <- floor(log10(rng[1])); pmax <- ceiling(log10(rng[2]))
  at   <- 10^seq(pmin, pmax)
  labs <- parse(text = paste0("10^", seq(pmin, pmax)))
  list(at = at, labs = labs)
}
yt <- mk_pow10_ticks(ylim)

xat_common  <- pretty(xlim, n = 4)
xlab_common <- ifelse(xat_common == 0, "0", format(xat_common, big.mark = ","))

# ---- Device: 7 × 7 inches ----
if (dosave) pdf(output_file, width = 7.0, height = 7.0)

# Layout with a thin blank spacer row between plot rows
lay <- matrix(c(1,2,
                0,0,
                3,4), nrow = 3, byrow = TRUE)
layout(lay, heights = c(1, 0.03, 1))  # inter-row spacing toned down to 0.03

# Global par
par(
  family = par_family,
  mar = c(3.6, 3.5, 1.4, 0.6),  # ↑ bottom margin to prevent xlab clipping
  oma = c(0.4, 0.6, 0.4, 0.4),
  mgp = c(2.0, 0.6, 0),         # bring axis titles a touch closer
  xaxs = "i", yaxs = "i",
  cex.lab = 1.6,                # axis label size (as you liked)
  cex.axis = 1.2,               # tick label size
  cex.main = 1.8                # panel title size
)

is_left_col  <- function(i) i %in% c(1, 3)

# ---- Draw panels ----
for (i in seq_along(thresholds)) {
  thr <- thresholds[i]
  
  plot.new()
  plot.window(xlim = xlim, ylim = ylim, log = "y")
  
  # Slightly darker horizontal guide lines at 10^k
  abline(h = yt$at, col = adjustcolor("grey60", alpha.f = 1.0), lty = "dotted")
  
  # Axes & labels
  axis(1, at = xat_common, labels = xlab_common); title(xlab = "Number of States")
  axis(2, at = yt$at, labels = yt$labs, las = 1)
  if (is_left_col(i)) title(ylab = "Decision Stages to Reach Threshold")
  
  box()
  title(main = titles[i], font.main = 1)
  
  # Series
  for (m in methods) {
    xy <- get_xy(m$key, thr); if (is.null(xy)) next
    lines(xy[, "states"], xy[, "steps"], col = m$col)
    points(xy[, "states"], xy[, "steps"], col = m$col, pch = 16)
  }
  
  # Legend in bottom-right panel
  if (i == 4) {
    present <- vapply(methods, function(m) any(vapply(thresholds, function(t2) !is.null(get_xy(m$key, t2)), logical(1))), logical(1))
    leg_labs <- vapply(methods[present], `[[`, character(1), "lab")
    leg_cols <- vapply(methods[present], `[[`, character(1), "col")
    legend("bottomright", legend = leg_labs, col = leg_cols, pch = 16,
           bg = adjustcolor("white", 0.85), cex = 1.2, inset = 0.02)
  }
}

if (dosave) dev.off()
