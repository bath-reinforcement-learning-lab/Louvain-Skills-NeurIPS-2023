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

# Helper
get_xy <- function(method_name, thr) {
  raw <- dataj[[method_name]][[thr]]
  if (is.null(raw) || length(raw) == 0) return(NULL)
  m <- do.call(rbind, raw)
  colnames(m) <- c("states", "steps")
  m[order(m[, "states"]), , drop = FALSE]
}

# Methods
methods <- list(
  list(key = "louvain",          col = col_louvain,   lab = "Louvain"),
  list(key = "node_betweenness", col = col_nodebet,   lab = "Node Betweenness"),
  list(key = "label_prop",       col = col_labelprop, lab = "Label Propagation"),
  list(key = "eigenoptions",     col = col_eig,       lab = "Eigenoptions"),
  list(key = "primitive",        col = col_primitive, lab = "Primitive")
)

# ---- Global axis limits ----
all_states <- c()
all_steps  <- c()
for (thr in thresholds) {
  for (m in methods) {
    xy <- get_xy(m$key, thr)
    if (!is.null(xy)) {
      all_states <- c(all_states, xy[, "states"])
      all_steps  <- c(all_steps,  xy[, "steps"])
    }
  }
}
if (length(all_states) == 0) stop("No data found in JSON.")

xlim <- range(all_states)                 # linear x
ymin <- min(all_steps[all_steps > 0])     # log y
ymax <- max(all_steps)
ylim <- c(ymin, ymax)

# Y ticks as powers of 10
mk_pow10_ticks <- function(rng) {
  pmin <- floor(log10(rng[1])); pmax <- ceiling(log10(rng[2]))
  at   <- 10^seq(pmin, pmax)
  labs <- parse(text = paste0("10^", seq(pmin, pmax)))
  list(at = at, labs = labs)
}
yt <- mk_pow10_ticks(ylim)

# Common X ticks (pretty) with explicit labels (Option A)
xat_common  <- pretty(xlim)
xlab_common <- ifelse(xat_common == 0, "0", format(xat_common, big.mark = ","))

# ---- Open device & layout ----
if (dosave) pdf(output_file, width = 12, height = 3.6)

# Equal plot areas; small outer right margin for breathing room
par(
  mfrow = c(1, 4),
  family = par_family,
  mar = c(3.5, 3.2, 1.5, 0.6),  # bottom, left, top, right
  oma = c(0, 0, 0, 1.4),
  mgp = c(2.2, 0.7, 0),
  xaxs = "r",
  yaxs = "r",
  cex.lab = 1.3
)

# ---- Plot each threshold ----
for (i in seq_along(thresholds)) {
  thr <- thresholds[i]
  
  plot.new()
  plot.window(xlim = xlim, ylim = ylim, log = "y")
  
  # X-axis: pretty ticks with explicit labels (commas; aligned "0")
  axis(1, at = xat_common, labels = xlab_common, cex.axis = 1.0)
  
  # Y-axis: ticks and labels on ALL panels
  axis(2, at = yt$at, labels = yt$labs, las = 1, cex.axis = 1.0)
  
  box()
  
  # Titles
  title(main = titles[i], font.main = 1)
  title(xlab = "Number of States")
  if (i == 1) title(ylab = "Decision Stages to Reach Threshold")
  
  # Series
  for (m in methods) {
    xy <- get_xy(m$key, thr)
    if (is.null(xy)) next
    lines(xy[, "states"], xy[, "steps"], col = m$col)
    points(xy[, "states"], xy[, "steps"], col = m$col, pch = 16)
  }
  
  # Legend only in the final panel
  if (i == length(thresholds)) {
    present <- vapply(methods, function(m) {
      any(vapply(thresholds, function(t2) !is.null(get_xy(m$key, t2)), logical(1)))
    }, logical(1))
    leg_labs <- vapply(methods[present], `[[`, character(1), "lab")
    leg_cols <- vapply(methods[present], `[[`, character(1), "col")
    
    legend(
      "bottomright",
      legend = leg_labs,
      col = leg_cols,
      pch = 16,
      bg = adjustcolor("white", alpha.f = 0.8),
      cex = 1.0,
      inset = 0.02
    )
  }
}

if (dosave) dev.off()
