library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- TRUE
max_epoch <- 300
n_runs <- 50
time_interval <- 1

output_file <- "hanoi.pdf"

# These match the actual file names exactly
resolutions <- list("0.0001", "0.001", "0.01", "0.1", "1.0", "10.0")

if (dosave) pdf(output_file, width = 5 * length(resolutions), height = 4.25)

par(mfrow = c(1, length(resolutions)), family = "serif", mar = c(3.5, 3.5, 1.5, 0.5) + 0.5, mgp = c(2.5, 1, 0))

for (res_str in resolutions) {
  input_file <- sprintf("hanoi_%s.json", res_str)

  dataj <- fromJSON(file = input_file)

  means_modularity <- dataj$modularity$mean[1:(max_epoch / time_interval)]
  sd_modularity <- dataj$modularity$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

  means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
  sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

  means_random <- dataj$random$mean[1:(max_epoch / time_interval)]
  sd_random <- dataj$random$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

  x <- seq(1, max_epoch, by = time_interval)
  cip_x <- c(x, rev(x))
  cip_y_modularity <- c(means_modularity - sd_modularity, rev(means_modularity + sd_modularity))
  cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
  cip_y_random <- c(means_random - sd_random, rev(means_random + sd_random))

  plot.new()
  plot.window(ylim = c(-0.05, 1.0), xlim = c(0, max_epoch))
  axis(1, at = c(0, 100, 200, 300), labels = TRUE, cex.axis = 1.7)
  axis(2, at = c(0.0, 0.25, 0.5, 0.75, 1.0), labels = TRUE, cex.axis = 1.7)
  title(main = paste("Resolution =", res_str), cex.main = 2.0, font.main = 1)
  title(xlab = "Epoch", cex.lab = 2.0)
  title(ylab = "Return", cex.lab = 2.0)
  box()

  polygon(cip_x, cip_y_modularity, col = adjustcolor("red", alpha.f = 0.3), border = FALSE)
  polygon(cip_x, cip_y_prim, col = adjustcolor("grey30", alpha.f = 0.3), border = FALSE)
  polygon(cip_x, cip_y_random, col = adjustcolor("blue", alpha.f = 0.3), border = FALSE)

  lines(x, means_modularity, col = "red")
  points(x, means_modularity, col = "red", pch = 16)

  lines(x, means_prim, col = "grey30")
  points(x, means_prim, col = "grey30", pch = 16)

  lines(x, means_random, col = "blue")
  points(x, means_random, col = "blue", pch = 16)

  # If it's the final plot, add the legend
  if (res_str == tail(resolutions, n=1)) {
    legend(x = 226, y = 0.18, legend = c("Modularity", "Random", "Primitive"),
           col = c("red", "blue", "grey30"), pch = 16,
           bg = adjustcolor("white", alpha.f = 0.7), cex = 1.5)
  }

}

if (dosave) dev.off()
