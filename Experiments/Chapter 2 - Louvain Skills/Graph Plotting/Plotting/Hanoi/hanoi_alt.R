library("rjson")
library("rstudioapi")
library("latex2exp")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- TRUE
max_epoch <- 300
n_runs <- 50
time_interval <- 2

output_file <- "hanoi_alt.pdf"

if (dosave) pdf(output_file, width = 5, height = 4.25)

par(mfrow = c(1, 1), family = "serif", mar = c(3.5, 3.5, 1.5, 0.5) + 0.5, mgp = c(2.5, 1, 0))


input_file <- sprintf("hanoi.json")

dataj <- fromJSON(file = input_file)

means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_louvain <- dataj$"louvain"$mean[1:(max_epoch / time_interval)]
sd_louvain <- dataj$"louvain"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_flat <- dataj$"flat"$mean[1:(max_epoch / time_interval)]
sd_flat <- dataj$"flat"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_lvl1 <- dataj$"level 0"$mean[1:(max_epoch / time_interval)]
sd_lvl1 <- dataj$"level 0"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_lvl2 <- dataj$"level 1"$mean[1:(max_epoch / time_interval)]
sd_lvl2 <- dataj$"level 1"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_lvl3 <- dataj$"level 2"$mean[1:(max_epoch / time_interval)]
sd_lvl3 <- dataj$"level 2"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)


x <- seq(1, max_epoch, by = time_interval)
cip_x <- c(x, rev(x))
cip_y_louvain <- c(means_louvain - sd_louvain, rev(means_louvain + sd_louvain))
cip_y_flat <- c(means_flat - sd_flat, rev(means_flat + sd_flat))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_lvl1 <- c(means_lvl1 - sd_lvl1, rev(means_lvl1 + sd_lvl1))
cip_y_lvl2 <- c(means_lvl2 - sd_lvl2, rev(means_lvl2 + sd_lvl2))
cip_y_lvl3 <- c(means_lvl3 - sd_lvl3, rev(means_lvl3 + sd_lvl3))

plot.new()
plot.window(ylim = c(-0.4, 0.9), xlim = c(0, max_epoch))
axis(1, at = c(0, 100, 200, 300), labels = TRUE, cex.axis = 1.7)
axis(2, at = c(-0.3, 0.0, 0.3, 0.6, 0.9), labels = TRUE, cex.axis = 1.7)
title(main = "Hanoi", cex.main = 2.0, font.main = 1)
title(xlab = "Epoch", cex.lab = 2.0)
title(ylab = "Return", cex.lab = 2.0)
box()

polygon(cip_x, cip_y_prim, col = adjustcolor("grey30", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_lvl1, col = adjustcolor("mediumorchid2", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_lvl2, col = adjustcolor("darkmagenta", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_lvl3, col = adjustcolor("mediumpurple2", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_flat, col = adjustcolor("darkred", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_louvain, col = adjustcolor("red", alpha.f = 0.3), border = FALSE)


lines(x, means_prim, col = "grey30")
points(x, means_prim, col = "grey30", pch = 16)

lines(x, means_lvl1, col = "mediumorchid2")
points(x, means_lvl1, col = "mediumorchid2", pch = 16)

lines(x, means_lvl2, col = "darkmagenta")
points(x, means_lvl2, col = "darkmagenta", pch = 16)

lines(x, means_lvl3, col = "mediumpurple2")
points(x, means_lvl3, col = "mediumpurple2", pch = 16)

lines(x, means_flat, col = "darkred")
points(x, means_flat, col = "darkred", pch = 16)

lines(x, means_louvain, col = "red")
points(x, means_louvain, col = "red", pch = 16)

# If it's the final plot, add the legend
legend(x = 188, y = 0.67, legend = c("Louvain", "Flat", "Level 1", "Level 2", "Level 3", "Level 4", "Primitive"),
       col = c("red", "darkred", "mediumorchid2", "darkmagenta", "mediumpurple2", "mediumpurple4", "grey30"), pch = 16,
       bg = adjustcolor("white", alpha.f = 0.7), cex = 1.5)


if (dosave) dev.off()

