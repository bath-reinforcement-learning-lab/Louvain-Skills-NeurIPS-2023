library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- TRUE
max_epoch <- 100
n_runs <- 50
time_interval <- 1

output_file <- "incremental.pdf"

if (dosave) pdf(output_file, width = 5, height = 4.25)

par(mfrow = c(1, 1), family = "serif", mar = c(3.5, 3.5, 1.5, 0.5) + 0.5, mgp = c(2.5, 1, 0))


input_file <- sprintf("incremental.json")

dataj <- fromJSON(file = input_file)

means_louvain <- dataj$"louvain"$mean[1:(max_epoch / time_interval)]
sd_louvain <- dataj$"louvain"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_replace <- dataj$"replace"$mean[1:(max_epoch / time_interval)]
sd_replace <- dataj$"replace"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_update <- dataj$"update"$mean[1:(max_epoch / time_interval)]
sd_update <- dataj$"update"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_hybrid <- dataj$"hybrid"$mean[1:(max_epoch / time_interval)]
sd_hybrid <- dataj$"hybrid"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

x <- seq(1, max_epoch, by = time_interval)
cip_x <- c(x, rev(x))
cip_y_louvain <- c(means_louvain - sd_louvain, rev(means_louvain + sd_louvain))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_replace <- c(means_replace - sd_replace, rev(means_replace + sd_replace))
cip_y_update <- c(means_update - sd_update, rev(means_update + sd_update))
cip_y_hybrid <- c(means_hybrid - sd_hybrid, rev(means_hybrid + sd_hybrid))

#plot.new()
#plot.window(ylim = c(-0.4, 0.9), xlim = c(0, max_epoch))
#axis(1, at = c(0, 25, 50, 75), labels = TRUE, cex.axis = 1.7)
#axis(2, at = c(-0.4, 0.0, 0.4, 0.8), labels = TRUE, cex.axis = 1.7)
#title(xlab = "Epoch", cex.lab = 2.0)
#title(ylab = "Return", cex.lab = 2.0)
#box()

plot.new()
plot.window(ylim = c(-1.0, 4.5), xlim = c(0, max_epoch))
axis(1, at = c(0, 25, 50, 75, 100), labels = TRUE, cex.axis = 1.7)
axis(2, at = c(-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0), labels = TRUE, cex.axis = 1.7)
title(xlab = "Epoch", cex.lab = 2.0)
title(ylab = "Sum of Rewards in Epoch", cex.lab = 2.0)
box()

# option retraining intervals
abline(v = 100 / 100,  col = "#cccccc")
abline(v = 500 / 100,  col = "#cccccc")
abline(v = 1000 / 100,  col = "#cccccc")
abline(v = 3000 / 100,  col = "#cccccc")
abline(v = 5000 / 100,  col = "#cccccc")
abline(v = 8000 / 100,  col = "#cccccc")

polygon(cip_x, cip_y_prim, col = adjustcolor("grey30", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_hybrid, col = adjustcolor("orange", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_update, col = adjustcolor("blue", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_replace, col = adjustcolor("forestgreen", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_louvain, col = adjustcolor("red", alpha.f = 0.3), border = FALSE)


lines(x, means_prim, col = "grey30")
points(x, means_prim, col = "grey30", pch = 16, cex=0.6)

lines(x, means_hybrid, col = "orange")
points(x, means_hybrid, col = "orange", pch = 16, cex=0.6)

lines(x, means_update, col = "blue")
points(x, means_update, col = "blue", pch = 16, cex=0.6)

lines(x, means_replace, col = "forestgreen")
points(x, means_replace, col = "forestgreen", pch = 16, cex=0.6)

lines(x, means_louvain, col = "red")
points(x, means_louvain, col = "red", pch = 16, cex=0.6)

legend("bottomright", legend = c("Louvain", "Replace", "Update", "Hybrid", "Primitive"),
       col = c("red", "forestgreen", "blue", "orange", "grey30"), pch = 16,
       bg = adjustcolor("white", alpha.f = 0.7), inset=0.01, cex = 1.3)


if (dosave) dev.off()

