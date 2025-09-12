library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- TRUE
max_epoch <- 300
n_runs <- 50
time_interval <- 2

output_file <- "hanoi.pdf"

if (dosave) pdf(output_file, width = 5, height = 4)

par(mfrow = c(1, 1), family = "serif", mar = c(3.5, 3.5, 1.5, 0.5) + 0.5, mgp = c(2.5, 1, 0))


input_file <- sprintf("hanoi.json")

dataj <- fromJSON(file = input_file)

means_0001 <- dataj$"0.001"$mean[1:(max_epoch / time_interval)]
sd_0001 <- dataj$"0.001"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_001 <- dataj$"0.01"$mean[1:(max_epoch / time_interval)]
sd_001 <- dataj$"0.01"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_01 <- dataj$"0.1"$mean[1:(max_epoch / time_interval)]
sd_01 <- dataj$"0.1"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_1 <- dataj$"1.0"$mean[1:(max_epoch / time_interval)]
sd_1 <- dataj$"1.0"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_10 <- dataj$"10.0"$mean[1:(max_epoch / time_interval)]
sd_10 <- dataj$"10.0"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_100 <- dataj$"100.0"$mean[1:(max_epoch / time_interval)]
sd_100 <- dataj$"100.0"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

x <- seq(1, max_epoch, by = time_interval)
cip_x <- c(x, rev(x))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_0001 <- c(means_0001 - sd_0001, rev(means_0001 + sd_0001))
cip_y_001 <- c(means_001 - sd_001, rev(means_001 + sd_001))
cip_y_01 <- c(means_01 - sd_01, rev(means_01 + sd_01))
cip_y_1 <- c(means_1 - sd_1, rev(means_1 + sd_1))
cip_y_10 <- c(means_10 - sd_10, rev(means_10 + sd_10))
cip_y_100 <- c(means_100 - sd_100, rev(means_100 + sd_100))

plot.new()
plot.window(ylim = c(-0.4, 0.9), xlim = c(0, max_epoch))
axis(1, at = c(0, 100, 200, 300), labels = TRUE, cex.axis = 1.7)
axis(2, at = c(-0.3, 0.0, 0.3, 0.6, 0.9), labels = TRUE, cex.axis = 1.7)
title(main = "Hanoi", cex.main = 2.0, font.main = 1)
title(xlab = "Epoch", cex.lab = 2.0)
title(ylab = "Return", cex.lab = 2.0)
box()

polygon(cip_x, cip_y_prim, col = adjustcolor("grey30", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_0001, col = adjustcolor("#440154", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_001, col = adjustcolor("#414487", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_01, col = adjustcolor("#2a788e", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_1, col = adjustcolor("#22a884", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_10, col = adjustcolor("#7ad151", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_100, col = adjustcolor("#fde725", alpha.f = 0.3), border = FALSE)


lines(x, means_prim, col = "grey30")
points(x, means_prim, col = "grey30", pch = 16)

lines(x, means_0001, col = "#440154")
points(x, means_0001, col = "#440154", pch = 16)

lines(x, means_001, col = "#414487")
points(x, means_001, col = "#414487", pch = 16)

lines(x, means_01, col = "#2a788e")
points(x, means_01, col = "#2a788e", pch = 16)

lines(x, means_1, col = "#22a884")
points(x, means_1, col = "#22a884", pch = 16)

lines(x, means_10, col = "#7ad151")
points(x, means_10, col = "#7ad151", pch = 16)

lines(x, means_100, col = "#fde725")
points(x, means_100, col = "#fde725", pch = 16)

legend("bottomright", inset=0.01,  legend = c("0.001", "0.01", "0.1", "1.0", "10.0", "100.0", "Primitive"),
       col = c("#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725", "grey30"), pch = 16,
       bg = adjustcolor("white", alpha.f = 0.7), cex = 1.5)

if (dosave) dev.off()

