library("rjson")
library("rstudioapi")
library("latex2exp")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- TRUE
max_epoch <- 300
n_runs <- 50
time_interval <- 1

output_file <- "playroom_alt.pdf"

if (dosave) pdf(output_file, width = 5, height = 4.25)

par(mfrow = c(1, 1), family = "serif", mar = c(3.5, 3.5, 1.5, 0.5) + 0.5, mgp = c(2.5, 1, 0))


input_file <- sprintf("playroom.json")

dataj <- fromJSON(file = input_file)

means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_001 <- dataj$"modularity (0.01)"$mean[1:(max_epoch / time_interval)]
sd_001 <- dataj$"modularity (0.01)"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_01 <- dataj$"modularity (0.1)"$mean[1:(max_epoch / time_interval)]
sd_01 <- dataj$"modularity (0.1)"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_1 <- dataj$"modularity (1.0)"$mean[1:(max_epoch / time_interval)]
sd_1 <- dataj$"modularity (1.0)"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_10 <- dataj$"modularity (10.0)"$mean[1:(max_epoch / time_interval)]
sd_10 <- dataj$"modularity (10.0)"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

x <- seq(1, max_epoch, by = time_interval)
cip_x <- c(x, rev(x))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_001 <- c(means_001 - sd_001, rev(means_001 + sd_001))
cip_y_01 <- c(means_01 - sd_01, rev(means_01 + sd_01))
cip_y_1 <- c(means_1 - sd_1, rev(means_1 + sd_1))
cip_y_10 <- c(means_10 - sd_10, rev(means_10 + sd_10))

plot.new()
plot.window(ylim = c(-0.4, 0.8), xlim = c(0, max_epoch))
axis(1, at = c(0, 100, 200, 300), labels = TRUE, cex.axis = 1.7)
axis(2, at = c(-0.4, 0.0, 0.4, 0.8), labels = TRUE, cex.axis = 1.7)
title(main = "Playroom", cex.main = 2.0, font.main = 1)
title(xlab = "Epoch", cex.lab = 2.0)
title(ylab = "Return", cex.lab = 2.0)
box()

polygon(cip_x, cip_y_prim, col = adjustcolor("grey30", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_001, col = adjustcolor("mediumorchid2", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_01, col = adjustcolor("darkmagenta", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_1, col = adjustcolor("mediumpurple2", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_10, col = adjustcolor("mediumpurple4", alpha.f = 0.3), border = FALSE)



lines(x, means_prim, col = "grey30")
points(x, means_prim, col = "grey30", pch = 16)

lines(x, means_001, col = "mediumorchid2")
points(x, means_001, col = "mediumorchid2", pch = 16)

lines(x, means_01, col = "darkmagenta")
points(x, means_01, col = "darkmagenta", pch = 16)

lines(x, means_1, col = "mediumpurple2")
points(x, means_1, col = "mediumpurple2", pch = 16)

lines(x, means_10, col = "mediumpurple4")
points(x, means_10, col = "mediumpurple4", pch = 16)


#legend(x = 31, y = 0.52, legend = c(TeX(r'($\rho = 0.01$)'), TeX(r'($\rho = 0.1$)'), TeX(r'($\rho = 1.0$)'), TeX(r'($\rho = 10.0$)'),  "Primitive"),
#       col = c("mediumorchid2", "darkmagenta", "mediumpurple2", "mediumpurple4", "grey30"), pch = 16,
#       bg = adjustcolor("white", alpha.f = 0.7), cex = 1.5)


if (dosave) dev.off()

