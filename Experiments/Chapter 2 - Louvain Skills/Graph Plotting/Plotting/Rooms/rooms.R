library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- TRUE
max_epoch <- 60
n_runs <- 50
time_interval <- 1

output_file <- "rooms.pdf"

if (dosave) pdf(output_file, width = 5, height = 4.25)

par(mfrow = c(1, 1), family = "serif", mar = c(3.5, 3.5, 1.5, 0.5) + 0.5, mgp = c(2.5, 1, 0))


input_file <- sprintf("rooms.json")

dataj <- fromJSON(file = input_file)

means_louvain <- dataj$"louvain"$mean[1:(max_epoch / time_interval)]
sd_louvain <- dataj$"louvain"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_lab_prop <- dataj$"label propagation"$mean[1:(max_epoch / time_interval)]
sd_lab_prop <- dataj$"label propagation"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_edge_bet <- dataj$"edge betweenness"$mean[1:(max_epoch / time_interval)]
sd_edge_bet <- dataj$"edge betweenness"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_node_bet <- dataj$"betweenness"$mean[1:(max_epoch / time_interval)]
sd_node_bet <- dataj$"betweenness"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_eig <- dataj$"eigenoptions"$mean[1:(max_epoch / time_interval)]
sd_eig <- dataj$"eigenoptions"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

means_xu <- dataj$"xu"$mean[1:(max_epoch / time_interval)]
sd_xu <- dataj$"xu"$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)

x <- seq(1, max_epoch, by = time_interval)
cip_x <- c(x, rev(x))
cip_y_louvain <- c(means_louvain - sd_louvain, rev(means_louvain + sd_louvain))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_lab_prop <- c(means_lab_prop - sd_lab_prop, rev(means_lab_prop + sd_lab_prop))
cip_y_edge_bet <- c(means_edge_bet - sd_edge_bet, rev(means_edge_bet + sd_edge_bet))
cip_y_node_bet <- c(means_node_bet - sd_node_bet, rev(means_node_bet + sd_node_bet))
cip_y_eig <- c(means_eig - sd_eig, rev(means_eig + sd_eig))
cip_y_xu <- c(means_xu - sd_xu, rev(means_xu + sd_xu))

plot.new()
plot.window(ylim = c(-0.4, 0.9), xlim = c(0, max_epoch))
axis(1, at = c(0, 10, 20, 30, 40, 50, 60, 70), labels = TRUE, cex.axis = 1.7)
axis(2, at = c(-0.4, 0.0, 0.4, 0.8), labels = TRUE, cex.axis = 1.7)
title(main = "Rooms", cex.main = 2.0, font.main = 1)
title(xlab = "Epoch", cex.lab = 2.0)
title(ylab = "Return", cex.lab = 2.0)
box()

polygon(cip_x, cip_y_prim, col = adjustcolor("grey30", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_lab_prop, col = adjustcolor("forestgreen", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_edge_bet, col = adjustcolor("blue", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_node_bet, col = adjustcolor("navy", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_eig, col = adjustcolor("orange", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_xu, col = adjustcolor("darkviolet", alpha.f = 0.3), border = FALSE)
polygon(cip_x, cip_y_louvain, col = adjustcolor("red", alpha.f = 0.3), border = FALSE)


lines(x, means_prim, col = "grey30")
points(x, means_prim, col = "grey30", pch = 16)

lines(x, means_lab_prop, col = "forestgreen")
points(x, means_lab_prop, col = "forestgreen", pch = 16)

lines(x, means_edge_bet, col = "blue")
points(x, means_edge_bet, col = "blue", pch = 16)

lines(x, means_node_bet, col = "navy")
points(x, means_node_bet, col = "navy", pch = 16)

lines(x, means_eig, col = "orange")
points(x, means_eig, col = "orange", pch = 16)

lines(x, means_xu, col = "darkviolet")
points(x, means_xu, col = "darkviolet", pch = 16)

lines(x, means_louvain, col = "red")
points(x, means_louvain, col = "red", pch = 16)

#legend(x = 25, y = 0.8, legend = c("Louvain", "Primitive", "Label Prop.", "Edge Bet.", "Node Bet.", "Eigenoptions", "Xu et al."),
#       col = c("red", "grey30", "forestgreen", "blue", "navy", "orange", "darkviolet"), pch = 16,
#       bg = adjustcolor("white", alpha.f = 0.7), cex = 1.5)


if (dosave) dev.off()

