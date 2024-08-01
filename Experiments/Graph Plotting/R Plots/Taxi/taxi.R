# Set your current working directory to this script's location.

library("rjson")
library("here")

here()

path <- getwd()
options(digits = 3)
dosave <- T
max_epoch = 100

### get the data
dataj <- fromJSON(file="taxi.json")
means_louvain <- dataj$louvain$mean[1:max_epoch]
sd_louvain <- dataj$`louvain`$std_dev[1:max_epoch] / sqrt(40)
#
means_prim <- dataj$primitive$mean[1:max_epoch]
sd_prim <- dataj$primitive$std_dev[1:max_epoch] / sqrt(40)
#
means_labelprop <- dataj$`label_prop`$mean[1:max_epoch]
sd_labelprop <- dataj$`label_prop`$std_dev[1:max_epoch] / sqrt(40)
#
means_edgebet <- dataj$`edge_betweenness`$mean[1:max_epoch]
sd_edgebet <- dataj$`edge_betweenness`$std_dev[1:max_epoch] / sqrt(40)
#
means_nodebet <- dataj$`node_betweenness`$mean[1:max_epoch]
sd_nodebet <- dataj$`node_betweenness`$std_dev[1:max_epoch] / sqrt(40)
#
means_eigen <- dataj$eigenoptions$mean[1:max_epoch]
sd_eigen <- dataj$eigenoptions$std_dev[1:max_epoch] / sqrt(40)
#
means_xu <- dataj$`level_3`$mean[1:max_epoch]
sd_xu <- dataj$`level_3`$std_dev[1:max_epoch] / sqrt(40)
######

# compute confidence intervals
x <- 1:max_epoch
cip_x <- c(x, rev(x))
cip_y_louvain <- c(means_louvain - sd_louvain, rev(means_louvain + sd_louvain))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_labelprop <- c(means_labelprop - sd_labelprop, rev(means_labelprop + sd_labelprop))
cip_y_edgebet <- c(means_edgebet - sd_edgebet, rev(means_edgebet + sd_edgebet))
cip_y_nodebet <- c(means_nodebet - sd_nodebet, rev(means_nodebet + sd_nodebet))
cip_y_eigen <- c(means_eigen - sd_eigen, rev(means_eigen + sd_eigen))
cip_y_xu <- c(means_xu - sd_xu, rev(means_xu + sd_xu))

# to make polygon where coordinates start with lower limit and then upper limit in reverse order
# polygon(ci_x,ci_y, col = "grey75", border = FALSE)

if (dosave) pdf(paste("taxi", ".pdf",sep=""), width = 5, height = 4.25)
par(family="serif", mar=c(3.5, 3.5, 1.5, 0) + 0.5, mgp=c(2.5, 1, 0))
plot.new()
plot.window(ylim=c(-0.7, 1.0), xlim=c(0,100))
xlabels = c(0, 20, 40, 60, 80, 100)
axis(1, at=xlabels, labels=xlabels, cex.axis=1.7)
ylabels <- c(-0.5, 0, 0.5, 1)
axis(2, at=ylabels, labels=ylabels, cex.axis=1.7)
title(main=paste("Taxi"), cex.main=2.0, font.main=1)
title(xlab="Epoch", cex.lab=2.0)
title(ylab="Return", cex.lab=2.0)
box()

polygon(cip_x,cip_y_louvain, col = adjustcolor("red", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_prim, col = adjustcolor("grey30", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_labelprop, col = adjustcolor("forestgreen", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_edgebet, col = adjustcolor("blue", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_nodebet, col = adjustcolor("navy", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_eigen, col = adjustcolor("orange", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_xu, col = adjustcolor("darkviolet", alpha.f=0.3), border = FALSE)



points(x, means_louvain, col = "red", pch = 16, xpd=T)
lines(x, means_louvain, col = "red", xpd=T)

points(x, means_prim, col = "grey30", pch = 16, xpd=T)
lines(x, means_prim, col = "grey30", xpd=T)

points(x, means_labelprop, col = "forestgreen", pch = 16, xpd=T)
lines(x, means_labelprop, col = "forestgreen", xpd=T)

points(x, means_edgebet, col = "blue", pch = 16, xpd=T)
lines(x, means_edgebet, col = "blue", xpd=T)

points(x, means_nodebet, col = "navy", pch = 16, xpd=T)
lines(x, means_nodebet, col = "navy", xpd=T)

points(x, means_eigen, col = "orange", pch = 16, xpd=T)
lines(x, means_eigen, col = "orange", xpd=T)

points(x, means_xu, col = "darkviolet", pch = 16, xpd=T)
lines(x, means_xu, col = "darkviolet", xpd=T)


if (dosave) dev.off()
