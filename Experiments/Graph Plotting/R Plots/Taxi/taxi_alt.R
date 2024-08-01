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
means_flatlouvain <- dataj$`louvain_flat`$mean[1:max_epoch]
sd_flatlouvain <- dataj$`louvain_flat`$std_dev[1:max_epoch] / sqrt(40)
#
means_level1 <- dataj$`level_1`$mean[1:max_epoch]
sd_level1 <- dataj$`level_1`$std_dev[1:max_epoch] / sqrt(40)
#
means_level2 <- dataj$`level_2`$mean[1:max_epoch]
sd_level2 <- dataj$`level_2`$std_dev[1:max_epoch] / sqrt(40)
#
means_level3 <- dataj$`level_3`$mean[1:max_epoch]
sd_level3 <- dataj$`level_3`$std_dev[1:max_epoch] / sqrt(40)
######

# compute confidence intervals
x <- 1:max_epoch
cip_x <- c(x, rev(x))
cip_y_louvain <- c(means_louvain - sd_louvain, rev(means_louvain + sd_louvain))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_flatlouvain <- c(means_flatlouvain - sd_flatlouvain, rev(means_flatlouvain + sd_flatlouvain))
cip_y_level1 <- c(means_level1 - sd_level1, rev(means_level1 + sd_level1))
cip_y_level2 <- c(means_level2 - sd_level2, rev(means_level2 + sd_level2))
cip_y_level3 <- c(means_level3 - sd_level3, rev(means_level3 + sd_level3))

# to make polygon where coordinates start with lower limit and then upper limit in reverse order
# polygon(ci_x,ci_y, col = "grey75", border = FALSE)

if (dosave) pdf(paste("taxi_alt", ".pdf",sep=""), width = 5, height = 4.25)
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

polygon(cip_x,cip_y_louvain, col = adjustcolor("red", alpha.f=0.2), border = FALSE)
polygon(cip_x,cip_y_prim, col = adjustcolor("grey30", alpha.f=0.2), border = FALSE)
polygon(cip_x,cip_y_flatlouvain, col = adjustcolor("darkred", alpha.f=0.2), border = FALSE)
polygon(cip_x,cip_y_level1, col = adjustcolor("mediumpurple4", alpha.f=0.2), border = FALSE)
polygon(cip_x,cip_y_level2, col = adjustcolor("mediumpurple2", alpha.f=0.2), border = FALSE)
polygon(cip_x,cip_y_level3, col = adjustcolor("mediumorchid2", alpha.f=0.2), border = FALSE)

points(x, means_louvain, col = "red", pch = 16, xpd=T)
lines(x, means_louvain, col = "red", xpd=T)

points(x, means_prim, col = "grey30", pch = 16, xpd=T)
lines(x, means_prim, col = "grey30", xpd=T)

points(x, means_flatlouvain, col = "darkred", pch = 16, xpd=T)
lines(x, means_flatlouvain, col = "darkred", xpd=T)

points(x, means_level1, col = "mediumpurple4", pch = 16, xpd=T)
lines(x, means_level1, col = "mediumpurple4", xpd=T)

points(x, means_level2, col = "mediumpurple2", pch = 16, xpd=T)
lines(x, means_level2, col = "mediumpurple2", xpd=T)

points(x, means_level3, col = "mediumorchid2", pch = 16, xpd=T)
lines(x, means_level3, col = "mediumorchid2", xpd=T)

if (dosave) dev.off()
