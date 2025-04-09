library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- T
max_epoch = 60
n_runs = 40
time_interval = 1

evaluation_type = "Episode"
input_file = sprintf("rooms_%s.json", evaluation_type)
output_file = "rooms_alt.pdf"

### get the data
dataj <- fromJSON(file=input_file)
means_louvain <- dataj$louvain$mean[1:(max_epoch / time_interval)]
sd_louvain <- dataj$`louvain`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_flatlouvain <- dataj$`louvain_flat`$mean[1:(max_epoch / time_interval)]
sd_flatlouvain <- dataj$`louvain_flat`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_level1 <- dataj$`level_1`$mean[1:(max_epoch / time_interval)]
sd_level1 <- dataj$`level_1`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_level2 <- dataj$`level_2`$mean[1:(max_epoch / time_interval)]
sd_level2 <- dataj$`level_2`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_level3 <- dataj$`level_3`$mean[1:(max_epoch / time_interval)]
sd_level3 <- dataj$`level_3`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
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

if (dosave) pdf(output_file, width = 5, height = 4.25)
par(family="serif", mar=c(3.5, 3.5, 1.5, 0) + 0.5, mgp=c(2.5, 1, 0))
plot.new()
plot.window(ylim=c(-0.1, 1.0), xlim=c(0,60))
xlabels = c(10, 20, 30, 40, 50, 60)
ylabels = c(0.0, 0.5, 1.0)
axis(1, at=xlabels, labels=xlabels, cex.axis=1.7)
axis(2, at=ylabels, labels=ylabels, cex.axis=1.7)
title(main=paste("Rooms"), cex.main=2.0, font.main=1)
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

nice_names <- c("Louvain",
                "Louvain flat",
                "Level 1",
                "Level 2",
                "Level 3",
                "Level 4",
                "Primitive")

ii <- c(1, 2, 3, 4, 5, 6, 7)
legend(36, -0.1, nice_names[ii], col=c("red", "darkred", "mediumpurple4", "mediumpurple2", "mediumorchid2", "orchid1", "grey30"),
       bg = adjustcolor("white", alpha.f=0.7), pch=16, xjust=0, yjust=0, cex=1.3)

if (dosave) dev.off()
