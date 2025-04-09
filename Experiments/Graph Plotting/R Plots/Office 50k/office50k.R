# Set your current working directory to this script's location.

library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- T
max_epoch = 20000
n_runs = 4
time_interval = 40

evaluation_type = "Episode"
input_file = sprintf("office50k_binary_%s.json", evaluation_type)
output_file = "office50k.pdf"

### get the data
dataj <- fromJSON(file=input_file)
means_louvain <- dataj$louvain$mean[1:(max_epoch / time_interval)]
sd_louvain <- dataj$`louvain`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_prim <- dataj$primitive$mean[1:(max_epoch / time_interval)]
sd_prim <- dataj$primitive$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
#
means_nodebet <- dataj$`node_betweenness`$mean[1:(max_epoch / time_interval)]
sd_nodebet <- dataj$`node_betweenness`$std_dev[1:(max_epoch / time_interval)] / sqrt(n_runs)
######

# compute confidence intervals
x <- seq(1, max_epoch, by=time_interval)
cip_x <- c(x, rev(x))
cip_y_louvain <- c(means_louvain - sd_louvain, rev(means_louvain + sd_louvain))
cip_y_prim <- c(means_prim - sd_prim, rev(means_prim + sd_prim))
cip_y_nodebet <- c(means_nodebet - sd_nodebet, rev(means_nodebet + sd_nodebet))

# to make polygon where coordinates start with lower limit and then upper limit in reverse order
# polygon(ci_x,ci_y, col = "grey75", border = FALSE)

if (dosave) pdf(output_file, width = 5, height = 4.25)
par(family="serif", mar=c(3.5, 3.5, 1.5, 0) + 0.5, mgp=c(2.5, 1, 0))
plot.new()
plot.window(ylim=c(-2.0, 0.5), xlim=c(0, max_epoch + 100))
xlabels = c(0, 4, 8, 12, 16, 20)
xlabels_at = c(0, 4000, 8000, 12000, 16000, 20000)
axis(1, at=xlabels_at, labels=xlabels, cex.axis=1.7)
ylabels <- c(-2.0, -1.0, 0.0)
axis(2, at=ylabels, labels=ylabels, cex.axis=1.7)
title(main=paste("Office (50k)"), cex.main=2.0, font.main=1)
title(xlab="Epoch (×10³)", cex.lab=2.0)
title(ylab="Return", cex.lab=2.0)
box()

polygon(cip_x,cip_y_louvain, col = adjustcolor("red", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_prim, col = adjustcolor("grey30", alpha.f=0.3), border = FALSE)
polygon(cip_x,cip_y_nodebet, col = adjustcolor("navy", alpha.f=0.3), border = FALSE)



points(x, means_louvain, col = "red", pch = 16, xpd=T)
lines(x, means_louvain, col = "red", xpd=T)

points(x, means_prim, col = "grey30", pch = 16, xpd=T)
lines(x, means_prim, col = "grey30", xpd=T)

points(x, means_nodebet, col = "navy", pch = 16, xpd=T)
lines(x, means_nodebet, col = "navy", xpd=T)

text(x = 11500, y = -1.8, labels = "-0.01 per action\n+1.0 at goal", font = 2, adj = 0)

if (dosave) dev.off()
