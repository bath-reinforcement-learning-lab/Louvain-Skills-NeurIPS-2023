# Set your current working directory to this script's location.

library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- T

input_file = "scaling_results.json"
output_file = "performance_scaling.pdf"

### get the data
dataj <- fromJSON(file=input_file)
#
louvain_x <- sapply(dataj$louvain, function(point) point[1])
louvain_y <- sapply(dataj$louvain, function(point) point[2])
#
prim_x <- sapply(dataj$primitive, function(point) point[1])
prim_y <- sapply(dataj$primitive, function(point) point[2])
#
nodebet_x <- sapply(dataj$`node_betweenness`, function(point) point[1])
nodebet_y <- sapply(dataj$`node_betweenness`, function(point) point[2])
#
labelprop_x <- sapply(dataj$`label_prop`, function(point) point[1])
labelprop_y <- sapply(dataj$`label_prop`, function(point) point[2])
######

x <- c(2000, 5000, 10000, 20000, 50000)

if (dosave) pdf(output_file, width = 7, height = 5)
#par(family="serif", mar=c(3.5, 3.5, 1.5, 0) + 0.5, mgp=c(2.5, 1, 0))
par(family="serif", mar=c(3.5, 3.5, 0, 0) + 0.5, mgp=c(2.5, 1, 0))
plot.new()
plot.window(ylim=c(90000, 16000000), xlim=c(2500, 50000 + 100), log="x")
xlabels = c(2500, 5000, 10000, 20000, 50000)
yticks <- c(100000, 1000000, 10000000)
ylabels <- c(as.expression(bquote(10^ .(5))), as.expression(bquote(10^ .(6))), as.expression(bquote(10^ .(7))))
axis(1, at=xlabels, labels=xlabels, cex.axis=1.5)
axis(2, at=yticks, labels=ylabels, cex.axis=1.5, las="1")
title(xlab="Number of States", cex.lab=2.0)
title(ylab="Decision Stages", cex.lab=2.0)
box()

points(louvain_x, louvain_y, col = "red", pch = 16, xpd=T)
lines(louvain_x, louvain_y, col = "red", xpd=T)

points(prim_x, prim_y, col = "grey30", pch = 16, xpd=T)
lines(prim_x, prim_y, col = "grey30", xpd=T)

points(labelprop_x, labelprop_y, col = "forestgreen", pch = 16, xpd=T)
lines(labelprop_x, labelprop_y, col = "forestgreen", xpd=T)

points(nodebet_x, nodebet_y, col = "navy", pch = 16, xpd=T)
lines(nodebet_x, nodebet_y, col = "navy", xpd=T)

nice_names <- c("Louvain",
                "Eigenoptions",
                "Label Prop.",
                "Node Bet.",
                "Primitive")

ii <- c(1, 2, 3, 4, 5)
legend(2272, 1700000, nice_names[ii], col=c("red", "orange", "forestgreen", "navy", "grey30"),
       bg = adjustcolor("white", alpha.f=0.7), pch=16, xjust=0, yjust=0, cex=1.2)

if (dosave) dev.off()
