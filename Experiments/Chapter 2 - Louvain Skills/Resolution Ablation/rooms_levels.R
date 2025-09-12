library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- T

library("rjson")

### get the data
dataj <- fromJSON(file="rooms_levels.json")
resolutions <- dataj$`resolutions`
level1 <- dataj$`Level 1`
level2 <- dataj$`Level 2`
level3 <- dataj$`Level 3`
level4 <- dataj$`Level 4`

level5 <- dataj$`Level 5`
######

if (dosave) pdf(paste(path,"/rooms_levels", ".pdf",sep=""), width = 5, height = 4)
par(family="serif", mar=c(3.5, 3.5, 1.5, 0) + 0.5, mgp=c(2.5, 1, 0), xpd=TRUE)
plot.new()
plot.window(ylim=c(2, 110), xlim=c(0.00001, 10), log="xy")



# Set up X-Axis.
x_ticks = c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
x_labels = sapply(-5:1, function(i) # We have to do some string manipulation wizardry
  as.expression(bquote(10^ .(i)))) # to produce nice-looking log axes.
axis(1, at=x_ticks, labels=x_labels, cex.axis=1.7)

# Set up Y-Axis.
ylabels = c(2, 8, 22, 60)
axis(2, at = ylabels, labels=ylabels, cex.axis=1.7)
title(main="Rooms", cex.main=2.0, font.main=1)
title(xlab="Resolution Parameter", cex.lab=2.0)
title(ylab="Nodes Per Cluster", cex.lab=2.0)
box()

points(head(resolutions, length(level1)), level1, col = "#ff0000", pch = 16, xpd=T)
lines(head(resolutions, length(level1)), level1, col = "#ff0000", xpd=T)

points(head(resolutions, length(level2)), level2, col = "#00ff00", pch = 16, xpd=T)
lines(head(resolutions, length(level2)), level2, col = "#00ff00", xpd=T)

points(head(resolutions, length(level3)), level3, col = "#0000ff", pch = 16, xpd=T)
lines(head(resolutions, length(level3)), level3, col = "#0000ff", xpd=T)

points(head(resolutions, length(level4)), level4, col = "#87cefa", pch = 16, xpd=T)
lines(head(resolutions, length(level4)), level4, col = "#87cefa", xpd=T)

points(head(resolutions, length(level5)), level5, col = "darkviolet", pch = 16, xpd=T)
lines(head(resolutions, length(level5)), level5, col = "darkviolet", xpd=T)

arrows(0.1, 0, 10, 81, col="black")

if (dosave) dev.off()

