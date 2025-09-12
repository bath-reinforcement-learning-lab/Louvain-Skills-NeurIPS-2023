library("rjson")
library("rstudioapi")

setwd(dirname(getActiveDocumentContext()$path))

path <- getwd()
options(digits = 3)
dosave <- T

### get the data
dataj <- fromJSON(file="skill_availability_scaling.json")
stg_sizes <- dataj$`STG Sizes`
available_skills <- dataj$`Available Skills`
######

if (dosave) pdf(paste(path,"/availability_scaling", ".pdf",sep=""), width = 5, height = 3.75)
par(family="serif", mar=c(3.5, 3.5, 0, 0) + 0.5, mgp=c(2.5, 1, 0))
plot.new()
plot.window(ylim=c(9.5, 18), xlim=c(1000, 1200000), log="x") 

# Set up X-Axis.
x_ticks = c(1000, 10000, 100000, 1000000)
x_labels = sapply(3:6, function(i) # We have to do some string manipulation wizardry
  as.expression(bquote(10^ .(i)))) # to produce nice-looking log axes.
axis(1, at=x_ticks, labels=x_labels, cex.axis=1.7)

# Set up Y-Axis.
ylabels = c(10, 12, 14, 16, 18, 20)
axis(2, at = ylabels, labels=ylabels, cex.axis=1.7)

# Set up titles.
# title(main="Hierarchy Depth Scaling", cex.main=2.0, font.main=1)
title(xlab="Number of States", cex.lab=2.0)
title(ylab="Available Skills", cex.lab=2.0)
box()

points(stg_sizes, available_skills + 1, col = "blue", pch = 16, xpd=T)
lines(stg_sizes, available_skills + 1, col = "blue", xpd=T)

if (dosave) dev.off()
