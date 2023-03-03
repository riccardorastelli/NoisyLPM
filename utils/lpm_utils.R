
distance <- function(x,y) sqrt( (x[1]-y[1])^2 + (x[2]-y[2])^2 )

edge_probability <- function(beta,theta,d) exp(beta-exp(theta)*d) / (1+exp(beta-exp(theta)*d))

plot_lpm <- function(edgelist, positions, main = "") {
  limit <- max(abs(positions))
  plot(positions, type="n", xlim=c(-limit,limit), ylim=c(-limit,limit), xlab="Dimension 1", ylab="Dimension 2", main=main)
  abline(h=0,v=0, col=2, lty=2)
  points(positions, pch=20, cex=1)
  for (l in 1:nrow(edgelist)) segments(x0 = positions[edgelist[l,1],1], y0 = positions[edgelist[l,1],2], x1 = positions[edgelist[l,2],1], positions[edgelist[l,2],2], lwd=0.075)
}


