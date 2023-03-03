require(Rcpp)
require(RcppArmadillo)
sourceCpp(file = "utils/procrustes.cpp")
sourceCpp(file = "utils/save_and_load.cpp")


WriteInputLPM <- function(path_temp, edgelist, init, mcmc, hypers)
{
  current_wd <- getwd()
  setwd(path_temp)
  write.table(x = edgelist-1, file = "data/edgelist.csv", sep = ",", row.names = F, col.names = F)
  write.table(x = init$beta, file = "data/init/beta.csv", row.names = F, col.names = F)
  write.table(x = init$theta, file = "data/init/theta.csv", row.names = F, col.names = F)
  write.table(x = init$z, file = "data/init/positions.csv", sep = ",", row.names = F, col.names = F)
  write.table(x = c(mcmc$n_iter,mcmc$burnin,mcmc$thin), file = "data/mcmc/mcmc.csv", row.names = F, col.names = F)
  write.table(x = mcmc$proposal_z, file = "data/mcmc/proposal_z.csv", row.names = F, col.names = F)
  write.table(x = mcmc$proposal_beta, file = "data/mcmc/proposal_beta.csv", row.names = F, col.names = F)
  write.table(x = mcmc$proposal_theta, file = "data/mcmc/proposal_theta.csv", row.names = F, col.names = F)
  write.table(x = hypers$beta_mu, file = "data/hypers/beta_mu.csv", row.names = F, col.names = F)
  write.table(x = hypers$beta_nu, file = "data/hypers/beta_nu.csv", row.names = F, col.names = F)
  write.table(x = hypers$theta_mu, file = "data/hypers/theta_mu.csv", row.names = F, col.names = F)
  write.table(x = hypers$theta_nu, file = "data/hypers/theta_nu.csv", row.names = F, col.names = F)
  write.table(x = hypers$gamma, file = "data/hypers/gamma.csv", row.names = F, col.names = F)
  write.table(x = hypers$S, file = "data/hypers/S.csv", row.names = F, col.names = F)
  setwd(current_wd)
}


RunLPM <- function(path_temp, n_procs = 1, ignore.stdout = F)
{
  current_wd <- getwd()
  setwd(path_temp)
  command <- paste("mpirun --bind-to core --nooversubscribe -np",n_procs,"--report-bindings --display-allocation --display-map lpm")
  system(command = command, ignore.stdout = ignore.stdout, ignore.stderr = ignore.stdout, wait = T)
  setwd(current_wd)
}


ReadOutputLPM <- function(path_temp, positions_true = NULL)
{
  current_wd <- getwd()
  setwd(path_temp)
  z_sample <- LoadBinaryCube("output/z_sample.bin")
  beta_sample <- read.table(file = "output/beta_sample.csv")$V1
  theta_sample <- read.table(file = "output/theta_sample.csv")$V1
  acceptance_beta <- read.table(file = "output/acceptance_beta.csv")$V1
  acceptance_theta <- read.table(file = "output/acceptance_theta.csv")$V1
  acceptance_z <- read.table(file = "output/acceptance_z.csv")$V1
  posterior_values <- read.table(file = "output/posterior_values.csv")$V1
  n_iter <- length(posterior_values)
  N <- length(acceptance_z)
  ## Procrustes' trasformations
  map_index <- which.max(posterior_values)
  positions_map <- z_sample[,,map_index]
  if (!is.null(positions_true)) positions_map <- Procrustes(x_ref = positions_true, y = z_sample[,,map_index])
  positions_sample <- array(NA,c(N,2,n_iter))
  if (is.null(positions_true)) for (iter in 1:n_iter) positions_sample[,,iter] <- as.matrix(Procrustes(x_ref = positions_map, y = z_sample[,,iter]))
  else for (iter in 1:n_iter) positions_sample[,,iter] <- as.matrix(Procrustes(x_ref = positions_true, y = z_sample[,,iter]))
  positions_avg <- apply(X = positions_sample, MARGIN = c(1,2), FUN = mean)
  beta_avg <- mean(beta_sample)
  theta_avg <- mean(theta_sample)
  setwd(current_wd)
  list(beta_sample = beta_sample, theta_sample = theta_sample, positions_sample = positions_sample, map_index = map_index, beta_avg = beta_avg, theta_avg = theta_avg, positions_avg = positions_avg, acceptance_beta = acceptance_beta, acceptance_theta = acceptance_theta, acceptance_z = acceptance_z, posterior_values = posterior_values)
}


GeneratePlotsLPM <- function(path_temp, mcmc_output, true_parameters, write_to_files, filename_extra = "")
{
  current_wd <- getwd()
  setwd(path_temp)
  
  res <- mcmc_output
  
  beta <- true_parameters$beta
  theta <- true_parameters$theta
  positions_true <- true_parameters$z
  
  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_trace_posterior.pdf",sep=""), width=6,height=6, bg="white")
  plot(1:n_iter,res$posterior_values, type="l", xlab="Iteration", ylab="log-posterior", main="Posterior values at each iteration")
  if (write_to_files) dev.off()
  
  limit_average<- max(abs(res$positions_avg))
  limit_true <- max(abs(positions_true))
  limit_sample <- max(abs(res$positions_sample))

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_trace_latent_positions.pdf",sep=""), width=6,height=6, bg="white")
  i <- 1
  for (i in 1:N) {
    sampled_points <- t(res$positions_sample[i,,])
    plot(sampled_points, type="n", xlim=c(-limit_sample,limit_sample), ylim=c(-limit_sample,limit_sample), xlab="Dimension 1", ylab="Dimension 2", main=paste("Sampled positions of node",i))
    abline(h=0,v=0,col=2,lty=2)
    points(sampled_points, pch=20, cex=0.1)
    points(res$positions_avg[i,1],res$positions_avg[i,2],col=5,pch=20,cex=1)
    points(res$positions_map[i,1],res$positions_map[i,2],col=4,pch=20,cex=1)
    points(positions_true[i,1],positions_true[i,2],col=2,pch=20,cex=1)
  }
  if (write_to_files) dev.off()

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_trace_beta.pdf",sep=""), width=6,height=6, bg="white")
  plot(1:n_iter,res$beta_sample, type="l", xlab="Iteration", ylab="beta", main="Sample for beta")
  abline(h=res$beta_avg, col=5)
  abline(h=beta, col=2)
  if (write_to_files) dev.off()

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_trace_theta.pdf",sep=""), width=6,height=6, bg="white")
  plot(1:n_iter,res$theta_sample, type="l", xlab="Iteration", ylab="theta", main="Sample for theta")
  abline(h=res$theta_avg, col=5)
  abline(h=theta, col=2)
  if (write_to_files) dev.off()

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_network_true.pdf",sep=""), width=6,height=6, bg="white")
  plot(positions_true, type="n", xlim=c(-limit_true,limit_true), ylim=c(-limit_true,limit_true), xlab="Dimension 1", ylab="Dimension 2", main="True network")
  points(positions_true, pch=20, cex=1)
  abline(h=0,v=0, col=2, lty=2)
  for (l in 1:nrow(edgelist)) segments(x0 = positions_true[edgelist[l,1],1], y0 = positions_true[edgelist[l,1],2], x1 = positions_true[edgelist[l,2],1], positions_true[edgelist[l,2],2], lwd=0.075)
  if (write_to_files) dev.off()

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_network_average.pdf",sep=""), width=6,height=6, bg="white")
  plot(res$positions_avg, type="n", xlim=c(-limit_average,limit_average), ylim=c(-limit_average,limit_average), xlab="Dimension 1", ylab="Dimension 2", main="Average network")
  points(res$positions_avg, pch=20, cex=1)
  abline(h=0,v=0, col=2, lty=2)
  for (l in 1:nrow(edgelist)) segments(x0 = res$positions_avg[edgelist[l,1],1], y0 = res$positions_avg[edgelist[l,1],2], x1 = res$positions_avg[edgelist[l,2],1], res$positions_avg[edgelist[l,2],2], lwd=0.075)
  if (write_to_files) dev.off()

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_network_true_to_average.pdf",sep=""), width=6,height=6, bg="white")
  plot(positions_true,type="n",pch=20,col=2,xlim=c(-limit_true,limit_true),ylim=c(-limit_true,limit_true), xlab="Dimension 1", ylab="Dimension 2", main="True to average positions")
  abline(h=0,v=0,lty=2)
  points(positions_true,pch=20,col=2)
  points(res$positions_avg,pch=20,col=3)
  for (i in 1:N) segments(x0 = positions_true[i,1], y0 = positions_true[i,2], x1 = res$positions_avg[i,1], y1 = res$positions_avg[i,2], lwd = 0.5)
  if (write_to_files) dev.off()

  setwd(current_wd)
}


GenerateDistancePlotsLPM <- function(path_temp, mcmc_output, true_parameters, write_to_files, filename_extra = "")
{
  current_wd <- getwd()
  setwd(path_temp)
  
  res <- mcmc_output
  
  beta <- true_parameters$beta
  theta <- true_parameters$theta
  positions_true <- true_parameters$z
  
  N <- nrow(positions_true)
  distances_true <- distances_mcmc <- edge_probabilities_true <- edge_probabilities_mcmc <- matrix(NA,N,N)
  for (i in 1:N) for (j in 1:N)
  {
    distances_true[i,j] = distance(positions_true[i,],positions_true[j,])
    distances_mcmc[i,j] = distance(res$positions_avg[i,],res$positions_avg[j,])
    edge_probabilities_true[i,j] = edge_probability(beta,theta,distances_true[i,j])
    edge_probabilities_mcmc[i,j] = edge_probability(res$beta_avg,res$theta_avg,distances_mcmc[i,j])
  }

  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_error_mcmc_distances.pdf",sep=""), width=6,height=6, bg="white")
  plot(as.numeric(distances_true),as.numeric(distances_mcmc), type = "n", xlab = "True", ylab = "Posterior mean", main = "Errors on latent distances")
  points(as.numeric(distances_true),as.numeric(distances_mcmc), pch = 20, cex = 0.5)
  abline(a=0,b=1,col=2)
  if (write_to_files) dev.off()
  
  if (write_to_files) pdf(paste("results/plots/",filename_extra,"_error_mcmc_probabilities.pdf",sep=""), width=6,height=6, bg="white")
  plot(as.numeric(edge_probabilities_true),as.numeric(edge_probabilities_mcmc), type = "n", xlab = "True", ylab = "Posterior mean", main = "Errors on edge probabilities")
  points(as.numeric(edge_probabilities_true),as.numeric(edge_probabilities_mcmc), pch = 20, cex = 0.5)
  abline(a=0,b=1,col=2)
  if (write_to_files) dev.off()
  
  setwd(current_wd)
  
}




