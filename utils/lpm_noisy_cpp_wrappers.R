require(Rcpp)
require(RcppArmadillo)
sourceCpp(file = "utils/procrustes.cpp")
sourceCpp(file = "utils/save_and_load.cpp")
source("utils/lpm_cpp_wrappers.R")


WriteInputNoisyLPM <- function(path_temp, edgelist, grid, init, mcmc, hypers)
{
  WriteInputLPM(path_temp = path_temp, edgelist = edgelist, init = init, mcmc = mcmc, hypers = hypers)
  current_wd <- getwd()
  setwd(path_temp)
  write.table(x = grid$M, file = "data/grid/M.csv", row.names = F, col.names = F)
  setwd(current_wd)
}


RunNoisyLPM <- function(path_temp, n_procs = 1, ignore.stdout = F)
{
  current_wd <- getwd()
  setwd(path_temp)
  command <- paste("mpirun --bind-to core --nooversubscribe -np",n_procs,"--report-bindings --display-allocation --display-map noisy_lpm")
  system(command = command, ignore.stdout = ignore.stdout, ignore.stderr = ignore.stdout, wait = T)
  setwd(current_wd)
}


