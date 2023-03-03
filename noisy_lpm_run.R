#!/usr/bin/env Rscript
rm(list=ls())
path <- "~/Desktop/"
path_temp <- "~/Desktop/cpp/"
setwd(path)


require(Rcpp)
require(RcppArmadillo)
sourceCpp(file = "utils/lpm_generator.cpp")
sourceCpp(file = "utils/procrustes.cpp")
sourceCpp(file = "utils/save_and_load.cpp")
source("utils/lpm_utils.R")
source("utils/lpm_cpp_wrappers.R")
source("utils/lpm_noisy_cpp_wrappers.R")


set.seed(12345)


## Parallel computing
n_procs <- 2


## Global variables and hyperparameters
N <- 60 # number of nodes
D <- 2 # number of latent dimensions (always set to 2)
beta_mu <- 0 # prior mean on intercept parameter
beta_nu <- 1000 # prior std deviation on intercept parameter
theta_mu <- 0 # prior mean on theta parameter (determines the std deviation of latent positions)
theta_nu <- 100 # prior std deviation on theta parameter
gamma <- 1 # truncated Gaussian prior parameter
S <- 1 # box limit
hypers <- list(beta_mu = beta_mu, beta_nu = beta_nu, theta_mu = theta_mu, theta_nu = theta_nu, gamma = gamma, S = S)


## True model parameters
beta <- 0.5
theta <- log(3)
positions_true <- matrix(runif(N*D,-S,S),N,D)
positions_true[1,] = c(0,0)
pars <- list(beta = beta, theta = theta, z = positions_true)


## Grid parameters
M <- 8


## Generate a random LPM with given parameters
lpm <- lpm_generator_cpp(beta = beta, theta = theta, positions = positions_true)
edgelist <- lpm$edgelist + 1
L <- nrow(edgelist)
L / ( N*(N-1)/2 )


## Plot the lpm
if (N <= 500) plot_lpm(edgelist = edgelist, positions = positions_true, main = "LPM")


## Mcmc settings
n_iter <- 100 # net number of MCMC iterations in the final sample, after burnin and thin
burnin <- 10 # gross number of MCMC iterations, before thinning
thin <- 10 # MCMC thinning (keep 1 every "thin" iterations)
proposal_z <- rep(0.1,N) # std devs for latent position's proposal
proposal_beta <- 0.1 # std devs for beta's proposal
proposal_theta <- 0.1 # std devs for theta's proposal
beta_init <- 0 # initial values
theta_init <- 0 # initial values
z_init <- matrix(runif(N*D,-S,S),N,D) # initial values
init <- list(beta = beta_init, theta = theta_init, z = z_init)
mcmc <- list(n_iter = n_iter, burnin = burnin, thin = thin, proposal_z = proposal_z, proposal_beta = proposal_beta, proposal_theta = proposal_theta)


## Noisy LPM
grid <- list(M = M)
WriteInputNoisyLPM(path_temp = path_temp, edgelist = edgelist, grid = grid, init = init, mcmc = mcmc, hypers = hypers) # write everything into the corresponding files in the data folder
RunNoisyLPM(path_temp = path_temp, n_procs = n_procs, ignore.stdout = F) # run C++ code
res_lpm_noisy <- ReadOutputLPM(path_temp = path_temp, positions_true = positions_true) # read back the output from the output folder


## Save current workspace
save.image("results/results.RData")


## Generate all of the plots for the Noisy LPM
GeneratePlotsLPM(path_temp = path, mcmc_output = res_lpm_noisy, true_parameters = pars, write_to_files = T, filename_extra = "noisy_")
GenerateDistancePlotsLPM(path_temp = path, mcmc_output = res_lpm_noisy, true_parameters = pars, write_to_files = T, filename_extra = "noisy_")



