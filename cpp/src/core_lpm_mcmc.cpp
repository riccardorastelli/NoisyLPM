#include "core_lpm_mcmc.h"

lpm_mcmc::lpm_mcmc(lpm_noisy network_)
{
  network = network_;
  arma::vec mcmc_vec;
  mcmc_vec.load("data/mcmc/mcmc.csv", arma::csv_ascii);
  net_n_iter = mcmc_vec.at(0);
  burnin = mcmc_vec.at(1);
  thin = mcmc_vec.at(2);
  total_n_iter = burnin + net_n_iter*thin;
  z_sample.set_size(network.N,network.D,net_n_iter);
  z_sample.fill(0);
  beta_sample.set_size(net_n_iter);
  beta_sample.fill(0);
  theta_sample.set_size(net_n_iter);
  theta_sample.fill(0);
  posterior_values.set_size(net_n_iter);
  
  proposal_z.load("data/mcmc/proposal_z.csv", arma::csv_ascii);
  proposal_beta = LoadDouble("data/mcmc/proposal_beta.csv");
  proposal_theta = LoadDouble("data/mcmc/proposal_theta.csv");
  
  accepted_counts_z.zeros(network.N);
  iteration_counts_z.zeros(network.N);
  accepted_counts_beta = 0;
  iteration_counts_beta = 0;
  accepted_counts_theta = 0;
  iteration_counts_theta = 0;
}

void lpm_mcmc::Print()
{
  network.Print();
  std::ostringstream strs;
  strs << "\n\nclass lpm_mcmc";
  strs << "\n\ntotal_n_iter\t=\t" << total_n_iter << "\n";
  strs << "net_n_iter\t=\t" << net_n_iter << "\n";
  strs << "burnin\t=\t" << burnin << "\n";
  strs << "thin\t=\t" << thin << "\n";
  strs << "\n\nproposal sd for the latent positions:\n";
  proposal_z.t().print(strs);
  strs << "\n\nproposal sd for beta\t=\t" << proposal_beta << "\n";
  strs << "\n\nproposal sd for theta\t=\t" << proposal_theta << "\n";
  strs << "\n\nacceptance ratio for the latent positions:\n";
  (accepted_counts_z / (iteration_counts_z+0.00001)).t().print(strs);
  strs << "\n\nacceptance ratio for beta\t=\t" << accepted_counts_beta / (iteration_counts_beta+0.00001) << "\n";
  strs << "\n\nacceptance ratio for theta\t=\t" << accepted_counts_theta / (iteration_counts_theta+0.00001) << "\n";
  if (network.world_rank == 0) std::cout << strs.str() << std::endl;
}

void lpm_mcmc::Summary()
{
  network.Summary();
  std::ostringstream strs;
  strs << "\n\nclass lpm_mcmc";
  strs << "\n\ntotal_n_iter\t=\t" << total_n_iter << "\n";
  strs << "net_n_iter\t=\t" << net_n_iter << "\n";
  strs << "burnin\t=\t" << burnin << "\n";
  strs << "thin\t=\t" << thin << "\n";
  strs << "\n\naverage acceptance ratio for the latent positions\t=\t" << accu((accepted_counts_z / (iteration_counts_z+0.00001)))/network.N << "\n";
  strs << "\n\nacceptance ratio for beta\t=\t" << accepted_counts_beta / (iteration_counts_beta+0.00001) << "\n";
  strs << "\n\nacceptance ratio for theta\t=\t" << accepted_counts_theta / (iteration_counts_theta+0.00001) << "\n";
  if (network.world_rank == 0) std::cout << strs.str() << std::endl;
}

void lpm_mcmc::GibbsSampler()
{
  if (network.world_rank == 0) std::cout << "\nGibbs sampling has started ..." << std::endl;
  unsigned int i, index_inner, index_outer;
  arma::wall_clock timer;
  timer.tic();
  index_outer = 0;
  index_inner = 0;
  while (index_inner < net_n_iter)
  {
    for (i=0; i<network.N; ++i)
    {
      accepted_counts_z.at(i) += network.UpdateZ(i,proposal_z.at(i));
      iteration_counts_z.at(i) ++;
    }
    accepted_counts_beta += network.UpdateBeta(proposal_beta);
    iteration_counts_beta ++;
    accepted_counts_theta += network.UpdateTheta(proposal_theta);
    iteration_counts_theta ++;
    if (index_outer > burnin) if (index_outer % thin == 0)
    {
      z_sample.slice(index_inner) = network.z;
      beta_sample.at(index_inner) = network.beta;
      theta_sample.at(index_inner) = network.theta;
      posterior_values.at(index_inner) = network.posterior_value;
      ++index_inner;
    }
    if (index_outer % 100 == 0) if (network.world_rank == 0) std::cout << "Elapsed Time " << floor(10*timer.toc())/10 << "\t\tEnd of iteration " << index_outer << " out of " << total_n_iter << std::endl;
    ++index_outer;
  }
  if (network.world_rank == 0) std::cout << "... Gibbs sampling has terminated after " << floor(10*timer.toc())/10 << " seconds\n" << std::endl;
  // network.CheckValues();
}

void lpm_mcmc::SaveOutput()
{
  if (network.world_rank == 0) 
  {
    std::cout << "\nlpm_mcmc: Exporting results ..." << std::endl;
    z_sample.save("output/z_sample.bin");
    beta_sample.save("output/beta_sample.csv", arma::csv_ascii);
    theta_sample.save("output/theta_sample.csv", arma::csv_ascii);
    arma::vec acceptance_z;
    acceptance_z = (accepted_counts_z / (iteration_counts_z+0.00001));
    acceptance_z.save("output/acceptance_z.csv",arma::csv_ascii);
    arma::vec acceptance_beta, acceptance_theta;
    acceptance_beta.zeros(1);
    acceptance_theta.zeros(1);
    acceptance_beta.at(0) = accepted_counts_beta / (iteration_counts_beta+0.00001);
    acceptance_theta.at(0) = accepted_counts_theta / (iteration_counts_theta+0.00001);
    acceptance_beta.save("output/acceptance_beta.csv", arma::csv_ascii);
    acceptance_theta.save("output/acceptance_theta.csv", arma::csv_ascii);
    posterior_values.save("output/posterior_values.csv",arma::csv_ascii);
    std::cout << "... results saved\n" << std::endl;
  }
}

