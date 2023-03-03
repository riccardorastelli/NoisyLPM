#include <mpi.h>
#include <armadillo>
#include "core_lpm_noisy.h"
#include "core_lpm_mcmc.h"
#include "utils_mpi.h"

int main (int argc,char **argv)
{
  MPI_Init (&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);

  arma::arma_rng::set_seed(12345);
  
  ////          INITIALISATION OF NOISY LPM
  lpm_noisy network_noisy;
  network_noisy.Summary();
  // network_noisy.Print();
  
  ////          WRITE OUT LIKELIHOOD VALUE FOR STARTING NETWORK
  arma::vec like_vec;
  like_vec.zeros(1);
  like_vec.at(0) = network_noisy.likelihood_value;
  int world_rank_temp;// process number
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank_temp);
  if (world_rank_temp == 0) network_noisy.box_counts.save("output/box_counts.csv",arma::csv_ascii);
  if (world_rank_temp == 0) like_vec.save("output/likelihood_init.csv",arma::csv_ascii);

  ////          MCMC ESTIMATION
  lpm_mcmc mcmc(network_noisy);
  mcmc.GibbsSampler();
  mcmc.SaveOutput();
  mcmc.Summary();
  
  MPI_Finalize();
  return 0;
}
