#ifndef LPM_MCMC_H
#define LPM_MCMC_H

#include <mpi.h>
#include <armadillo>
#include "core_lpm_noisy.h"
#include "utils_load_double.h"
#include "utils_load_unsigned.h"
#include "utils_pdf_gaussian.h"
#include "utils_pdf_trunc_gaussian.h"
#include "utils_sample_uniform.h"
#include "utils_sample_trunc_gaussian.h"
#include "utils_lpm.h"

class lpm_mcmc
{
    public:
        lpm_mcmc(lpm_noisy);
        void Print();
        void Summary();

        lpm_noisy network;
        unsigned int burnin;
        unsigned int thin;
        unsigned int total_n_iter;
        unsigned int net_n_iter;
        arma::cube z_sample;
        arma::vec beta_sample;
        arma::vec theta_sample;
        arma::vec posterior_values;
        arma::vec proposal_z;
        double proposal_beta;
        double proposal_theta;
        arma::vec accepted_counts_z;
        arma::vec iteration_counts_z;
        unsigned int accepted_counts_beta;
        unsigned int iteration_counts_beta;
        unsigned int accepted_counts_theta;
        unsigned int iteration_counts_theta;
        
        void GibbsSampler();
        void SaveOutput();

};

#endif // LPM_MCMC_H
