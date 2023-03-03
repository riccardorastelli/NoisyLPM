#ifndef LPM_NOISY_H
#define LPM_NOISY_H

#include <mpi.h>
#include <armadillo>
#include "utils_mpi.h"
#include "utils_load_double.h"
#include "utils_load_unsigned.h"
#include "utils_pdf_gaussian.h"
#include "utils_pdf_trunc_gaussian.h"
#include "utils_sample_uniform.h"
#include "utils_sample_trunc_gaussian.h"
#include "utils_lpm.h"

class lpm_noisy
{
public:
  lpm_noisy();
  void Print();
  void Summary();
  
  // Values inherited from a basic LPM
  unsigned int N;
  unsigned int L;
  arma::mat edgelist;
  arma::field<arma::vec> edge_positions;
  arma::mat adj;
  arma::vec degrees;
  unsigned int D;
  double S;
  arma::mat z;
  double gamma;
  double beta, beta_mu, beta_nu;
  double theta, theta_mu, theta_nu;
  
  // Grid characterisation
  unsigned int M;// M is the number of boxes on one side
  unsigned int Msq;// M*M = total number of boxes
  double b;// length of one side of each box
  arma::mat allocations;
  unsigned int n_boxes_proc;
  arma::mat list_of_boxes_proc;
  arma::mat box_counts;
  arma::cube box_edge_counts;
  
  // Inference
  double prior_value, likelihood_value, posterior_value;
  
  // Functions
  void UpdateValues();
  void EvaluateDegrees();
  void FindEdgePositions();
  void EvaluateAllocations();
  void EvaluateBoxCounts();
  void EvaluateBoxEdgeCounts();
  double DistanceToCentre(double, double, unsigned int, unsigned int);
  void EvaluatePrior();
  void EvaluateLikelihood();
  void EvaluatePosterior();
  void CheckValues();
  
  // Gibbs sampling
  double UpdateBeta(double);
  double UpdateTheta(double);
  double UpdateZ(unsigned int, double);
  void MoveZ(unsigned int, double, double, double, double);
  
  // Parallel computing
  int world_rank;// process number
  int world_size;// number of processes
  unsigned int box_proc_index_start;// denotes the start-index for the subset of the data used by this process
  unsigned int box_proc_index_end;// denotes the final-index for the subset of the data used by this process
  unsigned int box_proc_vec_size;// number of boxes handled by non-root processes
  unsigned int box_root_vec_size;// number of boxes handled by root-process
  
protected:
private:
};

#endif // LPM_NOISY_H
