#include "utils_lpm.h"

double ProposalRatio(double z_start, double z_end, double a, double b, double variance)
{
  double res = 0;
  res += ( CDFGaussian_logfalse((b-z_start)/sqrt(variance)) - CDFGaussian_logfalse((a-z_start)/sqrt(variance)) ) / ( CDFGaussian_logfalse((b-z_end)/sqrt(variance)) - CDFGaussian_logfalse((a-z_end)/sqrt(variance)) );
  return(res);
}

double EuclideanDistanceVec(arma::vec u, arma::vec v)
{
  double res = 0;
  unsigned int D = u.size();
  if (D != v.size()) throw std::runtime_error("Euclidean distance cannot be evaluated for vectors of different sizes.");
  for (unsigned int d = 0; d < D; ++d) res += (u.at(d)-v.at(d)) * (u.at(d)-v.at(d));
  return(sqrt(res));
}

double EuclideanDistance(double u1, double u2, double v1, double v2)
{
  double res = 0;
  res += sqrt(  (u1-v1)*(u1-v1) + (u2-v2)*(u2-v2)  );
  return(res);
}

double EdgeProbability(double beta, double theta, double distance)
{
  double eta = beta - exp(theta) * distance;
  double res = exp(eta) / (1 + exp(eta));
  if (!arma::is_finite(res)) throw std::runtime_error("EdgeProbability function has returned a NaN: maybe parameters values (e.g. initial values) were too large?");
  return(res);
}


