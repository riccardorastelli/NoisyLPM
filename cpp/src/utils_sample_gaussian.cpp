#include "utils_sample_gaussian.h"

double SampleGaussian()
{
  arma::vec u_vec;
  u_vec.randn(1);
  return (u_vec.at(0));
}
