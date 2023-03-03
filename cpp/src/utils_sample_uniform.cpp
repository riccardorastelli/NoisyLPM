#include "utils_sample_uniform.h"

double SampleUniform()
{
  arma::vec u_vec;
  u_vec.randu(1);
  return (u_vec.at(0));
}
