#include "utils_sample_trunc_gaussian.h"

double SampleTruncGaussian(double mu, double sd, double a, double b)
{
  double res = 0;
  bool stop = false;
  if (sd > (b-a) / sqrt(2*arma::datum::pi))
  {
    double reject_test;
    while (!stop)
    {
      res = a + (b-a)*SampleUniform();
      reject_test = SampleUniform() / sqrt(2*arma::datum::pi) / sd;
      if (  reject_test < exp( -0.5*(res-mu)*(res-mu)/(sd*sd) ) / sqrt(2*arma::datum::pi*sd*sd)  ) stop = true;
    }
  }
  else
  {
    while (!stop)
    {
      res = SampleGaussian()*sd + mu;
      if (res > a) if (res < b) stop = true;
    }
  }
  return (res);
}

