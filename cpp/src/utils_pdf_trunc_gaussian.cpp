#include "utils_pdf_trunc_gaussian.h"

double PDFTruncGaussian_logfalse(double x, double mean, double variance, double a, double b)
{
  double res = 0;
  double x_std = (x-mean)/sqrt(variance);
  res += PDFGaussian_logfalse(x_std) / sqrt(variance) / ( CDFGaussian_logfalse((b-mean)/sqrt(variance)) - CDFGaussian_logfalse((a-mean)/sqrt(variance)));
  return(res);
}

double PDFTruncGaussian(double x, double mean, double variance, double a, double b)
{
  double res = 0;
  res += log(PDFTruncGaussian_logfalse(x, mean, variance, a, b));
  return(res);
}

