#include "utils_pdf_gaussian.h"

double PDFGaussian(double x)
{
  double res = 0;
  res += - 0.5*std::log(2*arma::datum::pi) - 0.5*x*x;
  return(res);
}

double PDFGaussian_logfalse(double x)
{
  return(exp(PDFGaussian(x)));
}

double CDFGaussian_logfalse(double x)
{
  double res = 0;
  res += erfc(-x/std::sqrt(2))/2;
  return(res);
}

double CDFGaussian(double x)
{
  return(log(CDFGaussian_logfalse(x)));
}

