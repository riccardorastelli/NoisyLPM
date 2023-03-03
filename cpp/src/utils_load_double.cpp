#include "utils_load_double.h"

double LoadDouble(std::string filename)
{
  arma::vec vettore;
  vettore.load(filename,arma::csv_ascii);
  double res;
  res = vettore.at(0);
  return (res);
}

double LoadDoubleFromBinary(std::string filename)
{
  arma::vec vettore;
  vettore.load(filename);
  double res;
  res = vettore.at(0);
  return (res);
}
