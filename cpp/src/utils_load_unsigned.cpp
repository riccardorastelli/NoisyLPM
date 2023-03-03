#include "utils_load_unsigned.h"

unsigned int LoadUnsigned(std::string filename)
{
  arma::vec vettore;
  vettore.load(filename,arma::csv_ascii);
  unsigned int res;
  res = vettore.at(0);
  return (res);
}

unsigned int LoadUnsignedFromBinary(std::string filename)
{
  arma::vec vettore;
  vettore.load(filename);
  unsigned int res;
  res = vettore.at(0);
  return (res);
}
