#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::cube LoadBinaryCube(std::string filename)
{
  arma::cube res;
  res.load(filename);
  return(res);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::mat LoadBinaryMatrix(std::string filename)
{
  arma::mat res;
  res.load(filename);
  return(res);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec LoadBinaryVector(std::string filename)
{
  arma::vec res;
  res.load(filename);
  return(res);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

unsigned int LoadBinaryUnsigned(std::string filename)
{
  arma::vec v;
  v.load(filename);
  unsigned int res = v.at(0);
  return(res);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double LoadBinaryDouble(std::string filename)
{
  arma::vec v;
  v.load(filename);
  double res = v.at(0);
  return(res);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

void SaveBinaryCube(arma::cube data, std::string filename)
{
  data.save(filename);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

void SaveBinaryMatrix(arma::mat data, std::string filename)
{
  data.save(filename);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

void SaveBinaryVector(arma::vec data, std::string filename)
{
  data.save(filename);
}

