#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::mat Procrustes(arma::mat x_ref, arma::mat y)
{
    arma::mat y_rot;
    unsigned int N = y.n_rows;
    arma::mat x_centred = (x_ref-repmat(mean(x_ref,0),N,1));
    arma::mat y_centred = (y-repmat(mean(y,0),N,1));
    arma::mat xy = x_centred.t() * y_centred;
    arma::mat U, V;
    arma::vec s;
    arma::svd(U,s,V,xy);
    y_rot = y * V * U.t();
    return(y_rot);
}
