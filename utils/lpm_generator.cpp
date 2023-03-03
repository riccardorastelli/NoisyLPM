#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List lpm_generator_cpp(double beta, double theta, arma::mat positions)
{
	unsigned int i, j, d, l, degree_counter;
	unsigned int N, D;
	N = positions.n_rows;
	D = positions.n_cols;
	arma::field<arma::vec> edges_by_node;
	edges_by_node.set_size(N);
	arma::vec neighbours;
	arma::vec out_degrees;
	out_degrees.zeros(N);
	double d_ij, eta_ij, p_ij;
	arma::vec u;
	for (i=0; i<N-1; ++i)
	{
		neighbours.zeros(N);
		for (j=i+1; j<N; ++j)
		{
			d_ij = 0;
			for (d=0; d<D; ++d) d_ij += (positions.at(i,d) - positions.at(j,d))*(positions.at(i,d) - positions.at(j,d));
			d_ij = sqrt(d_ij);
			eta_ij = beta - exp(theta) * d_ij;
			p_ij = exp(eta_ij) / (1 + exp(eta_ij));
			u.randu(1);
			if (u.at(0) < p_ij) neighbours.at(j) = 1;
		}
		out_degrees.at(i) = sum(neighbours);
		edges_by_node.at(i).set_size(out_degrees.at(i));
		degree_counter = 0;
		for (j=i+1; j<N; ++j) if (neighbours.at(j) > 0) 
		{
			edges_by_node.at(i).at(degree_counter) = j;
			degree_counter++;
		}
	}
	unsigned int L = sum(out_degrees);
	arma::mat edgelist;
	edgelist.zeros(L,2);
	l = 0;
	for (i=0; i<N; ++i) for (degree_counter=0; degree_counter<out_degrees.at(i); ++degree_counter)
	{
		edgelist.at(l,0) = i;
		edgelist.at(l,1) = edges_by_node.at(i).at(degree_counter);
		++l;
	}
	return(Rcpp::List::create(Rcpp::Named("N")=N,
				Rcpp::Named("D")=D,
				Rcpp::Named("L")=L,
				Rcpp::Named("edgelist")=edgelist));
}
