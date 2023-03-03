#ifndef UTILS_PROPOSAL_H
#define UTILS_PROPOSAL_H

#include <armadillo>
#include "utils_pdf_gaussian.h"
#include "utils_pdf_trunc_gaussian.h"

double ProposalRatio(double, double, double, double, double);
double EuclideanDistanceVec(arma::vec, arma::vec);
double EuclideanDistance(double, double, double, double);
double EdgeProbability(double beta, double theta, double distance);

#endif // UTILS_H_INCLUDED
