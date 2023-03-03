#include "core_lpm_noisy.h"

lpm_noisy::lpm_noisy()
{
  unsigned int g, h, index;
  
  // Read all of the input data
  edgelist.load("data/edgelist.csv", arma::csv_ascii);
  beta = LoadDouble("data/init/beta.csv");
  theta = LoadDouble("data/init/theta.csv");
  z.load("data/init/positions.csv", arma::csv_ascii);
  N = z.n_rows;
  D = z.n_cols;
  gamma = LoadDouble("data/hypers/gamma.csv");
  S = LoadDouble("data/hypers/S.csv");
  L = edgelist.n_rows;
  beta_mu = LoadDouble("data/hypers/beta_mu.csv");
  beta_nu = LoadDouble("data/hypers/beta_nu.csv");
  theta_mu = LoadDouble("data/hypers/theta_mu.csv");
  theta_nu = LoadDouble("data/hypers/theta_nu.csv");
  
  // Set up the size of boxes in the grid
  M = LoadUnsigned("data/grid/M.csv");
  Msq = M*M;
  b = 2*S / (double)M;
  
  // Set up parallel computing
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);
  box_proc_vec_size = EvaluateBestSplitSize(Msq,world_size);
  box_root_vec_size = Msq - (world_size-1) * box_proc_vec_size;
  if (world_rank == 0)
  {
    box_proc_index_start = 0;
    box_proc_index_end = box_root_vec_size;
  } else {
    box_proc_index_start = box_root_vec_size + box_proc_vec_size * (world_rank - 1);
    box_proc_index_end = box_proc_index_start + box_proc_vec_size;
  }
  if (box_root_vec_size <= 0 || box_root_vec_size > box_proc_vec_size) throw std::runtime_error("Cannot find a reasonable way to parallelise the computations in the noisy part. Please give a different number of processes.");
  if (box_root_vec_size <= 4 || box_root_vec_size > box_proc_vec_size) throw std::runtime_error("The amount of data assigned to root process in the noisy part is rather small (<=5 nodes) -- too many processes");
  // std::cout << "I am process " << world_rank << " and I am taking care of the boxes identified by the range " << box_proc_index_start << " - " << box_proc_index_end << std::endl;
  
  // Assign boxes at random to each of the processes
  arma::mat list_of_boxes;
  list_of_boxes.zeros(Msq,2);
  index = 0;
  for (g=0; g<M; ++g) for (h=0; h<M; ++h)
  {
    list_of_boxes.at(index,0) = g;
    list_of_boxes.at(index,1) = h;
    index += 1;
  }
  arma::vec list_of_box_indices = arma::linspace<arma::vec>(0,Msq-1,Msq);
  list_of_box_indices = shuffle(list_of_box_indices);
  n_boxes_proc = box_proc_index_end-box_proc_index_start;
  list_of_boxes_proc.zeros(n_boxes_proc,2);
  for (index=0; index<n_boxes_proc; ++index) 
  {
    list_of_boxes_proc.at(index,0) = list_of_boxes.at(list_of_box_indices.at(box_proc_index_start+index),0);
    list_of_boxes_proc.at(index,1) = list_of_boxes.at(list_of_box_indices.at(box_proc_index_start+index),1);
  }
  UpdateValues();
}

void lpm_noisy::Print()
{
  std::ostringstream strs;
  strs << "\n\nclass lpm_noisy";
  strs << "\nN\t=\t" << N;
  strs << "\nD\t=\t" << D;
  strs << "\nL\t=\t" << L;
  strs << "\nM\t=\t" << M;
  strs << "\nb\t=\t" << b;
  strs << "\n\nEdgelist:\n";
  edgelist.print(strs);
  strs << "\n\nDegrees:\n";
  degrees.t().print(strs);
  strs << "\n\nLatent positions:\n";
  z.print(strs);
  strs << "\n\nbeta\t=\t" << beta << "\n";
  strs << "\n\ntheta\t=\t" << theta << "\n";
  strs << "\n\nHyperparameters:" << "\nS\t=\t" << S << "\ngamma\t=\t" << gamma << "\nbeta_mu\t=\t" << beta_mu << "\nbeta_nu\t=\t" << beta_nu << "\ntheta_mu\t=\t" << theta_mu << "\ntheta_nu\t=\t" << theta_nu << "\n";
  strs << "\n\nBox allocations:\n";
  allocations.print(strs);
  strs << "\n\nBox counts:\n";
  box_counts.print(strs);
  strs << "\n\nLog-prior value\t=\t" << prior_value << "\n";
  strs << "\n\nLog-likelihood value\t=\t" << likelihood_value << "\n";
  strs << "\n\nLog-posterior value\t=\t" << posterior_value << "\n";
  if (world_rank == 0) std::cout << strs.str() << std::endl;
}

void lpm_noisy::Summary()
{
  std::ostringstream strs;
  strs << "\n\nclass lpm_noisy";
  strs << "\n\nLog-prior value\t=\t" << prior_value << "\n";
  strs << "\n\nLog-likelihood value\t=\t" << likelihood_value << "\n";
  strs << "\n\nLog-posterior value\t=\t" << posterior_value << "\n";
  if (world_rank == 0) std::cout << strs.str() << std::endl;
}

void lpm_noisy::UpdateValues()
{
  EvaluateDegrees();
  FindEdgePositions();
  EvaluateAllocations();
  EvaluateBoxCounts();
  EvaluateBoxEdgeCounts();
  EvaluatePrior();
  EvaluateLikelihood();
  EvaluatePosterior();
}

void lpm_noisy::EvaluateDegrees()
{
  unsigned int l;
  degrees.set_size(N);
  degrees.fill(0);
  for (l=0; l<L; ++l)
  {
    degrees.at(edgelist.at(l,0)) ++;
    degrees.at(edgelist.at(l,1)) ++;
  }
}

void lpm_noisy::FindEdgePositions()
{
  unsigned int l, i, j;
  edge_positions.set_size(N);
  for (i=0; i<N; ++i) edge_positions.at(i).set_size(degrees.at(i));
  arma::vec degree_counter;
  degree_counter.zeros(N);
  for (l=0; l<L; ++l)
  {
    i = edgelist.at(l,0);
    j = edgelist.at(l,1);
    edge_positions.at(i).at(degree_counter.at(i)) = l;
    degree_counter.at(i) ++;
    edge_positions.at(j).at(degree_counter.at(j)) = l;
    degree_counter.at(j) ++;
  }
}

void lpm_noisy::EvaluateAllocations()
{
  unsigned int i, d;
  allocations.set_size(N,2);
  allocations.fill(M);
  for (i=0; i<N; ++i) for (d=0; d<D; ++d)
  {
    if (std::abs(z.at(i,d)) > S) throw std::runtime_error("positions outside the reticulate");
    else allocations.at(i,d) = std::floor((z.at(i,d) + S) / b);
  }
}

void lpm_noisy::EvaluateBoxCounts()
{
  box_counts.zeros(M,M);
  for (unsigned int i = 0; i < N; ++i) box_counts.at(allocations.at(i,0),allocations.at(i,1)) ++;
}

void lpm_noisy::EvaluateBoxEdgeCounts()
{
  unsigned int l, i, j;
  box_edge_counts.zeros(M,M,N);
  for (l=0; l<L; ++l)
  {
    i = edgelist.at(l,0);
    j = edgelist.at(l,1);
    box_edge_counts.at(allocations.at(j,0),allocations.at(j,1),i) ++;
    box_edge_counts.at(allocations.at(i,0),allocations.at(i,1),j) ++;
  }
}

double lpm_noisy::DistanceToCentre(double position_0, double position_1, unsigned int g, unsigned int h)
{
  double res = 0;
  double centre_0, centre_1;
  centre_0 = -S + b*g + 0.5*b;
  centre_1 = -S + b*h + 0.5*b;
  res += sqrt(  (position_0 - centre_0) * (position_0 - centre_0) + (position_1 - centre_1) * (position_1 - centre_1)  );
  return(res);
}

void lpm_noisy::EvaluatePrior()
{
  unsigned int i, d;
  prior_value = 0;
  prior_value += - 0.5*log(beta_nu) - 0.5*log(2*arma::datum::pi) - 0.5*(beta-beta_mu)*(beta-beta_mu)/beta_nu;
  prior_value += - 0.5*log(theta_nu) - 0.5*log(2*arma::datum::pi) - 0.5*(theta-theta_mu)*(theta-theta_mu)/theta_nu;
  for (i=0; i<N; ++i) for (d=0; d<D; ++d) prior_value += PDFTruncGaussian(z.at(i,d),0,gamma,-S,S);
}

void lpm_noisy::EvaluateLikelihood()
{
  likelihood_value = 0;
  double likelihood_value_proc = 0;
  unsigned int i, k, g, h;
  double d_igh, p_igh, n_nodes_in_box;
  for (i=0; i<N; ++i) for (k=0; k<n_boxes_proc; ++k)
  {
    g = list_of_boxes_proc.at(k,0);
    h = list_of_boxes_proc.at(k,1);
    n_nodes_in_box = box_counts.at(g,h);
    if (n_nodes_in_box > 0)
    {
      d_igh = DistanceToCentre(z.at(i,0),z.at(i,1),g,h);
      p_igh = EdgeProbability(beta,theta,d_igh);
      if (allocations.at(i,0) == g) if (allocations.at(i,1) == h) n_nodes_in_box --;
      if (box_edge_counts.at(g,h,i) > 0) likelihood_value_proc += box_edge_counts.at(g,h,i) * log(p_igh);
      if (n_nodes_in_box-box_edge_counts.at(g,h,i) > 0) likelihood_value_proc += (n_nodes_in_box-box_edge_counts.at(g,h,i)) * log(1-p_igh);
      if (!arma::is_finite(likelihood_value_proc)) throw std::runtime_error("Likelihood is NaN: maybe parameter values were too large?"); 
    }
  }
  MPI_Allreduce(&likelihood_value_proc, &likelihood_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  likelihood_value = likelihood_value / 2;// since network is undirected, likelihood contributions are counted twice
}

void lpm_noisy::EvaluatePosterior()
{
  posterior_value = prior_value + likelihood_value;
}

void lpm_noisy::CheckValues()
{
  double prior_check = prior_value;
  double likelihood_check = likelihood_value;
  UpdateValues();
  if (world_rank == 0) std::cout << "DEBUG: error on prior = " << std::abs(prior_value - prior_check)  << "\t\terror on likelihood = " << std::abs(likelihood_value - likelihood_check) << std::endl;
}

double lpm_noisy::UpdateZ(unsigned int i, double sd)
{
  double res = 0;
  double proposal_ratio, prior_delta, likelihood_delta, likelihood_delta_proc, n_nodes_in_box;
  unsigned int d, k, g, h;
  double d_igh, d_kgh, p_igh, p_kgh;
  arma::vec z_new;
  z_new.zeros(D);
  for (d=0; d<D; ++d) z_new.at(d) = SampleTruncGaussian(z.at(i,d),sd,-S,S);
  
  proposal_ratio = 0;
  for (d=0; d<D; ++d) proposal_ratio += log(ProposalRatio(z.at(i,d),z_new.at(d),-S,S,sd*sd));
  
  prior_delta = 0;
  for (d=0; d<D; ++d) prior_delta += PDFGaussian(z_new.at(d)/gamma) - PDFGaussian(z.at(i,d)/gamma);
  
  likelihood_delta = 0;
  likelihood_delta_proc = 0;
  for (k=0; k<n_boxes_proc; ++k)
  {
    g = list_of_boxes_proc.at(k,0);
    h = list_of_boxes_proc.at(k,1);
    n_nodes_in_box = box_counts.at(g,h);
    if (n_nodes_in_box > 0)
    {
      d_igh = DistanceToCentre(z.at(i,0),z.at(i,1),g,h);
      d_kgh = DistanceToCentre(z_new(0),z_new(1),g,h);
      p_igh = EdgeProbability(beta,theta,d_igh);
      p_kgh = EdgeProbability(beta,theta,d_kgh);
      if (allocations.at(i,0) == g) if (allocations.at(i,1) == h) n_nodes_in_box --;
      if (box_edge_counts.at(g,h,i) > 0) likelihood_delta_proc += box_edge_counts.at(g,h,i) * (log(p_kgh)-log(p_igh));
      if (n_nodes_in_box-box_edge_counts.at(g,h,i) > 0) likelihood_delta_proc += (n_nodes_in_box-box_edge_counts.at(g,h,i)) * (log(1-p_kgh)-log(1-p_igh));
    }
  }
  MPI_Allreduce(&likelihood_delta_proc, &likelihood_delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  if (log(SampleUniform()) < proposal_ratio + prior_delta + likelihood_delta)
  {
    res += 1;
    MoveZ(i, z_new.at(0), z_new.at(1), prior_delta, likelihood_delta);
  }
  // CheckValues();
  return(res);
}

void lpm_noisy::MoveZ(unsigned int i, double z_new_0, double z_new_1, double prior_delta, double likelihood_delta)
{
  unsigned int g_old, h_old, g_new, h_new, j, d, l;
  g_old = allocations.at(i,0);
  h_old = allocations.at(i,1);
  z.at(i,0) = z_new_0;
  z.at(i,1) = z_new_1;
  prior_value += prior_delta;
  likelihood_value += likelihood_delta;
  EvaluatePosterior();
  for (d=0; d<D; ++d)
  {
    if (std::abs(z.at(i,d)) > S) throw std::runtime_error("attempting to move a node outside of the reticulate");
    else allocations.at(i,d) = std::floor((z.at(i,d) + S) / b);
  }
  g_new = allocations.at(i,0);
  h_new = allocations.at(i,1);
  if (g_old != g_new || h_old != h_new)
  {
    box_counts.at(g_old,h_old) -= 1;
    box_counts.at(g_new,h_new) += 1;
    for (l=0; l<edge_positions.at(i).size(); ++l)
    {
      j = edgelist.at(edge_positions.at(i).at(l),0);
      if (j == i) j = edgelist.at(edge_positions.at(i).at(l),1);
      box_edge_counts.at(g_old,h_old,j) --;
      box_edge_counts.at(g_new,h_new,j) ++;
    }
  }
}

double lpm_noisy::UpdateBeta(double sd)
{
  double res = 0;
  double beta_new, prior_delta, likelihood_delta, likelihood_delta_proc;
  unsigned int i, k, g, h;
  double d_igh, p_igh, p_igh_new, n_nodes_in_box;
  beta_new = SampleGaussian()*sd + beta;
  
  prior_delta = 0;
  prior_delta += 0.5 * (beta*beta - 2*beta*beta_mu - beta_new*beta_new + 2*beta_new*beta_mu) / beta_nu;
  
  likelihood_delta = 0;
  likelihood_delta_proc = 0;
  for (i=0; i<N; ++i) for (k=0; k<n_boxes_proc; ++k)
  {
    g = list_of_boxes_proc.at(k,0);
    h = list_of_boxes_proc.at(k,1);
    n_nodes_in_box = box_counts.at(g,h);
    if (n_nodes_in_box > 0)
    {
      d_igh = DistanceToCentre(z.at(i,0),z.at(i,1),g,h);
      p_igh = EdgeProbability(beta,theta,d_igh);
      p_igh_new = EdgeProbability(beta_new,theta,d_igh);
      if (allocations.at(i,0) == g) if (allocations.at(i,1) == h) n_nodes_in_box --;
      if (box_edge_counts.at(g,h,i) > 0) likelihood_delta_proc += box_edge_counts.at(g,h,i) * (log(p_igh_new)-log(p_igh));
      if (n_nodes_in_box-box_edge_counts.at(g,h,i) > 0) likelihood_delta_proc += (n_nodes_in_box-box_edge_counts.at(g,h,i)) * (log(1-p_igh_new)-log(1-p_igh));
    }
  }
  MPI_Allreduce(&likelihood_delta_proc, &likelihood_delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  likelihood_delta = likelihood_delta / 2;// since network is undirected, likelihood contributions are counted twice
  
  if (log(SampleUniform()) < prior_delta + likelihood_delta)
  {
    res += 1;
    beta = beta_new;
    prior_value += prior_delta;
    likelihood_value += likelihood_delta;
    EvaluatePosterior();
  }
  // CheckValues();
  return(res);
}

double lpm_noisy::UpdateTheta(double sd)
{
  double res = 0;
  double theta_new, prior_delta, likelihood_delta, likelihood_delta_proc;
  unsigned int i, k, g, h;
  double d_igh, p_igh, p_igh_new, n_nodes_in_box;
  theta_new = SampleGaussian()*sd + theta;
  
  prior_delta = 0;
  prior_delta += 0.5 * (theta*theta - 2*theta*theta_mu - theta_new*theta_new + 2*theta_new*theta_mu) / theta_nu;
  
  likelihood_delta = 0;
  likelihood_delta_proc = 0;
  for (i=0; i<N; ++i) for (k=0; k<n_boxes_proc; ++k)
  {
    g = list_of_boxes_proc.at(k,0);
    h = list_of_boxes_proc.at(k,1);
    n_nodes_in_box = box_counts.at(g,h);
    if (n_nodes_in_box > 0)
    {
      d_igh = DistanceToCentre(z.at(i,0),z.at(i,1),g,h);
      p_igh = EdgeProbability(beta,theta,d_igh);
      p_igh_new = EdgeProbability(beta,theta_new,d_igh);
      if (allocations.at(i,0) == g) if (allocations.at(i,1) == h) n_nodes_in_box --;
      if (box_edge_counts.at(g,h,i) > 0) likelihood_delta_proc += box_edge_counts.at(g,h,i) * (log(p_igh_new)-log(p_igh));
      if (n_nodes_in_box-box_edge_counts.at(g,h,i) > 0) likelihood_delta_proc += (n_nodes_in_box-box_edge_counts.at(g,h,i)) * (log(1-p_igh_new)-log(1-p_igh));
    }
  }
  MPI_Allreduce(&likelihood_delta_proc, &likelihood_delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  likelihood_delta = likelihood_delta / 2;// since network is undirected, likelihood contributions are counted twice
  
  if (log(SampleUniform()) < prior_delta + likelihood_delta)
  {
    res += 1;
    theta = theta_new;
    prior_value += prior_delta;
    likelihood_value += likelihood_delta;
    EvaluatePosterior();
  }
  // CheckValues();
  return(res);
}
