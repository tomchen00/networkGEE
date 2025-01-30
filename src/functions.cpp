// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <iostream>
#include <RcppArmadillo.h>
#include <math.h>
#include <vector>
#include <Rcpp.h>
#include <algorithm>
#include <numeric>
#include <RcppArmadilloExtensions/sample.h>
using namespace std;
using namespace Rcpp;

//Sugar function for seq()
//[[Rcpp::export]]
arma::uvec Seq(int first, int last) {
  arma::uvec y(abs(last - first) + 1);
  if (first < last)
    std::iota(y.begin(), y.end(), first);
  else {
    iota(y.begin(), y.end(), last);
    std::reverse(y.begin(), y.end());
  }
  
  return y;
}

//Sugar function for pmin()
//[[Rcpp::export]]
arma::colvec mypmin(arma::colvec vec1, arma::colvec vec2) {
  unsigned n = vec1.n_rows;
  if(n != vec2.n_rows) return 0;
  else {
    arma::colvec out(n);
    for(unsigned i = 0; i < n; i++) {
      out[i] = min(vec1[i], vec2[i]);
    }
    return out;
  }
}

//Sugar function for rep()
//[[Rcpp::export]]
arma::colvec myrep(arma::colvec x, arma::colvec y) {
  int n = y.size();
  arma::colvec myvector(sum(y));
  int ind = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < y[i]; ++j) {
      myvector[ind] = x[i];
      ind = ind + 1;
    }
  }
  return myvector;
}

//Sugar function for rep() with uvec type
//[[Rcpp::export]]
arma::uvec myrep_uvec(arma::uvec x, arma::uvec y) {
  int n = y.size();
  arma::uvec myvector(sum(y));
  int ind = 0;
  for (int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < y[i]; ++j) {
      myvector[ind] = x[i];
      ind = ind + 1;
    }
  }
  return myvector;
}

//Sugar function for colSums()
//[[Rcpp::export]]
arma::rowvec colSums(const arma::mat & X){
  int nCols = X.n_cols;
  arma::rowvec out(nCols);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(X.col(i));
  }
  return(out);
}


//k-way combinations of {1, ..., n} indices
// [[Rcpp::export]]
arma::umat mycombnCpp(double n, double k) {
  
  double n_subsets = Rf_choose(n, k);
  arma::umat out = arma::zeros<arma::umat>(k, n_subsets);
  arma::uvec a = arma::linspace<arma::uvec>(1, k, k);
  out.col(0) = a;
  int m = 0;
  int h = k;
  arma::uvec j;
  
  for(long long i = 1; i < n_subsets; i++){
    if(m < (n - h)){
      h = 1;
      m = a(k - 1);
      j = arma::linspace<arma::uvec>(1, 1, 1);
    }
    else{
      m = a(k - h - 1);
      ++h;
      j = arma::linspace<arma::uvec>(1, h, h);
    }
    a.elem(k - h - 1 + j) = m + j;
    out.col(i) = a;
  }
  return(out);
}



//k-way product of {1, ..., n} indices
// [[Rcpp::export]]
arma::colvec pairwiseprod(arma::colvec M) {
  
  int p = M.n_rows;
  int counter = 0;
  arma::colvec output(p*(p-1)/2);
  
  for (int i = 0; i < p-1; i++){
    for (int j = i+1; j < p; j++){
      output(counter) = M(i)*M(j);
      counter += 1;
    }
  }
  return(output);
  
}

// [[Rcpp::export]]
arma::mat pairwiseprod2(arma::mat M, arma::colvec N, int dir) {
  
  int p = M.n_rows;
  int counter = 0;
  arma::mat output(p*(p-1)/2, M.n_cols);
  
  if (dir == 1){
    for (int i = 0; i < p-1; i++){
      for (int j = i+1; j < p; j++){
        output.row(counter) = M.row(i)*N[j];
        counter += 1;
      }
    }
  } else if (dir == 2) {
    for (int i = 0; i < p-1; i++){
      for (int j = i+1; j < p; j++){
        output.row(counter) = M.row(j)*N[i];
        counter += 1;
      }
    }
  }
  
  return(output);
  
}


//Set difference
// [[Rcpp::export]]
arma::uvec arma_setdiff(arma::uvec& x, arma::uvec& y) {
  
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  
  return arma::conv_to<arma::uvec>::from(out);
}

//Compute indices of off-block diagonal blocks corresponding to same first-level, but different second-level
//Same first & second level correlation is then the complement of this index vector
//Zero indexing
//[[Rcpp::export]]
arma::uvec index_fun2(arma::uvec J) {
  int m = J.n_elem;
  arma::uvec J_rev = arma::reverse(J);
  arma::uvec J_seq(sum(J.subvec(0, m-2)));
  int counter = 0;
  for(int i = 0; i < m-1; i++) {
    J_seq.subvec(counter, counter + J[i] - 1) = Seq(J[i] - 1, 0);
    counter += J[i];
  }
  
  arma::uvec J_rev_cumsum = myrep_uvec(arma::reverse(arma::cumsum(J_rev.subvec(0, m-2))), J.subvec(0, m-2));
  
  m = J_rev_cumsum.n_elem;
  arma::uvec idxidx = cumsum(J_rev_cumsum);
  arma::uvec idx_range(sum(J_rev_cumsum), arma::fill::zeros);
  int a = J_seq[0]; int b = a + J_rev_cumsum[0];
  idx_range.subvec(0, idxidx[0]-1) = Seq(a, b - 1);
  
  for(int i = 1; i < m; i++) {
    a = b + J_seq[i];
    b = a + J_rev_cumsum[i];
    idx_range.subvec(idxidx[i-1], idxidx[i]-1) = Seq(a, b - 1);
  }
  
  return(idx_range);
}

//Compute indices of same individual but in different time periods
//Zero indexing
//[[Rcpp::export]]
arma::uvec index_fun1(arma::uvec J) {
  int m = J[0];
  int n = J.n_elem;
  arma::uvec idx_range(m*n*(n-1)/2, arma::fill::zeros);
  int counter = 0;
  
  arma::uvec temp_vec = m*Seq(1, n)-1;
  for(int j = n-1; j > 0; j--) {
    temp_vec.shed_row(j);
    for(int i = 2*m-1; i > m-1; i--) {
      idx_range.subvec(counter, counter + j - 1) = temp_vec;
      temp_vec = temp_vec + (i + (j-1)*m);
      counter += j;
    }
  }
  return(idx_range);
}


// [[Rcpp::export]]
NumericVector combine(const List& list)
{
  std::size_t n = list.size();
  
  // Figure out the length of the output vector
  std::size_t total_length = 0;
  for (std::size_t i = 0; i < n; ++i)
    total_length += Rf_length(list[i]);
  
  // Allocate the vector
  NumericVector output = no_init(total_length);
  
  // Loop and fill
  std::size_t index = 0;
  for (std::size_t i = 0; i < n; ++i)
  {
    NumericVector el = list[i];
    std::copy(el.begin(), el.end(), output.begin() + index);
    
    // Update the index
    index += el.size();
  }
  
  return output;
  
}

// [[Rcpp::export]]
List cluster_characteristics(arma::mat clusterid) {
  
  int n = clusterid.n_rows;                         //Number of individuals
  int q = clusterid.n_cols;                         //Number of levels (in hierarchy)
  arma::uvec n_seq = Seq(1,n);                      //Label for each individual/observation
  arma::colvec I_seq = unique(clusterid.col(0));    //Sequence of 1st-order cluster IDs
  //Insert higher-order cluster IDs here in for future implementations (K, L, ...)
  
  int I = I_seq.size();                             //Number of 1st-order clusters
  List I_idx(I);                                    //Initialize list of length I to keep track of higher-order cluster IDs
  List return_list(4);
  
  arma::colvec I_diff = diff(clusterid.col(0)); I_diff.insert_rows(0,1); I_diff[0] = 1; I_diff.insert_rows(n,1); I_diff[n] = 1;
  arma::uvec J_bool = find(I_diff);                 //Indices for change in 1st-order cluster id
  arma::uvec J = diff(J_bool);                      //Number in 1st-order cluster
  
  if (q == 1){                                      //Only one level of clustering
    for(int i = 0; i < I; i++) {
      I_idx[i] = Seq(J_bool[i]+1,J_bool[i+1]);      //Set list element to those corresponding indices
    }
  } else if (q == 2) {                              //Two levels of clustering
    arma::uvec J_col_idx = {1};
    List K(I);
    for(int i = 0; i < I; i++) {
      arma::uvec a = Seq(J_bool[i]+1,J_bool[i+1]);                 //Indices for current 1st-order cluster
      arma::colvec J_subset = clusterid.submat(a-1, J_col_idx);    //2nd-order cluster IDs for current 1st-order cluster
      arma::colvec J_seq = unique(J_subset);                       //Sequence of 2nd-order cluster IDs
      int J_num = J_seq.size();                                    //Number of 2nd-order clusters
      arma::colvec J_diff = diff(J_subset); J_diff.insert_rows(0,1); J_diff[0] = 1; J_diff.insert_rows(J[i],1); J_diff[J[i]] = 1;
      arma::uvec K_bool = find(J_diff) + J_bool[i];                //Indices for change in 2st-order cluster id
      
      List J_idx(J_num);
      for(int j = 0; j < J_num; j++) {
        J_idx[j] = Seq(K_bool[j]+1,K_bool[j+1]);
      }
      I_idx[i] = J_idx;                                            //List of lists, with each nesting representing the hierarchichies
      //J[i] = J_num;
      //K[i] = diff(K_bool);                                         //Number in 2st-order cluster
    }
  }
  return I_idx;
}

//Sample clusters, clusters with clusters, and so on
//Default is 10 at 1st stage, 5 at 2nd cluster stage, 2 at 3rd stage (subject to future change)
// [[Rcpp::export]]
List hierarchical_sampling(List I_idx,
                           arma::colvec batch,
                           bool replace,
                           bool tscs) {
  int b = batch.n_rows;                //b = 3 implies 3-level, for example
  List I_idx_sub;                      //To hold nested list of indices for subsample
  List return_list(4);                 //Return: subindices, inital # of 1st-order clusters
  //   initial # of 2nd-order clusters,
  //   initial # of 3rd-order clusters
  int I_S = 0;
  
  
  if(batch[0] < I_idx.size()){
    I_S = batch[0];
    I_idx_sub = clone(RcppArmadillo::sample(I_idx, batch[0], replace, NumericVector::create()));
  } else {
    I_S = I_idx.size();
    I_idx_sub = clone(I_idx);
  }
  
  int I_full = I_idx.size();
  arma::colvec J_full(I_S);
  
  if (!tscs) {
    if (b == 2){                                       //Sampling at 2nd level
      for(int i = 0; i < I_idx_sub.size(); i++) {
        arma::colvec J_idx = I_idx_sub[i];
        J_full[i] = J_idx.n_rows;
        if(batch[1] < J_idx.n_rows){
          I_idx_sub[i] = RcppArmadillo::sample(J_idx, batch[1], replace, NumericVector::create());
        }
      }
      return_list = List::create(Rcpp::Named("I_idx") = I_idx_sub,
                                 Rcpp::Named("I") = I_full,
                                 Rcpp::Named("J") = J_full,
                                 Rcpp::Named("K") = 0,
                                 Rcpp::Named("I_vec") = as<arma::uvec>(combine(I_idx_sub))
      );
    } else if (b == 3) {                              //Sampling at 2nd and 3rd level
      List I_vec(I_S);
      List K_full(I_S);
      for(int i = 0; i < I_S; i++) {
        List J_idx_sub = I_idx_sub[i];
        J_full[i] = J_idx_sub.size();
        if(batch[1] < J_idx_sub.size()){
          I_idx_sub[i] = RcppArmadillo::sample(J_idx_sub, batch[1], replace, NumericVector::create());
        }
        
        //Additional loop for 3rd level indices
        J_idx_sub = I_idx_sub[i];
        arma::uvec K_fullsubset(J_idx_sub.size());
        for(int j = 0; j < J_idx_sub.size(); j++) {
          arma::colvec K_idx = J_idx_sub[j];
          K_fullsubset[j] = K_idx.n_rows;
          if(batch[2] < K_idx.n_rows){
            J_idx_sub[j] = RcppArmadillo::sample(K_idx, batch[2], replace, NumericVector::create());
          }
        }
        K_full[i] = K_fullsubset;
        I_idx_sub[i] = J_idx_sub;
        I_vec[i] = as<arma::uvec>(combine(J_idx_sub));
      }
      return_list = List::create(Rcpp::Named("I_idx") = I_idx_sub,
                                 Rcpp::Named("I") = I_full,          //Total original number of 1st order clusters
                                 Rcpp::Named("J") = J_full,          //For subset of 1st order clusters, the original number of 2nd-order clusters
                                 Rcpp::Named("K") = K_full,          //For subset of 1st and 2nd order clusters, the original number of 3rd-order clusters
                                 Rcpp::Named("I_vec") = as<arma::uvec>(combine(I_vec))
      );
    }
  } else {
    List I_vec(I_S);
    List K_full(I_S);
    for(int i = 0; i < I_S; i++) {
      List J_idx_sub = I_idx_sub[i];                  //Time periods associated with cluster i
      J_full[i] = J_idx_sub.size();                   //Number of time periods
      arma::uvec K_fullsubset(J_idx_sub.size());      //Initialize list of individuals to be subsampled
      
      //Subsample individuals (which will remain the same individuals over the different time periods)
      arma::colvec K_idx_0 = J_idx_sub[0];
      arma::uvec ind_subset;
      
      if (batch[1] < K_idx_0.n_rows) {
        ind_subset = RcppArmadillo::sample(Seq(0, K_idx_0.n_rows-1), batch[1], replace, NumericVector::create());
      } else {
        ind_subset = Seq(0, K_idx_0.n_rows-1);
      }
      
      
      for(int j = 0; j < J_idx_sub.size(); j++) {
        arma::colvec K_idx = J_idx_sub[j];
        K_fullsubset[j] = K_idx.n_rows;
        if(batch[2] < K_idx.n_rows){
          arma::colvec temp_vec = K_idx.rows(ind_subset);
          J_idx_sub[j] = temp_vec;
        }
      }
      
      K_full[i] = K_fullsubset;
      I_idx_sub[i] = J_idx_sub;
      I_vec[i] = as<arma::uvec>(combine(J_idx_sub));
    }
    return_list = List::create(Rcpp::Named("I_idx") = I_idx_sub,
                               Rcpp::Named("I") = I_full,          //Total original number of 1st order clusters
                               Rcpp::Named("J") = J_full,          //For subset of 1st order clusters, the original number of 2nd-order clusters
                               Rcpp::Named("K") = K_full,          //For subset of 1st and 2nd order clusters, the original number of 3rd-order clusters
                               Rcpp::Named("I_vec") = as<arma::uvec>(combine(I_vec))
    );
  }
  return return_list;
}


//[[Rcpp::export]]
Rcpp::List rcpp_lengths_recursive( const Rcpp::List& lst ) {
  std::size_t n = lst.size();
  Rcpp::List res( n );
  std::size_t i;
  
  for( i = 0; i < n; i++ ) {
    switch( TYPEOF( lst[i] ) ) {
    case VECSXP: {
      // might need an `Rf_inherits( lst[i], "data.frame" )` check,
      // depending on your use-case
      res[ i ] = rcpp_lengths_recursive( lst[i] );
      break;
    }
    default: {
      int n_elements = Rf_length( lst[i] );
      res[i] = n_elements;
    }
    }
  }
  return res;
}

//[[Rcpp::export]]
Rcpp::IntegerVector rcpp_lengths( const Rcpp::List& lst ) {
  std::size_t n = lst.size();
  Rcpp::IntegerVector res( n );
  std::size_t i;
  
  for( i = 0; i < n; i++ ) {
    res[i] = Rf_length( lst[i] );
  }
  return res;
}

List binomial_hier_solver(arma::colvec beta,
                          arma::colvec rho,
                          arma::colvec outcome,
                          arma::mat design_mat,
                          arma::mat Info1,
                          List I_idx) {
  
  arma::colvec mu = 1/(1+exp(-design_mat*beta));    //Mean vector
  double rho0 = 1 - rho[0];                       //Remaining ICC; appears in many computations
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = mu%(1-mu);                   //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  arma::mat std_design_mat = design_mat.each_col() % U_sqrt;
  arma::colvec etilde_1(outcome.n_rows, arma::fill::zeros);
  arma::colvec w_portion_resid = (1-2*mu)/U_sqrt;
  
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Dimension of linear predictor
  int q = rho.n_rows;                               //Dimension of correlation terms
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  
  if (q == 2){
    
    for(int i = 0; i < I; i++) {
      List I_idx_sub = I_idx[i];                                                   //Sublist for current 1st-order cluster
      arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Indices for current 1st-order cluster, zero-indexed
      arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Vector of K_{ij} (misnomer)
      double Ji = sum(J_sub);                                                      //K_{i\cdot} (misnomer)
      
      //Precomputed values which appear many times in computation; see paper
      //U = diagonal matrix of Var(Y|X) = phi*vf, where vf = is variance function matrix
      //D^T V^{-1} E = X^T vf phi^{-1/2}*vf^{-1/2} R^{-1} phi^{-1/2}*vf^{-1/2} E
      //             = (X^T U^{1/2})/phi R^{-1} (U^{-1/2} E)
      arma::colvec v = myrep(1/rho0 - (rho[0] - rho[1])/(rho0*(rho0/J_sub + rho[0] - rho[1])), J_sub);
      
      double c = 1/(1/rho[1] + sum(v));
      arma::mat Xtilde = std_design_mat.rows(I_sub_uvec);
      arma::colvec Etilde = std_resid.rows(I_sub_uvec);
      
      //One portion of block-matrix multiplication involves matrix of 1's
      //Vectorize entire operation using special structure of matrix of 1's
      
      double a = 0;
      double a0 = 0;
      
      
      for (unsigned int j = 0; j < J_sub.n_rows; j++) {
        arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1;               //J_idx subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
        arma::rowvec temp_vec_X = colSums(Xtilde.rows(J_idx));
        arma::colvec temp_vec_E = Etilde.rows(J_idx);
        
        G1.slice(i) += -temp_vec_X.t()*sum(temp_vec_E)*(rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*J_sub[j]));
        H1.slice(i) += -temp_vec_X.t()*temp_vec_X*(rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*J_sub[j]));
        
        //Now for Hessians/gradients of association parameters
        double temp_double_E = sum(temp_vec_E);
        a = sum(square(temp_vec_E) - 1);
        a0 = (pow(temp_double_E, 2) - sum(temp_vec_E % temp_vec_E))/2;
        G2(0,0,i) += a0 - J_sub[j]*(J_sub[j]-1)/2*rho[0];
        G2(1,0,i) -= (a + J_sub[j])/2 + a0;
      }
      
      //Add on remaining terms of summands (don't require 'for loop')
      G1.slice(i) += Xtilde.t()*Etilde/rho0 - c*(Xtilde.t()*v)*(v.t()*Etilde);
      H1.slice(i) += Xtilde.t()*Xtilde/rho0 - c*(Xtilde.t()*v)*(v.t()*Xtilde);
      
      //Association parameters
      double inner_idx_length = sum(J_sub%(J_sub-1)/2);
      G2(1,0,i) += pow(sum(Etilde), 2)/2 - (Ji*(Ji-1)/2-inner_idx_length)*rho[1];
      H2.slice(i) = {{inner_idx_length,0},{0, Ji*(Ji-1)/2-inner_idx_length}};
    }
    
  } else if (q == 1) {
    for(int i = 0; i < I; i++) {
      //Extract various indices to loop/subset on
      arma::uvec I_idx_sub = I_idx[i]; I_idx_sub -= 1;
      double Ji = I_idx_sub.n_elem;
      double c = rho[0]/(rho0*(rho0+rho[0]*Ji));
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xtilde = std_design_mat.rows(I_idx_sub);
      arma::colvec Etilde = std_resid.rows(I_idx_sub);
      
      arma::rowvec Xtilde_sum = colSums(Xtilde);
      double Etilde_sum = sum(Etilde);
      
      //Summands of gradient and Hessian for each cluster
      G1.slice(i) += Xtilde.t()*Etilde/rho0 - c*Xtilde_sum.t()*Etilde_sum;
      H1.slice(i) += Xtilde.t()*Xtilde/rho0 - c*Xtilde_sum.t()*Xtilde_sum;
      
      //Now for Hessians/gradients of association parameters
      G2(0,0,i) = (pow(Etilde_sum, 2) - sum(Etilde % Etilde))/2 - Ji*(Ji-1)/2*rho[0];
      H2(0,0,i) = Ji*(Ji-1)/2;
    }
  }
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//Computes each iteration of Fisher scoring / Newton-Raphson
//Also return each individual summand comprising Hessian / gradient for use in sandwich estimator
//[[Rcpp::export]]
List gaussian_hier_solver(arma::colvec beta,
                          double phi,
                          arma::colvec rho,
                          arma::colvec outcome,
                          arma::mat design_mat,
                          arma::mat Info1,
                          List I_idx) {
  
  arma::colvec mu = design_mat*beta;                //Mean vector
  double rho0 = 1 - rho[0];                       //Remaining ICC; appears in many computations
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = phi*arma::ones(mu.n_elem);       //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  arma::mat std_design_mat = design_mat.each_col() % U_sqrt/phi;
  arma::colvec etilde_1(outcome.n_rows, arma::fill::zeros);
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Dimension of linear predictor
  int q = rho.n_rows;                               //Dimension of correlation terms
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H1_5(1, 1, I, arma::fill::zeros);      //Separated Hessian for dispersion
  arma::cube G1_5(1, 1, I, arma::fill::zeros);      //Separated gradient for dispersion
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  if (q == 2){
    
    for(int i = 0; i < I; i++) {
      List I_idx_sub = I_idx[i];                                                   //Sublist for current 1st-order cluster
      arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Indices for current 1st-order cluster, zero-indexed
      arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Vector of K_{ij} (misnomer)
      double Ji = sum(J_sub);                                                      //K_{i\cdot} (misnomer)
      
      //Precomputed values which appear many times in computation; see paper
      //U = diagonal matrix of Var(Y|X) = phi*vf, where vf = is variance function matrix
      //D^T V^{-1} E = X^T vf phi^{-1/2}*vf^{-1/2} R^{-1} phi^{-1/2}*vf^{-1/2} E
      //             = (X^T U^{1/2})/phi R^{-1} (U^{-1/2} E)
      arma::colvec v = myrep(1/rho0 - (rho[0] - rho[1])/(rho0*(rho0/J_sub + rho[0] - rho[1])), J_sub);
      
      double c = 1/(1/rho[1] + sum(v));
      arma::mat Xtilde = std_design_mat.rows(I_sub_uvec);
      arma::colvec Etilde = std_resid.rows(I_sub_uvec);
      
      //One portion of block-matrix multiplication involves matrix of 1's
      //Vectorize entire operation using special structure of matrix of 1's
      
      double a = 0;
      double a0 = 0;
      
      
      for (unsigned int j = 0; j < J_sub.n_rows; j++) {
        arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1;               //J_idx subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
        arma::rowvec temp_vec_X = colSums(Xtilde.rows(J_idx));
        arma::colvec temp_vec_E = Etilde.rows(J_idx);
        
        G1.slice(i) += -temp_vec_X.t()*sum(temp_vec_E)*(rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*J_sub[j]));
        H1.slice(i) += -temp_vec_X.t()*temp_vec_X*(rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*J_sub[j]));
        
        //Now for Hessians/gradients of dispersion
        a = phi*sum(square(temp_vec_E) - 1);
        G1_5(0,0,i) += a;
        H1_5(0,0,i) += J_sub[j];
        
        //Now for Hessians/gradients of association parameters
        double temp_double_E = sum(temp_vec_E);
        a0 = (pow(temp_double_E, 2) - sum(temp_vec_E % temp_vec_E))/2;
        G2(0,0,i) += a0 - J_sub[j]*(J_sub[j]-1)/2*rho[0];
        G2(1,0,i) -= (a/phi + J_sub[j])/2 + a0;
      }
      
      //Add on remaining terms of summands (don't require 'for loop')
      G1.slice(i) += Xtilde.t()*Etilde/rho0 - c*(Xtilde.t()*v)*(v.t()*Etilde);
      H1.slice(i) += Xtilde.t()*Xtilde/rho0 - c*(Xtilde.t()*v)*(v.t()*Xtilde);
      
      //Association parameters
      double inner_idx_length = sum(J_sub%(J_sub-1)/2);
      G2(1,0,i) += pow(sum(Etilde), 2)/2 - (Ji*(Ji-1)/2-inner_idx_length)*rho[1];
      H2.slice(i) = {{inner_idx_length,0},{0, Ji*(Ji-1)/2-inner_idx_length}};
    }
    
  } else if (q == 1) {
    for(int i = 0; i < I; i++) {
      //Extract various indices to loop/subset on
      arma::uvec I_idx_sub = I_idx[i]; I_idx_sub -= 1;
      double Ji = I_idx_sub.n_elem;
      double c = rho[0]/(rho0*(rho0+rho[0]*Ji));
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xtilde = std_design_mat.rows(I_idx_sub);
      arma::colvec Etilde = std_resid.rows(I_idx_sub);
      
      arma::rowvec Xtilde_sum = colSums(Xtilde);
      double Etilde_sum = sum(Etilde);
      
      //Summands of gradient and Hessian for each cluster
      G1.slice(i) += Xtilde.t()*Etilde/rho0 - c*Xtilde_sum.t()*Etilde_sum;
      H1.slice(i) += Xtilde.t()*Xtilde/rho0 - c*Xtilde_sum.t()*Xtilde_sum;
      
      G1_5(0,0,i) = phi*sum(square(Etilde) - 1);
      H1_5(0,0,i) = Ji;
      
      //Now for Hessians/gradients of association parameters
      G2(0,0,i) = (pow(Etilde_sum, 2) - sum(Etilde % Etilde))/2 - Ji*(Ji-1)/2*rho[0];
      H2(0,0,i) = Ji*(Ji-1)/2;
      //}
    }
  }
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G1_5") = G1_5,
                      Rcpp::Named("H1_5") = H1_5,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//[[Rcpp::export]]
List binomial_hier_sandwich(arma::colvec beta,
                            arma::colvec rho,
                            arma::colvec outcome,
                            arma::mat design_mat,
                            List I_idx) {
  
  arma::colvec mu = 1/(1+exp(-design_mat*beta));    //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec v = mu%(1-mu);                       //Variance vector
  arma::colvec vsqrt = sqrt(v);
  arma::colvec dvdm = 1-2*mu;                       //dv/dm
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Dimension of linear predictor
  int q = rho.n_rows;                               //Dimension of correlation terms
  
  arma::cube B(1, p, I, arma::fill::zeros); B.fill(1e-15);
  arma::cube D(q, p, I, arma::fill::zeros);
  arma::cube E(q, 1, I, arma::fill::zeros); E.fill(1e-15);
  
  if (q == 2){
    for(int i = 0; i < I; i++) {
      List I_idx_sub = I_idx[i];                                                   //Sublist for current 1st-order cluster
      arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Indices for current 1st-order cluster, zero-indexed
      arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Vector of K_{ij} (misnomer)
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xmat = design_mat.rows(I_sub_uvec); //Xtilde.each_col() %= U_sqrt.rows(I_idx_sub)/phi;
      arma::colvec Evec = resid(I_sub_uvec);
      arma::colvec vsub = v(I_sub_uvec);
      arma::colvec vsqrtsub = vsqrt(I_sub_uvec);
      arma::colvec dvdmsub = dvdm(I_sub_uvec);
      
      arma::mat XE_portion_1 = Xmat.each_col() % vsqrtsub + Xmat.each_col() % (Evec%dvdmsub/vsqrtsub)/2;
      arma::colvec XE_portion_2 = Evec/vsqrtsub;
      
      for (unsigned int j = 0; j < J_sub.n_rows; j++) {
        arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1;               //J_idx subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
        arma::mat temp_mat_XE_portion_1 = XE_portion_1.rows(J_idx);
        arma::colvec temp_vec_XE_portion_2 = XE_portion_2.rows(J_idx);
        arma::rowvec temp_vec_XE_portion_1 = colSums(temp_mat_XE_portion_1);
        double temp_double_XE_portion_2 = sum(temp_vec_XE_portion_2);
        
        D.slice(i).row(0) += (temp_vec_XE_portion_1*temp_double_XE_portion_2 - temp_vec_XE_portion_2.t()*temp_mat_XE_portion_1);
      }
      
      arma::rowvec temp_vec_XE_portion_1 = colSums(XE_portion_1);
      double temp_double_XE_portion_2 = sum(XE_portion_2);
      
      D.slice(i).row(1) = (temp_vec_XE_portion_1*temp_double_XE_portion_2 - XE_portion_2.t()*XE_portion_1) - D.slice(i).row(0);
    }
  } else if (q == 1) {
    for(int i = 0; i < I; i++) {
      //Extract various indices to loop/subset on
      arma::uvec I_idx_sub = I_idx[i]; I_idx_sub -= 1;
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xmat = design_mat.rows(I_idx_sub);
      arma::colvec Evec = resid(I_idx_sub);
      arma::colvec vsub = v(I_idx_sub);
      arma::colvec vsqrtsub = vsqrt(I_idx_sub);
      arma::colvec dvdmsub = dvdm(I_idx_sub);
      
      arma::mat XE_portion_1 = Xmat.each_col() % vsqrtsub + Xmat.each_col() % (Evec%dvdmsub/vsqrtsub)/2;
      arma::colvec XE_portion_2 = Evec/vsqrtsub;
      arma::rowvec temp_vec_XE_portion_1 = colSums(XE_portion_1);
      double temp_double_XE_portion_2 = sum(XE_portion_2);
      
      D.slice(i) = (temp_vec_XE_portion_1*temp_double_XE_portion_2 - XE_portion_2.t()*XE_portion_1);
      
    }
  }
  
  return List::create(Rcpp::Named("B") = B,
                      Rcpp::Named("D") = D,
                      Rcpp::Named("E") = E
  );
}

//[[Rcpp::export]]
List gaussian_hier_sandwich(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome,
                            arma::mat design_mat, List I_idx) {
  
  arma::colvec mu = design_mat*beta;    //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Dimension of linear predictor
  int q = rho.n_rows;                               //Dimension of correlation terms
  
  arma::cube B(1, p, I, arma::fill::zeros);
  arma::cube D(q, p, I, arma::fill::zeros);
  arma::cube E(q, 1, I, arma::fill::zeros);
  
  if (q == 2){
    for(int i = 0; i < I; i++) {
      List I_idx_sub = I_idx[i];                                                   //Sublist for current 1st-order cluster
      arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Indices for current 1st-order cluster, zero-indexed
      arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Vector of K_{ij} (misnomer)
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xmat = design_mat.rows(I_sub_uvec); //Xtilde.each_col() %= U_sqrt.rows(I_idx_sub)/phi;
      arma::colvec Evec = resid(I_sub_uvec);
      
      
      B.slice(i) = 2*Evec.t()*Xmat;
      
      for (unsigned int j = 0; j < J_sub.n_rows; j++) {
        arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1;               //J_idx subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
        arma::mat temp_mat_XE_portion_1 = Xmat.rows(J_idx);
        arma::colvec temp_vec_XE_portion_2 = Evec.rows(J_idx);
        arma::rowvec temp_vec_XE_portion_1 = colSums(temp_mat_XE_portion_1);
        double temp_double_XE_portion_2 = sum(temp_vec_XE_portion_2);
        
        D.slice(i).row(0) += (temp_vec_XE_portion_1*temp_double_XE_portion_2 - temp_vec_XE_portion_2.t()*temp_mat_XE_portion_1)/phi;
        E.slice(i)(0,0) += (temp_double_XE_portion_2*temp_double_XE_portion_2 - sum(temp_vec_XE_portion_2 % temp_vec_XE_portion_2))/(2*pow(phi, 2));
      }
      
      arma::rowvec temp_vec_XE_portion_1 = colSums(Xmat);
      double temp_double_XE_portion_2 = sum(Evec);
      
      D.slice(i).row(1) = (temp_vec_XE_portion_1*temp_double_XE_portion_2 - Evec.t()*Xmat)/phi - D.slice(i).row(0);
      E.slice(i)(1,0) = (temp_double_XE_portion_2*temp_double_XE_portion_2 - sum(Evec % Evec))/(2*pow(phi, 2)) - E.slice(i)(0,0);
    }
  } else if (q == 1) {
    for(int i = 0; i < I; i++) { //These standard errors aren't completing matching geepack, but simulations show that they attain nominal coverage
      //Extract various indices to loop/subset on
      arma::uvec I_idx_sub = I_idx[i]; I_idx_sub -= 1;
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xmat = design_mat.rows(I_idx_sub); //Xtilde.each_col() %= U_sqrt.rows(I_idx_sub)/phi;
      arma::colvec Evec = resid(I_idx_sub);
      
      B.slice(i) = 2*Evec.t()*Xmat;
      
      D.slice(i) = (colSums(Xmat)*sum(Evec)- Evec.t()*Xmat)/phi;
      
      E.slice(i) = (pow(sum(Evec), 2)- sum(Evec % Evec))/(2*pow(phi, 2));
    }
  }
  
  return List::create(Rcpp::Named("B") = B,
                      Rcpp::Named("D") = D,
                      Rcpp::Named("E") = E
  );
}

//Computes each iteration of Fisher scoring / Newton-Raphson
//Also return each individual summand comprising Hessian / gradient for use in sandwich estimator
//[[Rcpp::export]]
List binomial_tscs_solver(arma::colvec beta,
                          arma::colvec rho,
                          arma::colvec outcome,
                          arma::mat design_mat,
                          arma::mat Info1,
                          List I_idx) {
  arma::colvec mu = 1/(1+exp(-design_mat*beta));    //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = mu%(1-mu);                       //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  arma::mat std_design_mat = design_mat.each_col() % U_sqrt;
  arma::colvec etilde_1(outcome.n_rows, arma::fill::zeros);
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Number of coefficients
  int q = rho.n_rows;                               //Dimension of correlation terms
  int T = p - 1;                                    //Number of fixed effects (and time periods)
  
  //Quantities as defined in Li et al. (2018)
  double lambda1 = 1 - rho[0] + rho[1] - rho[2];
  double lambda2 = 1 - rho[0] - (T - 1)*(rho[1]-rho[2]);
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H1_5(1, 1, I, arma::fill::zeros);      //Separated Hessian for dispersion
  arma::cube G1_5(1, 1, I, arma::fill::zeros);      //Separated gradient for dispersion
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  for(int i = 0; i < I; i++) {
    List I_idx_sub = I_idx[i];                                                   //IDs of individuals/times in current cluster
    arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Zero indexing
    arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Number of individuals at each time point for current cluster
    double Ji = J_sub[0];                                                        //Total number of individuals in the cluster (equal to number of individuals at a specific time point, just take first time period for simplicity. Add checks later to verify all time periods have the same number of individuals)
    double TJi = sum(J_sub);
    
    //Precomputed values which appear many times in computation; see paper
    double lambda3 = 1 + (Ji - 1)*(rho[0] - rho[1]) - rho[2];
    double lambda4 = 1 + (Ji - 1)*rho[0] + (T-1)*(Ji-1)*rho[1] + (T-1)*rho[2];
    double c = (rho[2] - rho[1])*(rho[0]-rho[1])/(lambda1*lambda2*lambda3) + (rho[2]*rho[0] - rho[1])/(lambda2*lambda3*lambda4);
    arma::mat Xtilde = std_design_mat.rows(I_sub_uvec);
    arma::colvec Etilde = std_resid.rows(I_sub_uvec);
    
    arma::rowvec Xtilde_sum = colSums(Xtilde);
    double Etilde_sum = sum(Etilde);
    
    //One portion of block-matrix multiplication involves matrix of 1's
    //Vectorize entire operation using special structure of matrix of 1's
    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, 1, arma::fill::zeros);
    
    double resid_phi = 0;
    double resid2_0 = 0;
    double resid2_1 = 0;
    double resid2_2 = 0;
    
    for (unsigned int j = 0; j < J_sub.n_rows; j++) {
      arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1; //J_idx is used to subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
      arma::mat temp_mat_X = Xtilde.rows(J_idx);
      arma::rowvec temp_vec_X = colSums(temp_mat_X);
      arma::colvec temp_vec_E = Etilde.rows(J_idx);
      double temp_double_E = sum(temp_vec_E);
      
      Xtilde_sub = Xtilde_sub + temp_mat_X;
      Etilde_sub = Etilde_sub + temp_vec_E;
      
      
      G1.slice(i) += -temp_vec_X.t()*temp_double_E*(rho[0] - rho[1])/(lambda1*lambda3);
      H1.slice(i) += -temp_vec_X.t()*temp_vec_X*(rho[0] - rho[1])/(lambda1*lambda3);
      
      
      //Now for Hessians/gradients of dispersion
      resid_phi = sum(square(temp_vec_E));
      
      //Now for Hessians/gradients of association parameters
      resid2_0 = (pow(temp_double_E, 2) - resid_phi)/2;
      G2(0,0,i) += resid2_0 - J_sub[j]*(J_sub[j]-1)/2*rho[0];
      resid2_1 -= resid_phi/2 + resid2_0;
      resid2_2 -= resid_phi/2;
    }
    
    
    
    //Add on remaining terms of summands (don't require 'for loop')
    G1.slice(i) += Xtilde.t()*Etilde/lambda1 - Xtilde_sub.t()*Etilde_sub*(rho[2]-rho[1])/(lambda1*lambda2) + c*Xtilde_sum.t()*Etilde_sum;
    H1.slice(i) += Xtilde.t()*Xtilde/lambda1 - Xtilde_sub.t()*Xtilde_sub*(rho[2]-rho[1])/(lambda1*lambda2) + c*Xtilde_sum.t()*Xtilde_sum;
    
    
    //Now for Hessians/gradients of ICC
    double diff_ind_same_period_length = sum(J_sub%(J_sub-1)/2);                                                    //Estimate rho0
    double same_ind_diff_period_length = T*(T-1)/2*J_sub[0];                                                        //Estimate rho2 (note the switching around!)
    double diff_ind_and_period_length = TJi*(TJi-1)/2 - diff_ind_same_period_length - same_ind_diff_period_length;  //Estimate rho1
    
    resid2_2 += sum(Etilde_sub % Etilde_sub)/2;
    resid2_1 += pow(sum(Etilde), 2)/2 - resid2_2;
    
    G2(1,0,i) += resid2_1 - diff_ind_and_period_length*rho[1];
    G2(2,0,i) += resid2_2 - same_ind_diff_period_length*rho[2];
    
    H2.slice(i) = {{diff_ind_same_period_length, 0, 0},{0, diff_ind_and_period_length, 0},{0, 0, same_ind_diff_period_length}};
  }
  
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//Computes each iteration of Fisher scoring / Newton-Raphson
//Also return each individual summand comprising Hessian / gradient for use in sandwich estimator
//[[Rcpp::export]]
List gaussian_tscs_solver(arma::colvec beta,
                          double phi, arma::colvec rho,
                          arma::colvec outcome,
                          arma::mat design_mat,
                          arma::mat Info1,
                          List I_idx) {
  
  arma::colvec mu = design_mat*beta;                //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = phi*arma::ones(mu.n_elem);       //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  arma::mat std_design_mat = design_mat.each_col() % U_sqrt/phi;
  arma::colvec etilde_1(outcome.n_rows, arma::fill::zeros);
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Number of coefficients
  int q = rho.n_rows;                               //Dimension of correlation terms
  int T = p - 1;                                    //Number of fixed effects (and time periods)
  
  //Quantities as defined in Li et al. (2018)
  double lambda1 = 1 - rho[0] + rho[1] - rho[2];
  double lambda2 = 1 - rho[0] - (T - 1)*(rho[1]-rho[2]);
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H1_5(1, 1, I, arma::fill::zeros);      //Separated Hessian for dispersion
  arma::cube G1_5(1, 1, I, arma::fill::zeros);      //Separated gradient for dispersion
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  for(int i = 0; i < I; i++) {
    List I_idx_sub = I_idx[i];                                                   //IDs of individuals/times in current cluster
    arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Zero indexing
    arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Number of individuals at each time point for current cluster
    double Ji = J_sub[0];                                                        //Total number of individuals in the cluster (equal to number of individuals at a specific time point, just take first time period for simplicity. Add checks later to verify all time periods have the same number of individuals)
    double TJi = sum(J_sub);
    
    //Precomputed values which appear many times in computation; see paper
    double lambda3 = 1 + (Ji - 1)*(rho[0] - rho[1]) - rho[2];
    double lambda4 = 1 + (Ji - 1)*rho[0] + (T-1)*(Ji-1)*rho[1] + (T-1)*rho[2];
    double c = (rho[2] - rho[1])*(rho[0]-rho[1])/(lambda1*lambda2*lambda3) + (rho[2]*rho[0] - rho[1])/(lambda2*lambda3*lambda4);
    arma::mat Xtilde = std_design_mat.rows(I_sub_uvec);
    arma::colvec Etilde = std_resid.rows(I_sub_uvec);
    
    arma::rowvec Xtilde_sum = colSums(Xtilde);
    double Etilde_sum = sum(Etilde);
    
    //One portion of block-matrix multiplication involves matrix of 1's
    //Vectorize entire operation using special structure of matrix of 1's
    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, 1, arma::fill::zeros);
    
    double resid_phi = 0;
    double resid2_0 = 0;
    double resid2_1 = 0;
    double resid2_2 = 0;
    
    for (unsigned int j = 0; j < J_sub.n_rows; j++) {
      arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1; //J_idx is used to subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
      arma::mat temp_mat_X = Xtilde.rows(J_idx);
      arma::rowvec temp_vec_X = colSums(temp_mat_X);
      arma::colvec temp_vec_E = Etilde.rows(J_idx);
      double temp_double_E = sum(temp_vec_E);
      
      Xtilde_sub = Xtilde_sub + temp_mat_X;
      Etilde_sub = Etilde_sub + temp_vec_E;
      
      
      G1.slice(i) += -temp_vec_X.t()*temp_double_E*(rho[0] - rho[1])/(lambda1*lambda3);
      H1.slice(i) += -temp_vec_X.t()*temp_vec_X*(rho[0] - rho[1])/(lambda1*lambda3);
      
      
      //Now for Hessians/gradients of dispersion
      resid_phi = sum(square(temp_vec_E));
      G1_5(0,0,i) += phi*(resid_phi - J_sub[j]);
      H1_5(0,0,i) += J_sub[j];
      
      //Now for Hessians/gradients of association parameters
      resid2_0 = (pow(temp_double_E, 2) - resid_phi)/2;
      G2(0,0,i) += resid2_0 - J_sub[j]*(J_sub[j]-1)/2*rho[0];
      resid2_1 -= resid_phi/2 + resid2_0;
      resid2_2 -= resid_phi/2;
    }
    
    
    
    //Add on remaining terms of summands (don't require 'for loop')
    G1.slice(i) += Xtilde.t()*Etilde/lambda1 - Xtilde_sub.t()*Etilde_sub*(rho[2]-rho[1])/(lambda1*lambda2) + c*Xtilde_sum.t()*Etilde_sum;
    H1.slice(i) += Xtilde.t()*Xtilde/lambda1 - Xtilde_sub.t()*Xtilde_sub*(rho[2]-rho[1])/(lambda1*lambda2) + c*Xtilde_sum.t()*Xtilde_sum;
    
    //Now for Hessians/gradients of dispersion
    G1_5(0,0,i) = phi*sum(square(Etilde) - 1);
    H1_5(0,0,i) = TJi;
    
    //Now for Hessians/gradients of ICC
    double diff_ind_same_period_length = sum(J_sub%(J_sub-1)/2);                                                    //Estimate rho0
    double same_ind_diff_period_length = T*(T-1)/2*J_sub[0];                                                        //Estimate rho2 (note the switching around!)
    double diff_ind_and_period_length = TJi*(TJi-1)/2 - diff_ind_same_period_length - same_ind_diff_period_length;  //Estimate rho1
    
    resid2_2 += sum(Etilde_sub % Etilde_sub)/2;
    resid2_1 += pow(sum(Etilde), 2)/2 - resid2_2;
    
    G2(1,0,i) += resid2_1 - diff_ind_and_period_length*rho[1];
    G2(2,0,i) += resid2_2 - same_ind_diff_period_length*rho[2];
    
    H2.slice(i) = {{diff_ind_same_period_length, 0, 0},{0, diff_ind_and_period_length, 0},{0, 0, same_ind_diff_period_length}};
  }
  
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G1_5") = G1_5,
                      Rcpp::Named("H1_5") = H1_5,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//[[Rcpp::export]]
List binomial_tscs_sandwich(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome,
                            arma::mat design_mat, List I_idx) {
  
  arma::colvec mu = 1/(1+exp(-design_mat*beta));    //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec v = mu%(1-mu);                       //Variance vector
  arma::colvec vsqrt = sqrt(v);
  arma::colvec dvdm = 1-2*mu;                       //dv/dm
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Number of coefficients
  int q = rho.n_rows;                               //Dimension of correlation terms
  //int T = p - 1;                                  //Number of fixed effects (and time periods)
  
  arma::cube B(1, p, I, arma::fill::zeros); B.fill(1e-15);
  arma::cube D(q, p, I, arma::fill::zeros);
  arma::cube E(q, 1, I, arma::fill::zeros); E.fill(1e-15);
  
  for(int i = 0; i < I; i++) {
    List I_idx_sub = I_idx[i];                                                   //IDs of individuals/times in current cluster
    arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Zero indexing
    arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Number of individuals at each time point for current cluster
    double Ji = J_sub[0];                                                        //Total number of individuals in the cluster (equal to number of individuals at a specific time point, just take first time period for simplicity. Add checks later to verify all time periods have the same number of individuals)
    //double TJi = sum(J_sub);
    
    //Precomputed values which appear many times in computation; see paper
    arma::mat Xmat = design_mat.rows(I_sub_uvec); //Xtilde.each_col() %= U_sqrt.rows(I_idx_sub)/phi;
    arma::colvec Evec = resid(I_sub_uvec);
    arma::colvec vsub = v(I_sub_uvec);
    arma::colvec vsqrtsub = vsqrt(I_sub_uvec);
    arma::colvec dvdmsub = dvdm(I_sub_uvec);
    
    
    
    arma::mat XE_portion_1 = Xmat.each_col() % vsqrtsub + Xmat.each_col() % (Evec%dvdmsub/vsqrtsub)/2;
    arma::colvec XE_portion_2 = Evec/vsqrtsub;
    
    arma::mat XE_sub_1(Ji, p, arma::fill::zeros);
    arma::colvec XE_sub_2(Ji, arma::fill::zeros);
    arma::rowvec D_resid(p, arma::fill::zeros);
    
    
    for (unsigned int j = 0; j < J_sub.n_rows; j++) {
      arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1;               //J_idx subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
      arma::mat temp_mat_XE_portion_1 = XE_portion_1.rows(J_idx);
      arma::colvec temp_vec_XE_portion_2 = XE_portion_2.rows(J_idx);
      arma::rowvec temp_vec_XE_portion_1 = colSums(temp_mat_XE_portion_1);
      double temp_double_XE_portion_2 = sum(temp_vec_XE_portion_2);
      
      D_resid += temp_vec_XE_portion_2.t()*temp_mat_XE_portion_1;
      
      
      D.slice(i).row(0) += temp_vec_XE_portion_1*temp_double_XE_portion_2/phi;
      
      XE_sub_1 += temp_mat_XE_portion_1;
      XE_sub_2 += temp_vec_XE_portion_2;
      
    }
    
    arma::rowvec temp_vec_XE_portion_1 = colSums(XE_portion_1);
    double temp_double_XE_portion_2 = sum(XE_portion_2);
    
    D.slice(i).row(0) -= D_resid/phi;
    D.slice(i).row(2) = XE_sub_2.t()*XE_sub_1/phi - D_resid/phi;
    D.slice(i).row(1) = temp_vec_XE_portion_1*temp_double_XE_portion_2/phi - D.slice(i).row(2) - D.slice(i).row(0) - D_resid/phi;
  }
  
  return List::create(Rcpp::Named("B") = B,
                      Rcpp::Named("D") = D,
                      Rcpp::Named("E") = E
  );
}

//[[Rcpp::export]]
List gaussian_tscs_sandwich(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome,
                            arma::mat design_mat, List I_idx) {
  
  arma::colvec mu = design_mat*beta;                //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Number of coefficients
  int q = rho.n_rows;                               //Dimension of correlation terms
  //int T = p - 1;                                    //Number of fixed effects (and time periods)
  
  arma::cube B(1, p, I, arma::fill::zeros);
  arma::cube D(q, p, I, arma::fill::zeros);
  arma::cube E(q, 1, I, arma::fill::zeros);
  
  for(int i = 0; i < I; i++) {
    List I_idx_sub = I_idx[i];                                                   //IDs of individuals/times in current cluster
    arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //Zero indexing
    arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Number of individuals at each time point for current cluster
    double Ji = J_sub[0];                                                        //Total number of individuals in the cluster (equal to number of individuals at a specific time point, just take first time period for simplicity. Add checks later to verify all time periods have the same number of individuals)
    //double TJi = sum(J_sub);
    
    //Precomputed values which appear many times in computation; see paper
    arma::mat Xmat = design_mat.rows(I_sub_uvec); //Xtilde.each_col() %= U_sqrt.rows(I_idx_sub)/phi;
    arma::colvec Evec = resid(I_sub_uvec);
    
    
    B.slice(i) = 2*Evec.t()*Xmat;
    
    
    arma::mat XE_sub_1(Ji, p, arma::fill::zeros);
    arma::colvec XE_sub_2(Ji, arma::fill::zeros);
    arma::rowvec D_resid(p, arma::fill::zeros);
    double E_resid = 0;
    for (unsigned int j = 0; j < J_sub.n_rows; j++) {
      arma::uvec J_idx = I_idx_sub[j]; J_idx -= min(I_sub_uvec)+1;               //J_idx subset Xtilde, Etilde; these are subvectors and therefore J_idx has to be relative to this subvectors
      arma::mat temp_mat_XE_portion_1 = Xmat.rows(J_idx);
      arma::colvec temp_vec_XE_portion_2 = Evec.rows(J_idx);
      arma::rowvec temp_vec_XE_portion_1 = colSums(temp_mat_XE_portion_1);
      double temp_double_XE_portion_2 = sum(temp_vec_XE_portion_2);
      
      D_resid += temp_vec_XE_portion_2.t()*temp_mat_XE_portion_1;
      E_resid += sum(temp_vec_XE_portion_2 % temp_vec_XE_portion_2);
      
      
      D.slice(i).row(0) += temp_vec_XE_portion_1*temp_double_XE_portion_2/phi;
      E.slice(i)(0,0) += temp_double_XE_portion_2*temp_double_XE_portion_2/(2*pow(phi, 2));
      
      XE_sub_1 += temp_mat_XE_portion_1;
      XE_sub_2 += temp_vec_XE_portion_2;
      
    }
    
    arma::rowvec temp_vec_XE_portion_1 = colSums(Xmat);
    double temp_double_XE_portion_2 = sum(Evec);
    
    D.slice(i).row(0) -= D_resid/phi;
    D.slice(i).row(2) = XE_sub_2.t()*XE_sub_1/phi - D_resid/phi;
    D.slice(i).row(1) = temp_vec_XE_portion_1*temp_double_XE_portion_2/phi - D.slice(i).row(2) - D.slice(i).row(0) - D_resid/phi;
    
    E.slice(i)(0,0) -= E_resid/(2*pow(phi, 2));
    E.slice(i)(2,0) = sum(XE_sub_2 % XE_sub_2)/(2*pow(phi, 2)) - E_resid/(2*pow(phi, 2));
    E.slice(i)(1,0) = temp_double_XE_portion_2*temp_double_XE_portion_2/(2*pow(phi, 2)) - E.slice(i)(2,0) - E.slice(i)(0,0) - E_resid/(2*pow(phi, 2));
  }
  
  return List::create(Rcpp::Named("B") = B,
                      Rcpp::Named("D") = D,
                      Rcpp::Named("E") = E
  );
}

//[[Rcpp::export]]
List stochastic_binomial_hier_solver(arma::colvec beta,
                                     arma::colvec rho,
                                     arma::colvec outcome,
                                     arma::mat design_mat,
                                     List I_idx,               //Indices for sub-cluster
                                     int I_full,               //Number of 1st-order clusters, full sample
                                     arma::colvec J_full,      //Number of 2nd-order clusters, full sample
                                     List K_full               //Number of 3rd-order clusters, full sample
) {
  
  arma::colvec mu = 1/(1+exp(-design_mat*beta));    //Mean vector
  double rho0 = 1 - rho[0];                       //Remaining ICC; appears in many computations
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = mu%(1-mu);                   //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Dimension of linear predictor
  int q = rho.n_rows;                               //Dimension of correlation terms
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  //A bunch more constants
  double Ji;
  double Kidot;
  double IJK;
  double K_minus_1;
  double J_minus_1_K;
  double m11;
  double m12;
  double m21;
  double m22;
  double m32;
  
  //Intermediary values to hold summed and weighted-summed operations of X matrices and E vectors
  arma::rowvec temp_vec_X(p, arma::fill::zeros);
  arma::rowvec temp_vec_v_X(p, arma::fill::zeros);
  double temp_double_E = 0;
  double temp_double_v_E = 0;
  
  if (q == 2){
    int idx_val = 0;
    for(int i = 0; i < I; i++) {
      List I_idx_sub = I_idx[i];                                                   //(Subsample) Sublist for current 1st-order cluster
      arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //(Subsample) Indices for current 1st-order cluster, one-indexed
      arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //(Subsample) K_S (as vector)
      Ji = J_sub.n_rows;                                                           //(Subsample) J_S
      Kidot = sum(J_sub);                                                          //(Subsample) K_{i\cdot}
      arma::colvec Kij = K_full[i];                                                //(Full sample) Vector of K_{ij}
      
      //Precomputed values which appear many times in computation; see paper
      //D^T V^{-1} E = X^T U phi^{-1/2}*U^{-1/2} R^{-1} phi^{-1/2}*U^{-1/2} E = (X^T U^{1/2})/phi^{1/2} R^{-1} (U^{-1/2} E)
      arma::colvec vij = 1/rho0 - (rho[0]-rho[1])/(rho0*(rho0/Kij + rho[0]-rho[1]));
      arma::colvec v = myrep(vij, J_sub);
      double c = 1/(1/rho[1] + sum(vij%Kij));
      
      arma::mat Xtilde = design_mat.rows(idx_val, idx_val + Kidot - 1); Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + Kidot - 1);
      arma::colvec Etilde = std_resid.rows(idx_val, idx_val + Kidot - 1);
      arma::mat Xtildev = Xtilde.each_col()%v;
      arma::colvec Etildev = Etilde%v;
      
      //M2 operations vectorized
      double a = 0;
      double a0 = 0;
      double a1 = 0;
      double b1 = 0;
      int idx_val2 = 0;
      double G1temp = 0;
      arma::rowvec H1temp(p, arma::fill::zeros);
      for (unsigned int j = 0; j < J_sub.n_rows; j++) {
        //A bunch of constants; better to declare right now
        IJK = I_full*J_full[i]*Kij[j]/(I*Ji*J_sub[j]);
        K_minus_1 = (Kij[j]-1)/(J_sub[j]-1);
        J_minus_1_K = ((J_full[i]-1)*Kij[j])/((Ji-1)*J_sub[j]);
        
        //These are going to act as coefficients to each of the decomposition components of the stochastic portions
        m11 = IJK/rho0 + (rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*Kij[j]))*(K_minus_1-1)*IJK;   //Second portion equals 0 if full sample
        m12 = c*(K_minus_1-1)*IJK;                                               //Equals 0 if full sample
        m21 = (rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*Kij[j]))*K_minus_1*IJK;
        m22 = c*(K_minus_1 - J_minus_1_K)*IJK;                                   //Equals 0 if full sample
        m32 = sqrt(c*J_minus_1_K*IJK);
        
        //Various versions of design matrix X and residual matrix E (full, multiplied by v, column-summed, multiplied by v & column-summed)
        arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        arma::mat temp_mat_v_X = Xtildev.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        temp_vec_X = colSums(temp_mat_X);
        temp_vec_v_X = colSums(temp_mat_v_X);
        arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        arma::colvec temp_vec_v_E = Etildev.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        temp_double_E = sum(temp_vec_E);
        temp_double_v_E = sum(temp_vec_v_E);
        
        
        G1.slice(i) += m11*temp_mat_X.t()*temp_vec_E + m12*temp_mat_v_X.t()*temp_vec_v_E - m21*temp_vec_X.t()*temp_double_E - m22*temp_vec_v_X.t()*temp_double_v_E;
        
        G1temp += m32*temp_double_v_E;
        
        H1.slice(i) += m11*temp_mat_X.t()*temp_mat_X + m12*temp_mat_v_X.t()*temp_mat_v_X - m21*temp_vec_X.t()*temp_vec_X - m22*temp_vec_v_X.t()*temp_vec_v_X;
        
        H1temp += m32*temp_vec_v_X;
        
        //Now for dispersion
        a = IJK*sum(square(temp_vec_E) - 1);
        
        //Now for Hessian
        double temp_double_E = sum(temp_vec_E);
        a0 = IJK*K_minus_1*(pow(temp_double_E, 2) - sum(temp_vec_E % temp_vec_E))/2;
        G2(0,0,i) += a0 - IJK*K_minus_1*J_sub[j]*(J_sub[j]-1)/2*rho[0];
        G2(1,0,i) -= ((a/(IJK) + J_sub[j])/2 + a0/(IJK*K_minus_1))*IJK*J_minus_1_K;
        
        double inner_idx_length1 = IJK*K_minus_1*J_sub[j]*(J_sub[j]-1)/2;
        
        H2.slice(i) += {{inner_idx_length1,0}, {0, -(J_sub[j]/2 + J_sub[j]*(J_sub[j]-1)/2)*IJK*J_minus_1_K}};
        
        a1 += sqrt(IJK*J_minus_1_K)*sum(temp_vec_E);
        b1 += sqrt(IJK*J_minus_1_K)*J_sub[j];
        
        idx_val2 += J_sub[j];
      }
      
      //Add on remaining terms of summands (don't require 'for loop')
      G1.slice(i) -= H1temp.t()*G1temp;
      H1.slice(i) -= H1temp.t()*H1temp;
      
      //Association parameters
      //double inner_idx_length = sum(J_sub%(J_sub-1)/2);
      H2.slice(i) += {{0,0},{0, pow(b1, 2)/2}};
      G2(1,0,i) += pow(a1, 2)/2 - H2(1,1,i)*rho[1];
      
      idx_val += Kidot;
    }
  } else if (q == 1) {
    int idx_val = 0;
    for(int i = 0; i < I; i++) {
      //Extract various indices to loop/subset on
      arma::uvec I_idx_sub = I_idx[i]; I_idx_sub -= 1;
      double Ji = I_idx_sub.n_elem;
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xtilde = design_mat.rows(idx_val, idx_val + Ji - 1); Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + Ji - 1);
      arma::colvec Etilde = std_resid.rows(idx_val, idx_val + Ji - 1);
      
      arma::rowvec Xtilde_sum = colSums(Xtilde);
      double Etilde_sum = sum(Etilde);
      
      //Summands of gradient and Hessian for each cluster
      double const1 = (I_full*J_full[i])/(I*Ji)*(1/rho0 + rho[0]/(rho0*(rho0+rho[0]*J_full[i]))*(J_full[i]-Ji)/(Ji-1));
      double const2 = (I_full*J_full[i]*(J_full[i]-1))/(I*Ji*(Ji-1))*rho[0]/(rho0*(rho0+rho[0]*J_full[i]));
      G1.slice(i) += const1*Xtilde.t()*Etilde - const2*Xtilde_sum.t()*Etilde_sum;
      H1.slice(i) += const1*Xtilde.t()*Xtilde - const2*Xtilde_sum.t()*Xtilde_sum;
      
      
      //Now for Hessians/gradients of association parameters
      G2(0,0,i) = (I_full*J_full[i]*(J_full[i]-1))/(I*Ji*(Ji-1))*((pow(Etilde_sum, 2) - sum(Etilde % Etilde))/2 - Ji*(Ji-1)/2*rho[0]);
      H2(0,0,i) = (I_full*J_full[i]*(J_full[i]-1))/(2*I);
      
      idx_val += Ji;
    }
  }
  
  // Rcout << beta << "\n";
  // Rcout << phi << "\n";
  // Rcout << rho << "\n";
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//[[Rcpp::export]]
List stochastic_gaussian_hier_solver(arma::colvec beta, double phi, arma::colvec rho,
                                     arma::colvec outcome,
                                     arma::mat design_mat,
                                     List I_idx,               //Indices for sub-cluster
                                     int I_full,               //Number of 1st-order clusters, full sample
                                     arma::colvec J_full,      //Number of 2nd-order clusters, full sample
                                     List K_full               //Number of 3rd-order clusters, full sample
) {
  
  arma::colvec mu = design_mat*beta;                //Mean vector
  double rho0 = 1 - rho[0];                         //Remaining ICC; appears in many computations
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = phi*arma::ones(mu.n_elem);       //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Dimension of linear predictor
  int q = rho.n_rows;                               //Dimension of correlation terms
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H1_5(1, 1, I, arma::fill::zeros);      //Separated Hessian for dispersion
  arma::cube G1_5(1, 1, I, arma::fill::zeros);      //Separated gradient for dispersion
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  //A bunch more constants
  double Ji;
  double Kidot;
  double IJK;
  double K_minus_1;
  double J_minus_1_K;
  double m11;
  double m12;
  double m21;
  double m22;
  double m32;
  
  //Intermediary values to hold summed and weighted-summed operations of X matrices and E vectors
  arma::rowvec temp_vec_X(p, arma::fill::zeros);
  arma::rowvec temp_vec_v_X(p, arma::fill::zeros);
  double temp_double_E = 0;
  double temp_double_v_E = 0;
  
  if (q == 2){
    int idx_val = 0;
    for(int i = 0; i < I; i++) {
      List I_idx_sub = I_idx[i];                                                   //(Subsample) Sublist for current 1st-order cluster
      arma::uvec I_sub_uvec = as<arma::uvec>(combine(I_idx_sub)); I_sub_uvec -= 1; //(Subsample) Indices for current 1st-order cluster, one-indexed
      arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //(Subsample) K_S (as vector)
      Ji = J_sub.n_rows;                                                           //(Subsample) J_S
      Kidot = sum(J_sub);                                                          //(Subsample) K_{i\cdot}
      arma::colvec Kij = K_full[i];                                                //(Full sample) Vector of K_{ij}
      
      //Precomputed values which appear many times in computation; see paper
      //D^T V^{-1} E = X^T U phi^{-1/2}*U^{-1/2} R^{-1} phi^{-1/2}*U^{-1/2} E = (X^T U^{1/2})/phi^{1/2} R^{-1} (U^{-1/2} E)
      arma::colvec vij = 1/rho0 - (rho[0]-rho[1])/(rho0*(rho0/Kij + rho[0]-rho[1]));
      arma::colvec v = myrep(vij, J_sub);
      double c = 1/(1/rho[1] + sum(vij%Kij));
      
      arma::mat Xtilde = design_mat.rows(idx_val, idx_val + Kidot - 1); Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + Kidot - 1)/phi;
      arma::colvec Etilde = std_resid.rows(idx_val, idx_val + Kidot - 1);
      arma::mat Xtildev = Xtilde.each_col()%v;
      arma::colvec Etildev = Etilde%v;
      
      //M2 operations vectorized
      double a = 0;
      double a0 = 0;
      double a1 = 0;
      double b1 = 0;
      int idx_val2 = 0;
      double G1temp = 0;
      arma::rowvec H1temp(p, arma::fill::zeros);
      for (unsigned int j = 0; j < J_sub.n_rows; j++) {
        //A bunch of constants; better to declare right now
        IJK = I_full*J_full[i]*Kij[j]/(I*Ji*J_sub[j]);
        K_minus_1 = (Kij[j]-1)/(J_sub[j]-1);
        J_minus_1_K = ((J_full[i]-1)*Kij[j])/((Ji-1)*J_sub[j]);
        
        //These are going to act as coefficients to each of the decomposition components of the stochastic portions
        m11 = IJK/rho0 + (rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*Kij[j]))*(K_minus_1-1)*IJK;   //Second portion equals 0 if full sample
        m12 = c*(K_minus_1-1)*IJK;                                               //Equals 0 if full sample
        m21 = (rho[0]-rho[1])/(rho0*(rho0+(rho[0]-rho[1])*Kij[j]))*K_minus_1*IJK;
        m22 = c*(K_minus_1 - J_minus_1_K)*IJK;                                   //Equals 0 if full sample
        m32 = sqrt(c*J_minus_1_K*IJK);
        
        //Various versions of design matrix X and residual matrix E (full, multiplied by v, column-summed, multiplied by v & column-summed)
        arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        arma::mat temp_mat_v_X = Xtildev.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        temp_vec_X = colSums(temp_mat_X);
        temp_vec_v_X = colSums(temp_mat_v_X);
        arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        arma::colvec temp_vec_v_E = Etildev.rows(idx_val2, idx_val2 + J_sub[j] - 1);
        temp_double_E = sum(temp_vec_E);
        temp_double_v_E = sum(temp_vec_v_E);
        
        
        G1.slice(i) += m11*temp_mat_X.t()*temp_vec_E + m12*temp_mat_v_X.t()*temp_vec_v_E - m21*temp_vec_X.t()*temp_double_E - m22*temp_vec_v_X.t()*temp_double_v_E;
        
        G1temp += m32*temp_double_v_E;
        
        H1.slice(i) += m11*temp_mat_X.t()*temp_mat_X + m12*temp_mat_v_X.t()*temp_mat_v_X - m21*temp_vec_X.t()*temp_vec_X - m22*temp_vec_v_X.t()*temp_vec_v_X;
        
        H1temp += m32*temp_vec_v_X;
        
        //Now for dispersion
        a = IJK*phi*sum(square(temp_vec_E) - 1);
        G1_5(0,0,i) += a;
        H1_5(0,0,i) += IJK*J_sub[j];
        
        //Now for Hessian
        double temp_double_E = sum(temp_vec_E);
        a0 = IJK*K_minus_1*(pow(temp_double_E, 2) - sum(temp_vec_E % temp_vec_E))/2;
        G2(0,0,i) += a0 - IJK*K_minus_1*J_sub[j]*(J_sub[j]-1)/2*rho[0];
        G2(1,0,i) -= ((a/(IJK*phi) + J_sub[j])/2 + a0/(IJK*K_minus_1))*IJK*J_minus_1_K;
        
        double inner_idx_length1 = IJK*K_minus_1*J_sub[j]*(J_sub[j]-1)/2;
        
        H2.slice(i) += {{inner_idx_length1,0}, {0, -(J_sub[j]/2 + J_sub[j]*(J_sub[j]-1)/2)*IJK*J_minus_1_K}};
        
        a1 += sqrt(IJK*J_minus_1_K)*sum(temp_vec_E);
        b1 += sqrt(IJK*J_minus_1_K)*J_sub[j];
        
        idx_val2 += J_sub[j];
      }
      
      //Add on remaining terms of summands (don't require 'for loop')
      G1.slice(i) -= H1temp.t()*G1temp;
      H1.slice(i) -= H1temp.t()*H1temp;
      
      //Association parameters
      //double inner_idx_length = sum(J_sub%(J_sub-1)/2);
      H2.slice(i) += {{0,0},{0, pow(b1, 2)/2}};
      G2(1,0,i) += pow(a1, 2)/2 - H2(1,1,i)*rho[1];
      
      idx_val += Kidot;
    }
  } else if (q == 1) {
    int idx_val = 0;
    for(int i = 0; i < I; i++) {
      //Extract various indices to loop/subset on
      arma::uvec I_idx_sub = I_idx[i]; I_idx_sub -= 1;
      double Ji = I_idx_sub.n_elem;
      
      //Precomputed values which appear many times in computation; see paper
      arma::mat Xtilde = design_mat.rows(idx_val, idx_val + Ji - 1); Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + Ji - 1)/phi;
      arma::colvec Etilde = std_resid.rows(idx_val, idx_val + Ji - 1);
      
      arma::rowvec Xtilde_sum = colSums(Xtilde);
      double Etilde_sum = sum(Etilde);
      
      //Summands of gradient and Hessian for each cluster
      double const1 = (I_full*J_full[i])/(I*Ji)*(1/rho0 + rho[0]/(rho0*(rho0+rho[0]*J_full[i]))*(J_full[i]-Ji)/(Ji-1));
      double const2 = (I_full*J_full[i]*(J_full[i]-1))/(I*Ji*(Ji-1))*rho[0]/(rho0*(rho0+rho[0]*J_full[i]));
      G1.slice(i) += const1*Xtilde.t()*Etilde - const2*Xtilde_sum.t()*Etilde_sum;
      H1.slice(i) += const1*Xtilde.t()*Xtilde - const2*Xtilde_sum.t()*Xtilde_sum;
      
      //Now for Hessians/gradients of dispersion
      arma::colvec std_resid_subvec = std_resid.rows(idx_val, idx_val + Ji - 1);
      G1_5(0,0,i) = (I_full*J_full[i])/(I*Ji)*phi*sum(square(std_resid_subvec) - 1);
      H1_5(0,0,i) = (I_full*J_full[i])/I;
      
      //Now for Hessians/gradients of association parameters
      G2(0,0,i) = (I_full*J_full[i]*(J_full[i]-1))/(I*Ji*(Ji-1))*((pow(Etilde_sum, 2) - sum(Etilde % Etilde))/2 - Ji*(Ji-1)/2*rho[0]);
      H2(0,0,i) = (I_full*J_full[i]*(J_full[i]-1))/(2*I);
      
      idx_val += Ji;
    }
  }
  
  // Rcout << beta << "\n";
  // Rcout << phi << "\n";
  // Rcout << rho << "\n";
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G1_5") = G1_5,
                      Rcpp::Named("H1_5") = H1_5,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//[[Rcpp::export]]
List stochastic_binomial_tscs_solver(arma::colvec beta,
                                     arma::colvec rho,
                                     arma::colvec outcome,
                                     arma::mat design_mat,
                                     List I_idx,               //Indices for sub-cluster
                                     int I_full,               //Number of 1st-order clusters, full sample
                                     arma::colvec J_full,      //Number of 2nd-order clusters, full sample
                                     List K_full               //Number of 3rd-order clusters, full sample)
) {
  
  arma::colvec mu = 1/(1+exp(-design_mat*beta));    //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = mu%(1-mu);                       //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Number of coefficients
  int q = rho.n_rows;                               //Dimension of correlation terms
  int T = p - 1;                                    //Number of fixed effects (and time periods)
  
  //Quantities as defined in Li et al. (2018)
  double lambda1 = 1 - rho[0] + rho[1] - rho[2];
  double lambda2 = 1 - rho[0] - (T - 1)*(rho[1]-rho[2]);
  
  double c1 = 0;
  double c2 = 0;
  double c3 = 0;
  double c4 = 0;
  double IK = 0;
  double K_minus_1 = 0;
  double c1_tilde = 0;
  double c2_tilde = 0;
  double c3_tilde = 0;
  double c4_tilde = 0;
  
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  int idx_val = 0;
  for(int i = 0; i < I; i++) {
    List I_idx_sub = I_idx[i];                                                   //IDs of individuals/times in current cluster
    arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Number of individuals at each time point for current cluster
    double Ji = J_sub[0];                                                        //(Subsample) Individuals in cluster i (at any time point)
    arma::colvec Kij = K_full[i];                                                //(Full sample) Vector of K_{ij}
    double J_full = Kij[0];                                                      //(Full sample) Total number of individuals in the cluster
    double TJi = sum(J_sub);
    
    
    //Precomputed values which appear many times in computation; see paper
    double lambda3 = 1 + (J_full - 1)*(rho[0] - rho[1]) - rho[2];
    double lambda4 = 1 + (J_full - 1)*rho[0] + (T-1)*(J_full-1)*rho[1] + (T-1)*rho[2];
    
    c1 = 1/lambda1;
    c2 = (rho[2]-rho[1])/(lambda1*lambda2);
    c3 = (rho[0] - rho[1])/(lambda1*lambda3);
    c4 = (rho[2] - rho[1])*(rho[0]-rho[1])/(lambda1*lambda2*lambda3) + (rho[2]*rho[0] - rho[1])/(lambda2*lambda3*lambda4);
    
    IK = I_full*Kij[0]/(I*J_sub[0]);
    K_minus_1 = (Kij[0]-1)/(J_sub[0]-1);
    
    c1_tilde = c1*IK + c3*IK*(K_minus_1-1);
    c2_tilde = c2*IK + c4*IK*(K_minus_1-1);
    c3_tilde = c3*IK*K_minus_1;
    c4_tilde = c4*IK*K_minus_1;
    
    arma::mat Xtilde = design_mat.rows(idx_val, idx_val + TJi - 1); Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + TJi - 1);
    arma::colvec Etilde = std_resid.rows(idx_val, idx_val + TJi - 1);
    
    arma::rowvec Xtilde_sum = colSums(Xtilde);
    double Etilde_sum = sum(Etilde);
    
    //One portion of block-matrix multiplication involves matrix of 1's
    //Vectorize entire operation using special structure of matrix of 1's
    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, 1, arma::fill::zeros);
    int idx_val2 = 0;
    
    double resid_phi = 0;
    double resid2_0 = 0;
    double resid2_1 = 0;
    double resid2_2 = 0;
    
    for (unsigned int j = 0; j < J_sub.n_rows; j++) {
      arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
      arma::rowvec temp_vec_X = colSums(temp_mat_X);
      arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
      double temp_double_E = sum(temp_vec_E);
      
      Xtilde_sub = Xtilde_sub + temp_mat_X;
      Etilde_sub = Etilde_sub + temp_vec_E;
      
      
      G1.slice(i) += -c3_tilde*temp_vec_X.t()*temp_double_E;
      H1.slice(i) += -c3_tilde*temp_vec_X.t()*temp_vec_X;
      
      
      //Now for Hessians/gradients of dispersion
      resid_phi = IK*sum(square(temp_vec_E));
      
      //Now for Hessians/gradients of association parameters
      resid2_0 = (pow(temp_double_E, 2) - resid_phi/IK)/2;         //All e_ij*e_ij' for individuals in cluster
      G2(0,0,i) += IK*K_minus_1*(resid2_0 - J_sub[j]*(J_sub[j]-1)/2*rho[0]);
      resid2_1 -= resid_phi/IK/2 + resid2_0;                          //-(e_i1 + ... + e_iJ_i)^2/2
      resid2_2 -= resid_phi/IK/2;                                     //-(e_i1^2 + ... + e_iJ_i^2)/2
      
      idx_val2 += J_sub[j];
    }
    
    
    
    //Add on remaining terms of summands (don't require 'for loop')
    G1.slice(i) += c1_tilde*Xtilde.t()*Etilde - c2_tilde*Xtilde_sub.t()*Etilde_sub + c4_tilde*Xtilde_sum.t()*Etilde_sum;
    H1.slice(i) += c1_tilde*Xtilde.t()*Xtilde - c2_tilde*Xtilde_sub.t()*Xtilde_sub + c4_tilde*Xtilde_sum.t()*Xtilde_sum;
    
    
    
    //Now for Hessians/gradients of ICC
    double diff_ind_same_period_length = sum(J_sub%(J_sub-1)/2);                                                    //Estimate rho0
    double same_ind_diff_period_length = T*(T-1)/2*J_sub[0];                                                        //Estimate rho2 (note the switching around!)
    double diff_ind_and_period_length = TJi*(TJi-1)/2 - diff_ind_same_period_length - same_ind_diff_period_length;  //Estimate rho1
    
    resid2_2 += sum(Etilde_sub % Etilde_sub)/2;
    resid2_1 += pow(sum(Etilde), 2)/2 - resid2_2;
    
    G2(1,0,i) += IK*K_minus_1*(resid2_1 - diff_ind_and_period_length*rho[1]);
    G2(2,0,i) += IK*(resid2_2 - same_ind_diff_period_length*rho[2]);
    
    H2.slice(i) = {{IK*K_minus_1*diff_ind_same_period_length, 0, 0},{0, IK*K_minus_1*diff_ind_and_period_length, 0},{0, 0, IK*same_ind_diff_period_length}};
    
    idx_val += TJi;
  }
  
  //Rcout << H1 << "\n";
  //Rcout << H1_5 << "\n";
  //Rcout << H2 << "\n";
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//[[Rcpp::export]]
List stochastic_gaussian_tscs_solver(arma::colvec beta,
                                     double phi,
                                     arma::colvec rho,
                                     arma::colvec outcome,
                                     arma::mat design_mat,
                                     List I_idx,               //Indices for sub-cluster
                                     int I_full,               //Number of 1st-order clusters, full sample
                                     arma::colvec J_full,      //Number of 2nd-order clusters, full sample
                                     List K_full               //Number of 3rd-order clusters, full sample)
) {
  
  arma::colvec mu = design_mat*beta;                //Mean vector
  arma::colvec resid = outcome - mu;                //Residual vector
  arma::colvec U = phi*arma::ones(mu.n_elem);       //Variance vector
  arma::colvec U_sqrt = sqrt(U);                    //Std Dev vector
  arma::colvec std_resid = resid/U_sqrt;            //Standardized residuals (used in hierarchical portion)
  
  int I = I_idx.size();                             //Number of superclusters
  int p = beta.n_rows;                              //Number of coefficients
  int q = rho.n_rows;                               //Dimension of correlation terms
  int T = p - 1;                                    //Number of fixed effects (and time periods)
  
  //Quantities as defined in Li et al. (2018)
  double lambda1 = 1 - rho[0] + rho[1] - rho[2];
  double lambda2 = 1 - rho[0] - (T - 1)*(rho[1]-rho[2]);
  
  double c1 = 0;
  double c2 = 0;
  double c3 = 0;
  double c4 = 0;
  double IK = 0;
  double K_minus_1 = 0;
  double c1_tilde = 0;
  double c2_tilde = 0;
  double c3_tilde = 0;
  double c4_tilde = 0;
  
  
  arma::cube H1(p, p, I, arma::fill::zeros);        //Separated Hessian for mean effects
  arma::cube G1(p, 1, I, arma::fill::zeros);        //Separated gradient for mean effects
  arma::cube H1_5(1, 1, I, arma::fill::zeros);      //Separated Hessian for dispersion
  arma::cube G1_5(1, 1, I, arma::fill::zeros);      //Separated gradient for dispersion
  arma::cube H2(q, q, I, arma::fill::zeros);        //Separated Hessian for association
  arma::cube G2(q, 1, I, arma::fill::zeros);        //Separated gradient for association
  
  int idx_val = 0;
  for(int i = 0; i < I; i++) {
    List I_idx_sub = I_idx[i];                                                   //IDs of individuals/times in current cluster
    arma::colvec J_sub = as<arma::colvec>(rcpp_lengths(I_idx_sub));              //Number of individuals at each time point for current cluster
    double Ji = J_sub[0];                                                        //(Subsample) Individuals in cluster i (at any time point)
    arma::colvec Kij = K_full[i];                                                //(Full sample) Vector of K_{ij}
    double J_full = Kij[0];                                                      //(Full sample) Total number of individuals in the cluster
    double TJi = sum(J_sub);
    
    
    //Precomputed values which appear many times in computation; see paper
    double lambda3 = 1 + (J_full - 1)*(rho[0] - rho[1]) - rho[2];
    double lambda4 = 1 + (J_full - 1)*rho[0] + (T-1)*(J_full-1)*rho[1] + (T-1)*rho[2];
    
    c1 = 1/lambda1;
    c2 = (rho[2]-rho[1])/(lambda1*lambda2);
    c3 = (rho[0] - rho[1])/(lambda1*lambda3);
    c4 = (rho[2] - rho[1])*(rho[0]-rho[1])/(lambda1*lambda2*lambda3) + (rho[2]*rho[0] - rho[1])/(lambda2*lambda3*lambda4);
    
    IK = I_full*Kij[0]/(I*J_sub[0]);
    K_minus_1 = (Kij[0]-1)/(J_sub[0]-1);
    
    c1_tilde = c1*IK + c3*IK*(K_minus_1-1);
    c2_tilde = c2*IK + c4*IK*(K_minus_1-1);
    c3_tilde = c3*IK*K_minus_1;
    c4_tilde = c4*IK*K_minus_1;
    
    arma::mat Xtilde = design_mat.rows(idx_val, idx_val + TJi - 1); Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + TJi - 1)/phi;
    arma::colvec Etilde = std_resid.rows(idx_val, idx_val + TJi - 1);
    
    arma::rowvec Xtilde_sum = colSums(Xtilde);
    double Etilde_sum = sum(Etilde);
    
    //One portion of block-matrix multiplication involves matrix of 1's
    //Vectorize entire operation using special structure of matrix of 1's
    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, 1, arma::fill::zeros);
    int idx_val2 = 0;
    
    double resid_phi = 0;
    double resid2_0 = 0;
    double resid2_1 = 0;
    double resid2_2 = 0;
    
    for (unsigned int j = 0; j < J_sub.n_rows; j++) {
      arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
      arma::rowvec temp_vec_X = colSums(temp_mat_X);
      arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + J_sub[j] - 1);
      double temp_double_E = sum(temp_vec_E);
      
      Xtilde_sub = Xtilde_sub + temp_mat_X;
      Etilde_sub = Etilde_sub + temp_vec_E;
      
      
      G1.slice(i) += -c3_tilde*temp_vec_X.t()*temp_double_E;
      H1.slice(i) += -c3_tilde*temp_vec_X.t()*temp_vec_X;
      
      
      //Now for Hessians/gradients of dispersion
      resid_phi = IK*sum(square(temp_vec_E));
      G1_5(0,0,i) += phi*(resid_phi - IK*J_sub[j]);
      H1_5(0,0,i) += IK*J_sub[j];
      
      //Now for Hessians/gradients of association parameters
      resid2_0 = (pow(temp_double_E, 2) - resid_phi/IK)/2;         //All e_ij*e_ij' for individuals in cluster
      G2(0,0,i) += IK*K_minus_1*(resid2_0 - J_sub[j]*(J_sub[j]-1)/2*rho[0]);
      resid2_1 -= resid_phi/IK/2 + resid2_0;                          //-(e_i1 + ... + e_iJ_i)^2/2
      resid2_2 -= resid_phi/IK/2;                                     //-(e_i1^2 + ... + e_iJ_i^2)/2
      
      idx_val2 += J_sub[j];
    }
    
    
    
    //Add on remaining terms of summands (don't require 'for loop')
    G1.slice(i) += c1_tilde*Xtilde.t()*Etilde - c2_tilde*Xtilde_sub.t()*Etilde_sub + c4_tilde*Xtilde_sum.t()*Etilde_sum;
    H1.slice(i) += c1_tilde*Xtilde.t()*Xtilde - c2_tilde*Xtilde_sub.t()*Xtilde_sub + c4_tilde*Xtilde_sum.t()*Xtilde_sum;
    
    //Now for Hessians/gradients of dispersion
    G1_5(0,0,i) = IK*phi*sum(square(Etilde) - 1);
    H1_5(0,0,i) = IK*TJi;
    
    
    //Now for Hessians/gradients of ICC
    double diff_ind_same_period_length = sum(J_sub%(J_sub-1)/2);                                                    //Estimate rho0
    double same_ind_diff_period_length = T*(T-1)/2*J_sub[0];                                                        //Estimate rho2 (note the switching around!)
    double diff_ind_and_period_length = TJi*(TJi-1)/2 - diff_ind_same_period_length - same_ind_diff_period_length;  //Estimate rho1
    
    resid2_2 += sum(Etilde_sub % Etilde_sub)/2;
    resid2_1 += pow(sum(Etilde), 2)/2 - resid2_2;
    
    G2(1,0,i) += IK*K_minus_1*(resid2_1 - diff_ind_and_period_length*rho[1]);
    G2(2,0,i) += IK*(resid2_2 - same_ind_diff_period_length*rho[2]);
    
    H2.slice(i) = {{IK*K_minus_1*diff_ind_same_period_length, 0, 0},{0, IK*K_minus_1*diff_ind_and_period_length, 0},{0, 0, IK*same_ind_diff_period_length}};
    
    idx_val += TJi;
  }
  
  //Rcout << H1 << "\n";
  //Rcout << H1_5 << "\n";
  //Rcout << H2 << "\n";
  
  return List::create(Rcpp::Named("G1") = G1,
                      Rcpp::Named("H1") = H1,
                      Rcpp::Named("G1_5") = G1_5,
                      Rcpp::Named("H1_5") = H1_5,
                      Rcpp::Named("G2") = G2,
                      Rcpp::Named("H2") = H2
  );
}

//Mean of sandwich standard error
//Includes Fay and Graubard (2001) small sample correction
//[[Rcpp::export]]
arma::mat meat_computation(arma::cube G,
                           arma::cube H,
                           arma::mat Info,
                           string se_adjust) {
  double d = G.n_rows;
  int I = G.n_slices;
  arma::mat G_outersum(d, d, arma::fill::zeros);
  
  if (se_adjust == "unadjusted") {
    for(int i = 0; i < I; i++){
      G_outersum += G.slice(i)*(G.slice(i)).t();
    }
  } else if (se_adjust == "FG") {
    arma::mat Q(d, d, arma::fill::zeros);
    arma::colvec G_tilde(d, arma::fill::zeros);
    for(int i = 0; i < I; i++){
      Q = H.slice(i)*Info;
      G_tilde = pow(1 - mypmin(myrep({{0.85}}, {{d}}), diagvec(Q)), -0.5)%G.slice(i);
      G_outersum += G_tilde*G_tilde.t();
    }
  }
  
  return(G_outersum);
  
}

//Performs the iterations of deterministic Newton-Raphson
//[[Rcpp::export]]
List NewRaph(arma::colvec beta,
             double phi,
             arma::colvec rho,
             arma::colvec outcome,
             arma::mat design_mat,
             arma::mat clusterid,
             string family, //string link,
             string corstr,
             string se_adjust,
             double tol) {
  
  List I_idx = cluster_characteristics(clusterid);
  double err = 1;
  int iter = 0;
  int p = beta.n_rows;
  int q = rho.n_rows;
  int I = I_idx.size();
  
  //Hessian and gradient variables for linear coefficients
  arma::cube H1(p,p,I);
  arma::mat H1_sum(p,p);
  arma::mat Info1(p,p, arma::fill::eye);
  arma::cube G1(p,1,I);
  arma::colvec G1_sum(p);
  arma::colvec change1(p);
  
  //Hessian and gradient variables for dispersion parameters (variance of residuals in Gaussian family)
  arma::cube H1_5(1,1,I);
  arma::mat H1_5_sum(1,1);
  arma::mat Info1_5(1,1);
  arma::cube G1_5(1,1,I);
  arma::colvec G1_5_sum(1);
  arma::colvec change1_5(1);
  
  //Hessian and gradient variables for association parameters
  arma::cube H2(q,q,I);
  arma::mat H2_sum(q,q);
  arma::mat Info2(q,q);
  arma::cube G2(q,1,I);
  arma::colvec G2_sum(q);
  arma::colvec change2(p);
  
  //Needed for sandwich estimators later
  List BDE(3);
  
  
  List HG_output;
  if (family == "binomial") {
    if (corstr == "independence") {
      //Initial iteration for beta coefficients for stability
      HG_output = binomial_hier_solver(beta,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      //////////////////
      while (err > tol){
        iter += 1;
        HG_output = binomial_hier_solver(beta,
                                         rho,
                                         outcome,
                                         design_mat,
                                         Info1,
                                         I_idx);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1;
        err = 2*sum(abs(change1)/(abs(beta)+0.001 + abs(abs(beta)-0.001)));
        }
      binomial_hier_sandwich(beta, rho, outcome, design_mat, I_idx);
    } else if (corstr == "nested-exchangeable") {
      //Initial iteration for beta coefficients for stability
      HG_output = binomial_hier_solver(beta,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      //////////////////
      while (err > tol){
        iter += 1;
        HG_output = binomial_hier_solver(beta,
                                         rho,
                                         outcome,
                                         design_mat,
                                         Info1,
                                         I_idx);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1;
        err = 2*sum(abs(change1)/(abs(beta)+0.001 + abs(abs(beta)-0.001)));
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2;
        err += 2*sum(abs(change2)/(abs(rho)+0.001 + abs(abs(rho)-0.001)));
      }
      BDE = binomial_hier_sandwich(beta, rho, outcome, design_mat, I_idx);
    } else if (corstr == "block-exchangeable") {
      //Initial iteration for beta coefficients for stability
      HG_output = binomial_tscs_solver(beta,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      //////////////////
      while (err > tol){
        iter += 1;
        HG_output = binomial_tscs_solver(beta,
                                         rho,
                                         outcome,
                                         design_mat,
                                         Info1,
                                         I_idx);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1;
        err = 2*sum(abs(change1)/(abs(beta)+0.001 + abs(abs(beta)-0.001)));
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2;
        err += 2*sum(abs(change2)/(abs(rho)+0.001 + abs(abs(rho)-0.001)));
      }
      BDE = binomial_tscs_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    }
  } else if (family == "gaussian") {
    if (corstr == "independence") {
      //Initial iteration for beta coefficients for stability
      HG_output = gaussian_hier_solver(beta,
                                       phi,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      //////////////////
      while (err > tol){
        iter += 1;
        HG_output = gaussian_hier_solver(beta,
                                         phi,
                                         rho,
                                         outcome,
                                         design_mat,
                                         Info1,
                                         I_idx);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1;
        err = 2*sum(abs(change1)/(abs(beta)+0.001 + abs(abs(beta)-0.001)));
        }
      BDE = gaussian_hier_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    } else if (corstr == "nested-exchangeable") {
      //Initial iteration for beta coefficients for stability
      HG_output = gaussian_hier_solver(beta,
                                       phi, 
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      //////////////////
      while (err > tol){
        iter += 1;
        HG_output = gaussian_hier_solver(beta,
                                         phi,
                                         rho,
                                         outcome,
                                         design_mat,
                                         Info1,
                                         I_idx);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1;
        err = 2*sum(abs(change1)/(abs(beta)+0.001 + abs(abs(beta)-0.001)));
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2;
        err += 2*sum(abs(change2)/(abs(rho)+0.001 + abs(abs(rho)-0.001)));
      }
      BDE = gaussian_hier_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    } else if (corstr == "block-exchangeable") {
      //Initial iteration for beta coefficients for stability
      HG_output = gaussian_tscs_solver(beta,
                                       phi,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      //////////////////
      while (err > tol){
        iter += 1;
        HG_output = gaussian_tscs_solver(beta,
                                         phi,
                                         rho,
                                         outcome,
                                         design_mat,
                                         Info1,
                                         I_idx);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1;
        err = 2*sum(abs(change1)/(abs(beta)+0.001 + abs(abs(beta)-0.001)));
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2;
        err += 2*sum(abs(change2)/(abs(rho)+0.001 + abs(abs(rho)-0.001)));
      }
      BDE = gaussian_tscs_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    }
  }
  
  if (family == "binomial") {
    arma::cube H(p+q, p+q, I, arma::fill::zeros);
    H.tube(0, 0, p-1, p-1) = H1;
    H.tube(p, 0, p+q-1, p-1) = as<arma::cube>( BDE["D"]);
    H.tube(p, p, p+q-1, p+q-1) = H2;
    
    arma::cube G(p+q, 1, I, arma::fill::zeros);
    G.tube(0, 0, p-1, 0) = G1;
    G.tube(p, 0, p+q-1, 0) = G2;
    
    if (corstr == "independence") {
      arma::mat H_sum = sum(H.tube(0, 0, p-1, p-1), 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G.tube(0, 0, p-1, 0), H.tube(0, 0, p-1, p-1), Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t(),
                          Rcpp::Named("iter") = iter
      );
    } else {
      arma::mat H_sum = sum(H, 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G, H, Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t(),
                          Rcpp::Named("iter") = iter
      );
    }
    
  } else {
    arma::cube H(p+q+1, p+q+1, I, arma::fill::zeros);
    H.tube(0, 0, p-1, p-1) = H1;
    H.tube(p, 0, p, p-1) = as<arma::cube>( BDE["B"]);
    H.tube(p, p, p, p) = H1_5;
    H.tube(p+1, 0, p+q, p-1) = as<arma::cube>( BDE["D"]);
    H.tube(p+1, p, p+q, p) = as<arma::cube>( BDE["E"]);
    H.tube(p+1, p+1, p+q, p+q) = H2;
    
    arma::cube G(p+q+1, 1, I, arma::fill::zeros);
    G.tube(0, 0, p-1, 0) = G1;
    G.tube(p, 0, p, 0) = G1_5;
    G.tube(p+1, 0, p+q, 0) = G2;
    
    if (corstr == "independence") {
      arma::mat H_sum = sum(H.tube(0, 0, p, p), 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G.tube(0, 0, p, 0), H.tube(0, 0, p, p), Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("phi") = phi,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info1_5") = Info1_5,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t(),
                          Rcpp::Named("iter") = iter
      );
    } else {
      arma::mat H_sum = sum(H, 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G, H, Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("phi") = phi,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info1_5") = Info1_5,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t(),
                          Rcpp::Named("iter") = iter
      );
    }
    
  }
}

//Performs the iterations of stochastic Newton-Raphson
//[[Rcpp::export]]
List StochNewRaph(arma::colvec beta,
                  double phi,
                  arma::colvec rho,
                  arma::colvec outcome,
                  arma::mat design_mat,
                  arma::mat clusterid,
                  string family, //string link,
                  string corstr,
                  string se_adjust,
                  arma::colvec batch_size,
                  int burnin,
                  int avgiter) {
  
  List I_idx = cluster_characteristics(clusterid);
  int p = beta.n_rows;
  int q = rho.n_rows;
  int I = I_idx.size();
  
  //Hessian and gradient variables for linear coefficients
  arma::cube H1(p,p,I);
  arma::mat H1_sum(p,p);
  arma::mat Info1(p,p);
  arma::cube G1(p,1,I);
  arma::colvec G1_sum(p);
  arma::colvec change1(p);
  
  //Hessian and gradient variables for dispersion parameters (variance of residuals in Gaussian family)
  arma::cube H1_5(1,1,I);
  arma::mat H1_5_sum(1,1);
  arma::mat Info1_5(1,1);
  arma::cube G1_5(1,1,I);
  arma::colvec G1_5_sum(1);
  arma::colvec change1_5(1);
  
  //Hessian and gradient variables for association parameters
  arma::cube H2(q,q,I);
  arma::mat H2_sum(q,q);
  arma::mat Info2(q,q);
  arma::cube G2(q,1,I);
  arma::colvec G2_sum(q);
  arma::colvec change2(p);
  
  //Needed for sandwich estimators later
  List BDE(3);
  
  List HG_output;
  if (family == "binomial") {
    if (corstr == "independence") {
      //Step 1. Initial iteration for stability
      List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
      HG_output = stochastic_binomial_hier_solver(beta,
                                                  rho,
                                                  outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                  design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                  subsample[0],                                    //Indices for subsample
                                                           subsample[1],                                    //No. first-order clusters in subsample
                                                                    subsample[2],                                    //No. second-order clusters in subsample
                                                                             subsample[3]);                                   //No. third-order clusters in subsample
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Step 2. Burn-in iterations
      for(int iter = 1; iter < burnin; iter++){                                        //Stochastic iterations
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_binomial_hier_solver(beta, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                    subsample[0],                                    //Indices for subsample
                                                             subsample[1],                                    //No. first-order clusters in subsample
                                                                      subsample[2],                                    //No. second-order clusters in subsample
                                                                               subsample[3]);                                   //No. third-order clusters in subsample
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
      }
      
      //Step 3. Averaging iterations
      arma::colvec beta_avg(p, arma::fill::zeros);
      arma::colvec rho_avg(q, arma::fill::zeros);
      
      for(int iter = burnin; iter < burnin + avgiter; iter++){                         //Averaged SGD, including for Hessians and gradients
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_binomial_hier_solver(beta, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                    subsample[0],
                                                             subsample[1],
                                                                      subsample[2],
                                                                               subsample[3]);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        beta_avg += beta;
      }
      beta = beta_avg/avgiter;
      rho = rho_avg/avgiter;
      
      //Step 4. One final deterministic iteration for sandwich estimator (meat of sandwich under stochastic is too difficult to derive)//////////////////////////
      HG_output = binomial_hier_solver(beta,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Sandwich (must be deterministic)
      BDE = binomial_hier_sandwich(beta, rho, outcome, design_mat, I_idx);
    } else if (corstr == "nested-exchangeable") {
      //Step 1. Initial iteration for stability
      List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
      HG_output = stochastic_binomial_hier_solver(beta,
                                                  rho,
                                                  outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                  design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                  subsample[0],                                    //Indices for subsample
                                                           subsample[1],                                    //No. first-order clusters in subsample
                                                                    subsample[2],                                    //No. second-order clusters in subsample
                                                                             subsample[3]);                                   //No. third-order clusters in subsample
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Step 2. Burn-in iterations
      for(int iter = 1; iter < burnin; iter++){                                        //Stochastic iterations
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_binomial_hier_solver(beta, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                    subsample[0],                                    //Indices for subsample
                                                             subsample[1],                                    //No. first-order clusters in subsample
                                                                      subsample[2],                                    //No. second-order clusters in subsample
                                                                               subsample[3]);                                   //No. third-order clusters in subsample
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        H2 = as<arma::cube>(HG_output["H2"]);
        
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
      }
      
      //Step 3. Averaging iterations
      arma::colvec beta_avg(p, arma::fill::zeros);
      arma::colvec rho_avg(q, arma::fill::zeros);
      
      for(int iter = burnin; iter < burnin + avgiter; iter++){                         //Averaged SGD, including for Hessians and gradients
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_binomial_hier_solver(beta, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                    subsample[0],
                                                             subsample[1],
                                                                      subsample[2],
                                                                               subsample[3]);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        beta_avg += beta;
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
        rho_avg += rho;
      }
      beta = beta_avg/avgiter;
      rho = rho_avg/avgiter;
      
      //Step 4. One final deterministic iteration for sandwich estimator (meat of sandwich under stochastic is too difficult to derive)//////////////////////////
      HG_output = binomial_hier_solver(beta,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      H2 = as<arma::cube>(HG_output["H2"]);
      H2_sum = sum(H2, 2);
      Info2 = H2_sum.i();
      G2 = as<arma::cube>(HG_output["G2"]);
      G2_sum = sum(G2, 2);
      change2 = Info2*G2_sum;
      rho += change2;
      
      //Sandwich (must be deterministic)
      BDE = binomial_hier_sandwich(beta, rho, outcome, design_mat, I_idx);
    } else if (corstr == "block-exchangeable") {
      //Step 1. Initial iteration for stability
      List subsample = hierarchical_sampling(I_idx, batch_size, false, true);
      HG_output = stochastic_binomial_tscs_solver(beta, rho,
                                                  outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                  design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                  subsample[0],
                                                           subsample[1],
                                                                    subsample[2],
                                                                             subsample[3]);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Step 2. Burn-in iterations
      for(int iter = 1; iter < burnin; iter++){                                        //Stochastic iterations
        List subsample = hierarchical_sampling(I_idx, batch_size, false, true);
        HG_output = stochastic_binomial_tscs_solver(beta,
                                                    rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                    subsample[0],                                    //Indices for subsample
                                                             subsample[1],                                    //No. first-order clusters in subsample
                                                                      subsample[2],                                    //No. second-order clusters in subsample
                                                                               subsample[3]);     
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51); 											//Iterations with learning rates
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
      }
      
      //Step 3. Averaging iterations
      arma::colvec beta_avg(p, arma::fill::zeros);
      arma::colvec rho_avg(q, arma::fill::zeros);
      
      for(int iter = burnin; iter < burnin + avgiter; iter++){                         //Averaged SGD, including for Hessians and gradients
        List subsample = hierarchical_sampling(I_idx, batch_size, false, true);
        HG_output = stochastic_binomial_tscs_solver(beta,
                                                    rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                    subsample[0],
                                                             subsample[1],
                                                                      subsample[2],
                                                                               subsample[3]);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        beta_avg += beta;
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
        rho_avg += rho;
      }
      beta = beta_avg/avgiter;
      rho = rho_avg/avgiter;
      
      //Step 4. One final deterministic iteration for sandwich estimator (meat of sandwich under stochastic is too difficult to derive)//////////////////////////
      HG_output = binomial_tscs_solver(beta,
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      H2 = as<arma::cube>(HG_output["H2"]);
      H2_sum = sum(H2, 2);
      Info2 = H2_sum.i();
      G2 = as<arma::cube>(HG_output["G2"]);
      G2_sum = sum(G2, 2);
      change2 = Info2*G2_sum;
      rho += change2;
      
      //Sandwich (must be deterministic)
      BDE = binomial_tscs_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    }
  } else if (family == "gaussian") {
    if (corstr == "independence") {
      //Step 1. Initial iteration for stability
      List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
      HG_output = stochastic_gaussian_hier_solver(beta,
                                                  phi,
                                                  rho,
                                                  outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                  design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                  subsample[0],                                    //Indices for subsample
                                                           subsample[1],                                    //No. first-order clusters in subsample
                                                                    subsample[2],                                    //No. second-order clusters in subsample
                                                                             subsample[3]);                                   //No. third-order clusters in subsample
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Step 2. Burn-in iterations
      for(int iter = 1; iter < burnin; iter++){                                        //Stochastic iterations
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_gaussian_hier_solver(beta, phi, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                    subsample[0],                                    //Indices for subsample
                                                             subsample[1],                                    //No. first-order clusters in subsample
                                                                      subsample[2],                                    //No. second-order clusters in subsample
                                                                               subsample[3]);                                   //No. third-order clusters in subsample
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
      }
      
      //Step 3. Averaging iterations
      arma::colvec beta_avg(p, arma::fill::zeros);
      arma::colvec rho_avg(q, arma::fill::zeros);
      
      for(int iter = burnin; iter < burnin + avgiter; iter++){                         //Averaged SGD, including for Hessians and gradients
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_gaussian_hier_solver(beta, phi, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                    subsample[0],
                                                             subsample[1],
                                                                      subsample[2],
                                                                               subsample[3]);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        beta_avg += beta;
      }
      beta = beta_avg/avgiter;
      rho = rho_avg/avgiter;
      
      //Step 4. One final deterministic iteration for sandwich estimator (meat of sandwich under stochastic is too difficult to derive)//////////////////////////
      HG_output = gaussian_hier_solver(beta,
                                       phi, 
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Sandwich (must be deterministic)
      BDE = gaussian_hier_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    } else if (corstr == "nested-exchangeable") {
      //Step 1. Initial iteration for stability
      List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
      HG_output = stochastic_gaussian_hier_solver(beta,
                                                  phi, 
                                                  rho,
                                                  outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                  design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                  subsample[0],                                    //Indices for subsample
                                                           subsample[1],                                    //No. first-order clusters in subsample
                                                                    subsample[2],                                    //No. second-order clusters in subsample
                                                                             subsample[3]);                                   //No. third-order clusters in subsample
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Step 2. Burn-in iterations
      for(int iter = 1; iter < burnin; iter++){                                        //Stochastic iterations
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_gaussian_hier_solver(beta, phi, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                    subsample[0],                                    //Indices for subsample
                                                             subsample[1],                                    //No. first-order clusters in subsample
                                                                      subsample[2],                                    //No. second-order clusters in subsample
                                                                               subsample[3]);                                   //No. third-order clusters in subsample
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        H2 = as<arma::cube>(HG_output["H2"]);
        
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
      }
      
      //Step 3. Averaging iterations
      arma::colvec beta_avg(p, arma::fill::zeros);
      arma::colvec rho_avg(q, arma::fill::zeros);
      
      for(int iter = burnin; iter < burnin + avgiter; iter++){                         //Averaged SGD, including for Hessians and gradients
        List subsample = hierarchical_sampling(I_idx, batch_size, false, false);
        HG_output = stochastic_gaussian_hier_solver(beta, phi, rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                    subsample[0],
                                                             subsample[1],
                                                                      subsample[2],
                                                                               subsample[3]);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        beta_avg += beta;
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
        rho_avg += rho;
      }
      beta = beta_avg/avgiter;
      rho = rho_avg/avgiter;
      
      //Step 4. One final deterministic iteration for sandwich estimator (meat of sandwich under stochastic is too difficult to derive)//////////////////////////
      HG_output = gaussian_hier_solver(beta,
                                       phi, 
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      H2 = as<arma::cube>(HG_output["H2"]);
      H2_sum = sum(H2, 2);
      Info2 = H2_sum.i();
      G2 = as<arma::cube>(HG_output["G2"]);
      G2_sum = sum(G2, 2);
      change2 = Info2*G2_sum;
      rho += change2;
      
      //Sandwich (must be deterministic)
      BDE = gaussian_hier_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    } else if (corstr == "block-exchangeable") {
      //Step 1. Initial iteration for stability
      List subsample = hierarchical_sampling(I_idx, batch_size, false, true);
      HG_output = stochastic_gaussian_tscs_solver(beta, phi, rho,
                                                  outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                  design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                  subsample[0],
                                                           subsample[1],
                                                                    subsample[2],
                                                                             subsample[3]);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      //Step 2. Burn-in iterations
      for(int iter = 1; iter < burnin; iter++){                                        //Stochastic iterations
        List subsample = hierarchical_sampling(I_idx, batch_size, false, true);
        HG_output = stochastic_gaussian_tscs_solver(beta,
                                                    phi, 
                                                    rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),    //Subsample of outcomes
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1), //Subsample of covariates
                                                    subsample[0],                                    //Indices for subsample
                                                             subsample[1],                                    //No. first-order clusters in subsample
                                                                      subsample[2],                                    //No. second-order clusters in subsample
                                                                               subsample[3]);     
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51); 											//Iterations with learning rates
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
      }
      
      //Step 3. Averaging iterations
      arma::colvec beta_avg(p, arma::fill::zeros);
      arma::colvec rho_avg(q, arma::fill::zeros);
      
      for(int iter = burnin; iter < burnin + avgiter; iter++){                         //Averaged SGD, including for Hessians and gradients
        List subsample = hierarchical_sampling(I_idx, batch_size, false, true);
        HG_output = stochastic_gaussian_tscs_solver(beta,
                                                    phi, 
                                                    rho,
                                                    outcome.rows(as<arma::uvec>(subsample[4])-1),
                                                    design_mat.rows(as<arma::uvec>(subsample[4])-1),
                                                    subsample[0],
                                                             subsample[1],
                                                                      subsample[2],
                                                                               subsample[3]);
        
        H1 = as<arma::cube>(HG_output["H1"]);
        H1_sum = sum(H1, 2);
        Info1 = H1_sum.i();
        
        G1 = as<arma::cube>(HG_output["G1"]);
        G1_sum = sum(G1, 2);
        change1 = Info1*G1_sum;
        beta += change1/pow(iter+1,0.51);                                               //Iterations with learning rates
        beta_avg += beta;
        
        H2 = as<arma::cube>(HG_output["H2"]);
        H2_sum = sum(H2, 2);
        Info2 = H2_sum.i();
        
        G2 = as<arma::cube>(HG_output["G2"]);
        G2_sum = sum(G2, 2);
        change2 = Info2*G2_sum;
        rho += change2/pow(iter+1,0.51);                                             //Iterations with learning rates
        rho_avg += rho;
      }
      beta = beta_avg/avgiter;
      rho = rho_avg/avgiter;
      
      //Step 4. One final deterministic iteration for sandwich estimator (meat of sandwich under stochastic is too difficult to derive)//////////////////////////
      HG_output = gaussian_tscs_solver(beta,
                                       phi, 
                                       rho,
                                       outcome,
                                       design_mat,
                                       Info1,
                                       I_idx);
      
      H1 = as<arma::cube>(HG_output["H1"]);
      H1_sum = sum(H1, 2);
      Info1 = H1_sum.i();
      G1 = as<arma::cube>(HG_output["G1"]);
      G1_sum = sum(G1, 2);
      change1 = Info1*G1_sum;
      beta += change1;
      
      H2 = as<arma::cube>(HG_output["H2"]);
      H2_sum = sum(H2, 2);
      Info2 = H2_sum.i();
      G2 = as<arma::cube>(HG_output["G2"]);
      G2_sum = sum(G2, 2);
      change2 = Info2*G2_sum;
      rho += change2;
      
      //Sandwich (must be deterministic)
      BDE = gaussian_tscs_sandwich(beta, phi, rho, outcome, design_mat, I_idx);
    }
  }
  
  if (family == "binomial") {
    arma::cube H(p+q, p+q, I, arma::fill::zeros);
    H.tube(0, 0, p-1, p-1) = H1;
    H.tube(p, 0, p+q-1, p-1) = as<arma::cube>( BDE["D"]);
    H.tube(p, p, p+q-1, p+q-1) = H2;
    
    arma::cube G(p+q, 1, I, arma::fill::zeros);
    G.tube(0, 0, p-1, 0) = G1;
    G.tube(p, 0, p+q-1, 0) = G2;
    
    if (corstr == "independence") {
      arma::mat H_sum = sum(H.tube(0, 0, p-1, p-1), 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G.tube(0, 0, p-1, 0), H.tube(0, 0, p-1, p-1), Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t());
    } else {
      arma::mat H_sum = sum(H, 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G, H, Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t());
    }
    
  } else {
    arma::cube H(p+q+1, p+q+1, I, arma::fill::zeros);
    H.tube(0, 0, p-1, p-1) = H1;
    H.tube(p, 0, p, p-1) = as<arma::cube>( BDE["B"]);
    H.tube(p, p, p, p) = H1_5;
    H.tube(p+1, 0, p+q, p-1) = as<arma::cube>( BDE["D"]);
    H.tube(p+1, p, p+q, p) = as<arma::cube>( BDE["E"]);
    H.tube(p+1, p+1, p+q, p+q) = H2;
    
    arma::cube G(p+q+1, 1, I, arma::fill::zeros);
    G.tube(0, 0, p-1, 0) = G1;
    G.tube(p, 0, p, 0) = G1_5;
    G.tube(p+1, 0, p+q, 0) = G2;
    
    if (corstr == "independence") {
      arma::mat H_sum = sum(H.tube(0, 0, p, p), 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G.tube(0, 0, p, 0), H.tube(0, 0, p, p), Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("phi") = phi,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info1_5") = Info1_5,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t());
    } else {
      arma::mat H_sum = sum(H, 2);
      arma::mat Info = H_sum.i();
      arma::mat G_outersum = meat_computation(G, H, Info, se_adjust);
      return List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("phi") = phi,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("Info1") = Info1,
                          Rcpp::Named("Info1_5") = Info1_5,
                          Rcpp::Named("Info2") = Info2,
                          Rcpp::Named("Info") = Info,
                          Rcpp::Named("G_outersum") = G_outersum,
                          Rcpp::Named("var_sandwich") = Info*G_outersum*Info.t());
    }
    
  }
}
