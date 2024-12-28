// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Seq
arma::uvec Seq(int first, int last);
RcppExport SEXP _networkGEE_Seq(SEXP firstSEXP, SEXP lastSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type first(firstSEXP);
    Rcpp::traits::input_parameter< int >::type last(lastSEXP);
    rcpp_result_gen = Rcpp::wrap(Seq(first, last));
    return rcpp_result_gen;
END_RCPP
}
// mypmin
arma::colvec mypmin(arma::colvec vec1, arma::colvec vec2);
RcppExport SEXP _networkGEE_mypmin(SEXP vec1SEXP, SEXP vec2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type vec1(vec1SEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type vec2(vec2SEXP);
    rcpp_result_gen = Rcpp::wrap(mypmin(vec1, vec2));
    return rcpp_result_gen;
END_RCPP
}
// myrep
arma::colvec myrep(arma::colvec x, arma::colvec y);
RcppExport SEXP _networkGEE_myrep(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(myrep(x, y));
    return rcpp_result_gen;
END_RCPP
}
// myrep_uvec
arma::uvec myrep_uvec(arma::uvec x, arma::uvec y);
RcppExport SEXP _networkGEE_myrep_uvec(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(myrep_uvec(x, y));
    return rcpp_result_gen;
END_RCPP
}
// colSums
arma::rowvec colSums(const arma::mat& X);
RcppExport SEXP _networkGEE_colSums(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(colSums(X));
    return rcpp_result_gen;
END_RCPP
}
// mycombnCpp
arma::umat mycombnCpp(double n, double k);
RcppExport SEXP _networkGEE_mycombnCpp(SEXP nSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(mycombnCpp(n, k));
    return rcpp_result_gen;
END_RCPP
}
// pairwiseprod
arma::colvec pairwiseprod(arma::colvec M);
RcppExport SEXP _networkGEE_pairwiseprod(SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(pairwiseprod(M));
    return rcpp_result_gen;
END_RCPP
}
// pairwiseprod2
arma::mat pairwiseprod2(arma::mat M, arma::colvec N, int dir);
RcppExport SEXP _networkGEE_pairwiseprod2(SEXP MSEXP, SEXP NSEXP, SEXP dirSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type M(MSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type dir(dirSEXP);
    rcpp_result_gen = Rcpp::wrap(pairwiseprod2(M, N, dir));
    return rcpp_result_gen;
END_RCPP
}
// arma_setdiff
arma::uvec arma_setdiff(arma::uvec& x, arma::uvec& y);
RcppExport SEXP _networkGEE_arma_setdiff(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(arma_setdiff(x, y));
    return rcpp_result_gen;
END_RCPP
}
// index_fun2
arma::uvec index_fun2(arma::uvec J);
RcppExport SEXP _networkGEE_index_fun2(SEXP JSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type J(JSEXP);
    rcpp_result_gen = Rcpp::wrap(index_fun2(J));
    return rcpp_result_gen;
END_RCPP
}
// index_fun1
arma::uvec index_fun1(arma::uvec J);
RcppExport SEXP _networkGEE_index_fun1(SEXP JSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec >::type J(JSEXP);
    rcpp_result_gen = Rcpp::wrap(index_fun1(J));
    return rcpp_result_gen;
END_RCPP
}
// combine
NumericVector combine(const List& list);
RcppExport SEXP _networkGEE_combine(SEXP listSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List& >::type list(listSEXP);
    rcpp_result_gen = Rcpp::wrap(combine(list));
    return rcpp_result_gen;
END_RCPP
}
// cluster_characteristics
List cluster_characteristics(arma::mat clusterid);
RcppExport SEXP _networkGEE_cluster_characteristics(SEXP clusteridSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type clusterid(clusteridSEXP);
    rcpp_result_gen = Rcpp::wrap(cluster_characteristics(clusterid));
    return rcpp_result_gen;
END_RCPP
}
// hierarchical_sampling
List hierarchical_sampling(List I_idx, arma::colvec batch, bool replace, bool tscs);
RcppExport SEXP _networkGEE_hierarchical_sampling(SEXP I_idxSEXP, SEXP batchSEXP, SEXP replaceSEXP, SEXP tscsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type batch(batchSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< bool >::type tscs(tscsSEXP);
    rcpp_result_gen = Rcpp::wrap(hierarchical_sampling(I_idx, batch, replace, tscs));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_lengths_recursive
Rcpp::List rcpp_lengths_recursive(const Rcpp::List& lst);
RcppExport SEXP _networkGEE_rcpp_lengths_recursive(SEXP lstSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type lst(lstSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_lengths_recursive(lst));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_lengths
Rcpp::IntegerVector rcpp_lengths(const Rcpp::List& lst);
RcppExport SEXP _networkGEE_rcpp_lengths(SEXP lstSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type lst(lstSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_lengths(lst));
    return rcpp_result_gen;
END_RCPP
}
// binomial_hier_solver
List binomial_hier_solver(arma::colvec beta, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, arma::mat Info1, List I_idx);
RcppExport SEXP _networkGEE_binomial_hier_solver(SEXP betaSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP Info1SEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Info1(Info1SEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(binomial_hier_solver(beta, rho, outcome, design_mat, Info1, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// gaussian_hier_solver
List gaussian_hier_solver(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, arma::mat Info1, List I_idx);
RcppExport SEXP _networkGEE_gaussian_hier_solver(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP Info1SEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Info1(Info1SEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(gaussian_hier_solver(beta, phi, rho, outcome, design_mat, Info1, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// binomial_hier_sandwich
List binomial_hier_sandwich(arma::colvec beta, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx);
RcppExport SEXP _networkGEE_binomial_hier_sandwich(SEXP betaSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(binomial_hier_sandwich(beta, rho, outcome, design_mat, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// gaussian_hier_sandwich
List gaussian_hier_sandwich(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx);
RcppExport SEXP _networkGEE_gaussian_hier_sandwich(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(gaussian_hier_sandwich(beta, phi, rho, outcome, design_mat, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// binomial_tscs_solver
List binomial_tscs_solver(arma::colvec beta, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, arma::mat Info1, List I_idx);
RcppExport SEXP _networkGEE_binomial_tscs_solver(SEXP betaSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP Info1SEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Info1(Info1SEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(binomial_tscs_solver(beta, rho, outcome, design_mat, Info1, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// gaussian_tscs_solver
List gaussian_tscs_solver(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, arma::mat Info1, List I_idx);
RcppExport SEXP _networkGEE_gaussian_tscs_solver(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP Info1SEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Info1(Info1SEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(gaussian_tscs_solver(beta, phi, rho, outcome, design_mat, Info1, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// binomial_tscs_sandwich
List binomial_tscs_sandwich(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx);
RcppExport SEXP _networkGEE_binomial_tscs_sandwich(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(binomial_tscs_sandwich(beta, phi, rho, outcome, design_mat, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// gaussian_tscs_sandwich
List gaussian_tscs_sandwich(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx);
RcppExport SEXP _networkGEE_gaussian_tscs_sandwich(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(gaussian_tscs_sandwich(beta, phi, rho, outcome, design_mat, I_idx));
    return rcpp_result_gen;
END_RCPP
}
// stochastic_binomial_hier_solver
List stochastic_binomial_hier_solver(arma::colvec beta, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx, int I_full, arma::colvec J_full, List K_full);
RcppExport SEXP _networkGEE_stochastic_binomial_hier_solver(SEXP betaSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP, SEXP I_fullSEXP, SEXP J_fullSEXP, SEXP K_fullSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    Rcpp::traits::input_parameter< int >::type I_full(I_fullSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type J_full(J_fullSEXP);
    Rcpp::traits::input_parameter< List >::type K_full(K_fullSEXP);
    rcpp_result_gen = Rcpp::wrap(stochastic_binomial_hier_solver(beta, rho, outcome, design_mat, I_idx, I_full, J_full, K_full));
    return rcpp_result_gen;
END_RCPP
}
// stochastic_gaussian_hier_solver
List stochastic_gaussian_hier_solver(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx, int I_full, arma::colvec J_full, List K_full);
RcppExport SEXP _networkGEE_stochastic_gaussian_hier_solver(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP, SEXP I_fullSEXP, SEXP J_fullSEXP, SEXP K_fullSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    Rcpp::traits::input_parameter< int >::type I_full(I_fullSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type J_full(J_fullSEXP);
    Rcpp::traits::input_parameter< List >::type K_full(K_fullSEXP);
    rcpp_result_gen = Rcpp::wrap(stochastic_gaussian_hier_solver(beta, phi, rho, outcome, design_mat, I_idx, I_full, J_full, K_full));
    return rcpp_result_gen;
END_RCPP
}
// stochastic_binomial_tscs_solver
List stochastic_binomial_tscs_solver(arma::colvec beta, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx, int I_full, arma::colvec J_full, List K_full);
RcppExport SEXP _networkGEE_stochastic_binomial_tscs_solver(SEXP betaSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP, SEXP I_fullSEXP, SEXP J_fullSEXP, SEXP K_fullSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    Rcpp::traits::input_parameter< int >::type I_full(I_fullSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type J_full(J_fullSEXP);
    Rcpp::traits::input_parameter< List >::type K_full(K_fullSEXP);
    rcpp_result_gen = Rcpp::wrap(stochastic_binomial_tscs_solver(beta, rho, outcome, design_mat, I_idx, I_full, J_full, K_full));
    return rcpp_result_gen;
END_RCPP
}
// stochastic_gaussian_tscs_solver
List stochastic_gaussian_tscs_solver(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, List I_idx, int I_full, arma::colvec J_full, List K_full);
RcppExport SEXP _networkGEE_stochastic_gaussian_tscs_solver(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP I_idxSEXP, SEXP I_fullSEXP, SEXP J_fullSEXP, SEXP K_fullSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< List >::type I_idx(I_idxSEXP);
    Rcpp::traits::input_parameter< int >::type I_full(I_fullSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type J_full(J_fullSEXP);
    Rcpp::traits::input_parameter< List >::type K_full(K_fullSEXP);
    rcpp_result_gen = Rcpp::wrap(stochastic_gaussian_tscs_solver(beta, phi, rho, outcome, design_mat, I_idx, I_full, J_full, K_full));
    return rcpp_result_gen;
END_RCPP
}
// meat_computation
arma::mat meat_computation(arma::cube G, arma::cube H, arma::mat Info, std::string se_adjust);
RcppExport SEXP _networkGEE_meat_computation(SEXP GSEXP, SEXP HSEXP, SEXP InfoSEXP, SEXP se_adjustSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type G(GSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type H(HSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Info(InfoSEXP);
    Rcpp::traits::input_parameter< std::string >::type se_adjust(se_adjustSEXP);
    rcpp_result_gen = Rcpp::wrap(meat_computation(G, H, Info, se_adjust));
    return rcpp_result_gen;
END_RCPP
}
// NewRaph
List NewRaph(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, arma::mat clusterid, std::string family, std::string corstr, std::string design, std::string se_adjust, double tol);
RcppExport SEXP _networkGEE_NewRaph(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP clusteridSEXP, SEXP familySEXP, SEXP corstrSEXP, SEXP designSEXP, SEXP se_adjustSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type clusterid(clusteridSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type corstr(corstrSEXP);
    Rcpp::traits::input_parameter< std::string >::type design(designSEXP);
    Rcpp::traits::input_parameter< std::string >::type se_adjust(se_adjustSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(NewRaph(beta, phi, rho, outcome, design_mat, clusterid, family, corstr, design, se_adjust, tol));
    return rcpp_result_gen;
END_RCPP
}
// StochNewRaph
List StochNewRaph(arma::colvec beta, double phi, arma::colvec rho, arma::colvec outcome, arma::mat design_mat, arma::mat clusterid, std::string family, std::string corstr, std::string design, std::string se_adjust, arma::colvec batch_size, int burnin, int avgiter);
RcppExport SEXP _networkGEE_StochNewRaph(SEXP betaSEXP, SEXP phiSEXP, SEXP rhoSEXP, SEXP outcomeSEXP, SEXP design_matSEXP, SEXP clusteridSEXP, SEXP familySEXP, SEXP corstrSEXP, SEXP designSEXP, SEXP se_adjustSEXP, SEXP batch_sizeSEXP, SEXP burninSEXP, SEXP avgiterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type outcome(outcomeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type clusterid(clusteridSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type corstr(corstrSEXP);
    Rcpp::traits::input_parameter< std::string >::type design(designSEXP);
    Rcpp::traits::input_parameter< std::string >::type se_adjust(se_adjustSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type avgiter(avgiterSEXP);
    rcpp_result_gen = Rcpp::wrap(StochNewRaph(beta, phi, rho, outcome, design_mat, clusterid, family, corstr, design, se_adjust, batch_size, burnin, avgiter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_networkGEE_Seq", (DL_FUNC) &_networkGEE_Seq, 2},
    {"_networkGEE_mypmin", (DL_FUNC) &_networkGEE_mypmin, 2},
    {"_networkGEE_myrep", (DL_FUNC) &_networkGEE_myrep, 2},
    {"_networkGEE_myrep_uvec", (DL_FUNC) &_networkGEE_myrep_uvec, 2},
    {"_networkGEE_colSums", (DL_FUNC) &_networkGEE_colSums, 1},
    {"_networkGEE_mycombnCpp", (DL_FUNC) &_networkGEE_mycombnCpp, 2},
    {"_networkGEE_pairwiseprod", (DL_FUNC) &_networkGEE_pairwiseprod, 1},
    {"_networkGEE_pairwiseprod2", (DL_FUNC) &_networkGEE_pairwiseprod2, 3},
    {"_networkGEE_arma_setdiff", (DL_FUNC) &_networkGEE_arma_setdiff, 2},
    {"_networkGEE_index_fun2", (DL_FUNC) &_networkGEE_index_fun2, 1},
    {"_networkGEE_index_fun1", (DL_FUNC) &_networkGEE_index_fun1, 1},
    {"_networkGEE_combine", (DL_FUNC) &_networkGEE_combine, 1},
    {"_networkGEE_cluster_characteristics", (DL_FUNC) &_networkGEE_cluster_characteristics, 1},
    {"_networkGEE_hierarchical_sampling", (DL_FUNC) &_networkGEE_hierarchical_sampling, 4},
    {"_networkGEE_rcpp_lengths_recursive", (DL_FUNC) &_networkGEE_rcpp_lengths_recursive, 1},
    {"_networkGEE_rcpp_lengths", (DL_FUNC) &_networkGEE_rcpp_lengths, 1},
    {"_networkGEE_binomial_hier_solver", (DL_FUNC) &_networkGEE_binomial_hier_solver, 6},
    {"_networkGEE_gaussian_hier_solver", (DL_FUNC) &_networkGEE_gaussian_hier_solver, 7},
    {"_networkGEE_binomial_hier_sandwich", (DL_FUNC) &_networkGEE_binomial_hier_sandwich, 5},
    {"_networkGEE_gaussian_hier_sandwich", (DL_FUNC) &_networkGEE_gaussian_hier_sandwich, 6},
    {"_networkGEE_binomial_tscs_solver", (DL_FUNC) &_networkGEE_binomial_tscs_solver, 6},
    {"_networkGEE_gaussian_tscs_solver", (DL_FUNC) &_networkGEE_gaussian_tscs_solver, 7},
    {"_networkGEE_binomial_tscs_sandwich", (DL_FUNC) &_networkGEE_binomial_tscs_sandwich, 6},
    {"_networkGEE_gaussian_tscs_sandwich", (DL_FUNC) &_networkGEE_gaussian_tscs_sandwich, 6},
    {"_networkGEE_stochastic_binomial_hier_solver", (DL_FUNC) &_networkGEE_stochastic_binomial_hier_solver, 8},
    {"_networkGEE_stochastic_gaussian_hier_solver", (DL_FUNC) &_networkGEE_stochastic_gaussian_hier_solver, 9},
    {"_networkGEE_stochastic_binomial_tscs_solver", (DL_FUNC) &_networkGEE_stochastic_binomial_tscs_solver, 8},
    {"_networkGEE_stochastic_gaussian_tscs_solver", (DL_FUNC) &_networkGEE_stochastic_gaussian_tscs_solver, 9},
    {"_networkGEE_meat_computation", (DL_FUNC) &_networkGEE_meat_computation, 4},
    {"_networkGEE_NewRaph", (DL_FUNC) &_networkGEE_NewRaph, 11},
    {"_networkGEE_StochNewRaph", (DL_FUNC) &_networkGEE_StochNewRaph, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_networkGEE(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}