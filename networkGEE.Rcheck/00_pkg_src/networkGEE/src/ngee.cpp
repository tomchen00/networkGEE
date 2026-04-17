// [[Rcpp::depends(RcppArmadillo)]]
/*
 ------------------------------------------------------------------------------
 networkGEE core C++ implementation

 This file implements generalized estimating equations for several working
 correlation structures used in cluster-randomized and stepped-wedge designs.
 The code is organized into five layers:

 1. Index / precomputation layer
 - Convert sorted cluster identifiers into flat zero-based offsets.
 - Provide fast internal samplers for stochastic fitting.

 2. Deterministic estimating-equation kernels
 - Hierarchical family: independence, simple exchangeable,
 nested exchangeable.
 - Block-exchangeable family for closed-cohort / TSCS designs.

 3. Deterministic sandwich layer
 - Recompute clusterwise score and Hessian contributions at the final
 parameter values and assemble robust covariance estimators.

 4. Stochastic fitting layer
 - Reuse the same deterministic kernels on subsamples.
 - Optionally perform one final full-data deterministic refinement.

 5. R-facing wrappers
 - Export functions for fitting, sandwich estimation, and indexing.

 Notation used throughout
 ------------------------
 For an observation v in a cluster,
 mu_v  = E(Y_v | X_v)
 U_v   = Var(Y_v | X_v)
 R_v   = (Y_v - mu_v) / sqrt(U_v)

 For a correlation class r, the second-order estimating equations are based on
 pair-products R_v R_v' with working mean rho_r on the corresponding set of
 pairs F_{ir}. The first-order estimating equations are standard GEE score
 updates for beta, and for Gaussian outcomes phi is updated through the usual
 residual-scale estimating equation.

 Conventions
 -----------
 - All stored offsets are zero-based.
 - All exported indexing objects expose zero-based offsets for C++ and a
 one-based copy of sampled observation indices for easy use in R.
 - cluster_id must already be sorted lexicographically by its columns.
 ------------------------------------------------------------------------------
 */
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

/***** PRECOMPUTE / INDEX LAYER ***********************************************
 Conventions:
 - all *_start fields are ZERO-BASED OFFSETS
 - obs_index0 is ZERO-BASED original row index
 - obs_index1 is ONE-BASED original row index (for easy use in R)
 - for q = 1 structures: cluster_start / cluster_size are sufficient
 - for q = 2 structures: cluster_sub_start / cluster_n_sub / sub_start / sub_size
 describe the second-level blocks
 *******************************************************************************/

// internal helpers ------------------------------------------------------------

/*
 Return the zero-based sequence 0, 1, ..., n-1.
 This is used repeatedly when building cluster and observation offsets.
 */
static inline arma::uvec ngee_seq0(const unsigned int n) {
  if (n == 0) return arma::uvec();
  return arma::regspace<arma::uvec>(0, n - 1);
}

static inline arma::uvec ngee_vec_to_uvec(const std::vector<unsigned int>& x) {
  arma::uvec out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i];
  return out;
}

static inline void ngee_append_uvec(std::vector<unsigned int>& out,
                                    const arma::uvec& x) {
  const std::size_t old_n = out.size();
  out.resize(old_n + x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    out[old_n + i] = x[i];
  }
}

static inline bool ngee_is_simple_corstr(const std::string& corstr) {
  return (corstr == "independence" || corstr == "simple-exchangeable");
}

static inline bool ngee_is_nested_corstr(const std::string& corstr) {
  return (corstr == "nested-exchangeable");
}

static inline bool ngee_is_block_corstr(const std::string& corstr) {
  return (corstr == "block-exchangeable");
}

static inline void ngee_validate_sorted_cluster_id(const arma::mat& cluster_id) {
  const int n = cluster_id.n_rows;
  const int q = cluster_id.n_cols;

  if (q == 1) {
    for (int i = 1; i < n; ++i) {
      if (cluster_id(i, 0) < cluster_id(i - 1, 0)) {
        Rcpp::stop("cluster_id must be sorted by the first column.");
      }
    }
  } else if (q == 2) {
    for (int i = 1; i < n; ++i) {
      const double c0_prev = cluster_id(i - 1, 0);
      const double c1_prev = cluster_id(i - 1, 1);
      const double c0_curr = cluster_id(i, 0);
      const double c1_curr = cluster_id(i, 1);

      if (c0_curr < c0_prev) {
        Rcpp::stop("cluster_id must be lexicographically sorted by columns 1, then 2.");
      }
      if (c0_curr == c0_prev && c1_curr < c1_prev) {
        Rcpp::stop("cluster_id must be lexicographically sorted by columns 1, then 2.");
      }
    }
  } else {
    Rcpp::stop("Only 1-column and 2-column cluster_id matrices are supported right now.");
  }
}

static inline arma::uvec ngee_sample_values(const arma::uvec& values,
                                            const unsigned int k,
                                            const bool replace) {
  if (values.n_elem == 0) return arma::uvec();
  if (!replace && k >= values.n_elem) return values;

  Rcpp::IntegerVector pool = Rcpp::wrap(values);
  Rcpp::IntegerVector draw = RcppArmadillo::sample(
    pool, static_cast<int>(k), replace, Rcpp::NumericVector::create()
  );
  arma::uvec out = Rcpp::as<arma::uvec>(draw);

  if (!replace) out = arma::sort(out);
  return out;
}

/*
 Precompute the flat index representation used by all downstream kernels.

 For a one-column cluster_id, this stores cluster starts and cluster sizes.
 For a two-column cluster_id, this additionally stores subgroup starts and
 subgroup sizes inside each top-level cluster.

 For block-exchangeable designs, the second column is interpreted as period.
 Balanced period sizes within cluster are checked here so that the working
 block-exchangeable algebra can assume a fixed cohort size per period.
 */
// [[Rcpp::export]]
Rcpp::List ngee_precompute_index_cpp(const arma::mat& cluster_id,
                                     const std::string& corstr,
                                     const bool validate_sorted = true,
                                     const bool validate_balanced = true) {
  const int n = cluster_id.n_rows;
  const int q = cluster_id.n_cols;

  if (n == 0) Rcpp::stop("cluster_id has zero rows.");
  if (q < 1 || q > 2) Rcpp::stop("cluster_id must have 1 or 2 columns.");

  if (ngee_is_simple_corstr(corstr)) {
    if (q != 1) {
      Rcpp::stop("For independence/simple-exchangeable, cluster_id must currently have 1 column.");
    }
  } else if (ngee_is_nested_corstr(corstr) || ngee_is_block_corstr(corstr)) {
    if (q != 2) {
      Rcpp::stop("For nested-exchangeable/block-exchangeable, cluster_id must currently have 2 columns.");
    }
  } else {
    Rcpp::stop("Unsupported corstr.");
  }

  if (validate_sorted) {
    ngee_validate_sorted_cluster_id(cluster_id);
  }

  std::vector<unsigned int> cluster_start_v;
  std::vector<unsigned int> cluster_size_v;
  std::vector<unsigned int> cluster_sub_start_v;
  std::vector<unsigned int> cluster_n_sub_v;
  std::vector<unsigned int> sub_start_v;
  std::vector<unsigned int> sub_size_v;
  std::vector<unsigned int> cluster_subject_n_v;

  Rcpp::LogicalVector cluster_balanced_v;

  unsigned int total_sub = 0;
  int row = 0;

  while (row < n) {
    const int cluster_row_start = row;
    const double cluster_val = cluster_id(row, 0);

    while (row < n && cluster_id(row, 0) == cluster_val) {
      ++row;
    }

    const int cluster_row_end = row;
    const unsigned int this_cluster_size =
      static_cast<unsigned int>(cluster_row_end - cluster_row_start);

    cluster_start_v.push_back(static_cast<unsigned int>(cluster_row_start));
    cluster_size_v.push_back(this_cluster_size);

    if (q == 2) {
      cluster_sub_start_v.push_back(total_sub);

      int sub_row = cluster_row_start;
      unsigned int n_sub_this_cluster = 0;
      bool balanced_this_cluster = true;
      unsigned int first_sub_size = 0;

      while (sub_row < cluster_row_end) {
        const int sub_start = sub_row;
        const double sub_val = cluster_id(sub_row, 1);

        while (sub_row < cluster_row_end && cluster_id(sub_row, 1) == sub_val) {
          ++sub_row;
        }

        const unsigned int this_sub_size =
          static_cast<unsigned int>(sub_row - sub_start);

        sub_start_v.push_back(static_cast<unsigned int>(sub_start));
        sub_size_v.push_back(this_sub_size);

        if (n_sub_this_cluster == 0) {
          first_sub_size = this_sub_size;
        } else if (this_sub_size != first_sub_size) {
          balanced_this_cluster = false;
        }

        ++n_sub_this_cluster;
        ++total_sub;
      }

      cluster_n_sub_v.push_back(n_sub_this_cluster);
      cluster_subject_n_v.push_back(first_sub_size);
      cluster_balanced_v.push_back(balanced_this_cluster);

      if (ngee_is_block_corstr(corstr) && validate_balanced && !balanced_this_cluster) {
        Rcpp::stop(
          "block-exchangeable currently requires equal subgroup/period sizes within each cluster."
        );
      }
    }
  }

  const arma::uvec cluster_start = ngee_vec_to_uvec(cluster_start_v);
  const arma::uvec cluster_size = ngee_vec_to_uvec(cluster_size_v);
  const arma::uvec cluster_sub_start = ngee_vec_to_uvec(cluster_sub_start_v);
  const arma::uvec cluster_n_sub = ngee_vec_to_uvec(cluster_n_sub_v);
  const arma::uvec sub_start = ngee_vec_to_uvec(sub_start_v);
  const arma::uvec sub_size = ngee_vec_to_uvec(sub_size_v);
  const arma::uvec cluster_subject_n = ngee_vec_to_uvec(cluster_subject_n_v);

  return Rcpp::List::create(
    Rcpp::Named("version") = "ngee_index_v1",
    Rcpp::Named("corstr") = corstr,
    Rcpp::Named("n_obs") = n,
    Rcpp::Named("n_cluster") = static_cast<int>(cluster_start.n_elem),
    Rcpp::Named("n_sub_total") = static_cast<int>(sub_start.n_elem),
    Rcpp::Named("cluster_start") = cluster_start,
    Rcpp::Named("cluster_size") = cluster_size,
    Rcpp::Named("cluster_sub_start") = cluster_sub_start,
    Rcpp::Named("cluster_n_sub") = cluster_n_sub,
    Rcpp::Named("sub_start") = sub_start,
    Rcpp::Named("sub_size") = sub_size,
    Rcpp::Named("cluster_subject_n") = cluster_subject_n,
    Rcpp::Named("cluster_balanced") = cluster_balanced_v
  );
}

/*
 Sample clusters and, depending on the correlation structure, lower-level units.

 Returned fields mirror the full index object but now describe only the sampled
 subproblem. This function is exported for transparency and testing; the hot
 stochastic loops use lighter internal helpers defined later in the file.
 */
// [[Rcpp::export]]
Rcpp::List ngee_sample_index_cpp(const Rcpp::List& index_data,
                                 const arma::uvec& batch_size,
                                 const bool replace = false) {
  const std::string corstr = Rcpp::as<std::string>(index_data["corstr"]);

  const arma::uvec cluster_start = Rcpp::as<arma::uvec>(index_data["cluster_start"]);
  const arma::uvec cluster_size = Rcpp::as<arma::uvec>(index_data["cluster_size"]);
  const arma::uvec cluster_sub_start = Rcpp::as<arma::uvec>(index_data["cluster_sub_start"]);
  const arma::uvec cluster_n_sub = Rcpp::as<arma::uvec>(index_data["cluster_n_sub"]);
  const arma::uvec sub_start_full = Rcpp::as<arma::uvec>(index_data["sub_start"]);
  const arma::uvec sub_size_full = Rcpp::as<arma::uvec>(index_data["sub_size"]);
  const arma::uvec cluster_subject_n = Rcpp::as<arma::uvec>(index_data["cluster_subject_n"]);
  const Rcpp::LogicalVector cluster_balanced = Rcpp::as<Rcpp::LogicalVector>(index_data["cluster_balanced"]);

  const unsigned int I_full = cluster_start.n_elem;
  const arma::uvec all_clusters = ngee_seq0(I_full);

  if (I_full == 0) Rcpp::stop("index_data contains zero clusters.");
  if (batch_size.n_elem == 0) Rcpp::stop("batch_size must have positive length.");

  arma::uvec cluster_sel = ngee_sample_values(all_clusters, batch_size[0], replace);

  std::vector<unsigned int> obs_index_v;
  std::vector<unsigned int> cluster_start_v;
  std::vector<unsigned int> cluster_size_v;
  std::vector<unsigned int> cluster_sub_start_v;
  std::vector<unsigned int> cluster_n_sub_v;
  std::vector<unsigned int> sub_start_v;
  std::vector<unsigned int> sub_size_v;

  std::vector<unsigned int> full_cluster_size_sel_v;
  std::vector<unsigned int> full_cluster_n_sub_sel_v;
  std::vector<unsigned int> full_sub_size_sel_v;

  unsigned int obs_cursor = 0;
  unsigned int sub_cursor = 0;

  if (ngee_is_simple_corstr(corstr)) {
    if (batch_size.n_elem < 2) {
      Rcpp::stop("For independence/simple-exchangeable, batch_size must have length >= 2.");
    }

    const unsigned int n_obs_target = batch_size[1];

    for (arma::uword ii = 0; ii < cluster_sel.n_elem; ++ii) {
      const unsigned int ci = cluster_sel[ii];
      const unsigned int cstart = cluster_start[ci];
      const unsigned int csize = cluster_size[ci];

      arma::uvec obs_pos = ngee_seq0(csize);
      obs_pos = ngee_sample_values(obs_pos, n_obs_target, replace);
      obs_pos += cstart;

      cluster_start_v.push_back(obs_cursor);
      cluster_size_v.push_back(obs_pos.n_elem);
      full_cluster_size_sel_v.push_back(csize);

      ngee_append_uvec(obs_index_v, obs_pos);
      obs_cursor += obs_pos.n_elem;
    }
  } else if (ngee_is_nested_corstr(corstr)) {
    if (batch_size.n_elem < 3) {
      Rcpp::stop("For nested-exchangeable, batch_size must have length >= 3.");
    }

    const unsigned int n_sub_target = batch_size[1];
    const unsigned int n_obs_target = batch_size[2];

    for (arma::uword ii = 0; ii < cluster_sel.n_elem; ++ii) {
      const unsigned int ci = cluster_sel[ii];
      const unsigned int sub0 = cluster_sub_start[ci];
      const unsigned int nsub_full = cluster_n_sub[ci];

      arma::uvec sub_ids = ngee_seq0(nsub_full);
      sub_ids += sub0;
      sub_ids = ngee_sample_values(sub_ids, n_sub_target, replace);

      const unsigned int cluster_obs_start = obs_cursor;

      cluster_start_v.push_back(cluster_obs_start);
      cluster_sub_start_v.push_back(sub_cursor);
      cluster_n_sub_v.push_back(sub_ids.n_elem);

      full_cluster_size_sel_v.push_back(cluster_size[ci]);
      full_cluster_n_sub_sel_v.push_back(nsub_full);

      for (arma::uword jj = 0; jj < sub_ids.n_elem; ++jj) {
        const unsigned int sid = sub_ids[jj];
        const unsigned int sstart = sub_start_full[sid];
        const unsigned int ssize = sub_size_full[sid];

        arma::uvec obs_pos = ngee_seq0(ssize);
        obs_pos = ngee_sample_values(obs_pos, n_obs_target, replace);
        obs_pos += sstart;

        sub_start_v.push_back(obs_cursor);
        sub_size_v.push_back(obs_pos.n_elem);
        full_sub_size_sel_v.push_back(ssize);

        ngee_append_uvec(obs_index_v, obs_pos);
        obs_cursor += obs_pos.n_elem;
        ++sub_cursor;
      }

      cluster_size_v.push_back(obs_cursor - cluster_obs_start);
    }
  } else if (ngee_is_block_corstr(corstr)) {
    if (batch_size.n_elem < 2) {
      Rcpp::stop("For block-exchangeable, batch_size must have length >= 2.");
    }

    const unsigned int n_subject_target = batch_size[1];

    for (arma::uword ii = 0; ii < cluster_sel.n_elem; ++ii) {
      const unsigned int ci = cluster_sel[ii];

      if (cluster_balanced.size() > 0 && !cluster_balanced[ci]) {
        Rcpp::stop(
          "Encountered an unbalanced cluster in block-exchangeable sampling. "
          "Rebuild index_data with balanced clusters only."
        );
      }

      const unsigned int sub0 = cluster_sub_start[ci];
      const unsigned int nsub_full = cluster_n_sub[ci];
      const unsigned int subject_n_full =
        (cluster_subject_n.n_elem > 0) ? cluster_subject_n[ci] : sub_size_full[sub0];

      arma::uvec subject_pos = ngee_seq0(subject_n_full);
      subject_pos = ngee_sample_values(subject_pos, n_subject_target, replace);

      const unsigned int cluster_obs_start = obs_cursor;

      cluster_start_v.push_back(cluster_obs_start);
      cluster_sub_start_v.push_back(sub_cursor);
      cluster_n_sub_v.push_back(nsub_full);

      full_cluster_size_sel_v.push_back(cluster_size[ci]);
      full_cluster_n_sub_sel_v.push_back(nsub_full);

      for (unsigned int t = 0; t < nsub_full; ++t) {
        const unsigned int sid = sub0 + t;
        const unsigned int sstart = sub_start_full[sid];
        const unsigned int ssize = sub_size_full[sid];

        arma::uvec obs_pos = subject_pos;
        obs_pos += sstart;

        sub_start_v.push_back(obs_cursor);
        sub_size_v.push_back(obs_pos.n_elem);
        full_sub_size_sel_v.push_back(ssize);

        ngee_append_uvec(obs_index_v, obs_pos);
        obs_cursor += obs_pos.n_elem;
        ++sub_cursor;
      }

      cluster_size_v.push_back(obs_cursor - cluster_obs_start);
    }
  } else {
    Rcpp::stop("Unsupported corstr.");
  }

  const arma::uvec obs_index0 = ngee_vec_to_uvec(obs_index_v);
  const arma::uvec obs_index1 = obs_index0 + 1;
  const arma::uvec cluster_start_s = ngee_vec_to_uvec(cluster_start_v);
  const arma::uvec cluster_size_s = ngee_vec_to_uvec(cluster_size_v);
  const arma::uvec cluster_sub_start_s = ngee_vec_to_uvec(cluster_sub_start_v);
  const arma::uvec cluster_n_sub_s = ngee_vec_to_uvec(cluster_n_sub_v);
  const arma::uvec sub_start_s = ngee_vec_to_uvec(sub_start_v);
  const arma::uvec sub_size_s = ngee_vec_to_uvec(sub_size_v);
  const arma::uvec full_cluster_size_sel = ngee_vec_to_uvec(full_cluster_size_sel_v);
  const arma::uvec full_cluster_n_sub_sel = ngee_vec_to_uvec(full_cluster_n_sub_sel_v);
  const arma::uvec full_sub_size_sel = ngee_vec_to_uvec(full_sub_size_sel_v);

  return Rcpp::List::create(
    Rcpp::Named("version") = "ngee_sample_v1",
    Rcpp::Named("corstr") = corstr,
    Rcpp::Named("n_obs") = static_cast<int>(obs_index0.n_elem),
    Rcpp::Named("n_cluster") = static_cast<int>(cluster_sel.n_elem),
    Rcpp::Named("cluster_selected0") = cluster_sel,
    Rcpp::Named("cluster_selected1") = cluster_sel + 1,
    Rcpp::Named("obs_index0") = obs_index0,
    Rcpp::Named("obs_index1") = obs_index1,
    Rcpp::Named("cluster_start") = cluster_start_s,
    Rcpp::Named("cluster_size") = cluster_size_s,
    Rcpp::Named("cluster_sub_start") = cluster_sub_start_s,
    Rcpp::Named("cluster_n_sub") = cluster_n_sub_s,
    Rcpp::Named("sub_start") = sub_start_s,
    Rcpp::Named("sub_size") = sub_size_s,
    Rcpp::Named("full_n_cluster") = static_cast<int>(I_full),
    Rcpp::Named("full_cluster_size") = full_cluster_size_sel,
    Rcpp::Named("full_cluster_n_sub") = full_cluster_n_sub_sel,
    Rcpp::Named("full_sub_size") = full_sub_size_sel
  );
}

/***** DETERMINISTIC HIERARCHICAL KERNELS + OPTIMIZER ***************************
 Supports:
 - corstr: "independence", "simple-exchangeable", "nested-exchangeable"
 - family: "binomial", "gaussian"

 Uses flat index data from ngee_precompute_index_cpp().

 Notes:
 - For "independence", rho is ignored/fixed and not updated.
 - For "simple-exchangeable", q must be 1 and rho is updated.
 - For "nested-exchangeable", q must be 2 and rho is updated.
 - Gaussian explicitly updates phi each iteration.
 *******************************************************************************/

/*
 Lightweight C++ view of the hierarchical index object.
 This avoids repeated Rcpp list unpacking inside deterministic kernels.
 */
struct ngee_hier_index_view {
  std::string corstr;
  arma::uvec cluster_start;
  arma::uvec cluster_size;
  arma::uvec cluster_sub_start;
  arma::uvec cluster_n_sub;
  arma::uvec sub_start;
  arma::uvec sub_size;
  arma::uword n_cluster;
  arma::uword n_obs;
};

struct ngee_hier_totals {
  arma::mat H_beta;
  arma::colvec G_beta;
  double H_phi;
  double G_phi;
  arma::mat H_rho;
  arma::colvec G_rho;
};

static inline ngee_hier_index_view ngee_get_hier_index_view(const Rcpp::List& index_data) {
  ngee_hier_index_view idx;
  idx.corstr = Rcpp::as<std::string>(index_data["corstr"]);
  idx.cluster_start = Rcpp::as<arma::uvec>(index_data["cluster_start"]);
  idx.cluster_size = Rcpp::as<arma::uvec>(index_data["cluster_size"]);
  idx.cluster_sub_start = Rcpp::as<arma::uvec>(index_data["cluster_sub_start"]);
  idx.cluster_n_sub = Rcpp::as<arma::uvec>(index_data["cluster_n_sub"]);
  idx.sub_start = Rcpp::as<arma::uvec>(index_data["sub_start"]);
  idx.sub_size = Rcpp::as<arma::uvec>(index_data["sub_size"]);
  idx.n_cluster = idx.cluster_start.n_elem;
  idx.n_obs = Rcpp::as<int>(index_data["n_obs"]);
  return idx;
}

static inline arma::colvec ngee_invlogit_vec(const arma::colvec& eta) {
  return 1.0 / (1.0 + arma::exp(-eta));
}

static inline double ngee_rel_change_vec(const arma::colvec& change,
                                         const arma::colvec& updated) {
  return 2.0 * arma::sum(
      arma::abs(change) /
        (arma::abs(updated) + 0.001 + arma::abs(arma::abs(updated) - 0.001))
  );
}

static inline double ngee_rel_change_scalar(const double change,
                                            const double updated) {
  return 2.0 * std::abs(change) /
    (std::abs(updated) + 0.001 + std::abs(std::abs(updated) - 0.001));
}

static inline arma::colvec ngee_safe_solve_vec(const arma::mat& A,
                                               const arma::colvec& b) {
  arma::colvec x;
  bool ok = arma::solve(x, A, b, arma::solve_opts::fast);
  if (!ok) {
    x = arma::pinv(A) * b;
  }
  return x;
}

static inline arma::mat ngee_safe_inv_mat(const arma::mat& A) {
  arma::mat X;
  bool ok = arma::inv(X, A, arma::inv_opts::fast);
  if (!ok) {
    X = arma::pinv(A);
  }
  return X;
}

static inline double ngee_max_cluster_size(const ngee_hier_index_view& idx) {
  if (idx.cluster_size.n_elem == 0) return 1.0;
  return static_cast<double>(idx.cluster_size.max());
}

static inline double ngee_min_sub_size(const ngee_hier_index_view& idx) {
  if (idx.sub_size.n_elem == 0) return 1.0;
  return static_cast<double>(idx.sub_size.min());
}

/*
 Clamp the simple-exchangeable working correlation into a numerically safe
 interval. The lower bound uses the largest observed cluster size so that the
 exchangeable correlation matrix stays away from singularity.
 */
static inline void ngee_stabilize_simple_rho(arma::colvec& rho,
                                             const ngee_hier_index_view& idx,
                                             const double rho_eps) {
  if (rho.n_elem != 1) Rcpp::stop("simple-exchangeable requires rho0 of length 1.");
  const double jmax = std::max(2.0, ngee_max_cluster_size(idx));
  const double lower = -1.0 / (jmax - 1.0) + rho_eps;
  const double upper = 1.0 - rho_eps;
  if (!std::isfinite(rho[0])) rho[0] = 0.05;
  rho[0] = std::max(lower, std::min(upper, rho[0]));
}

/*
 Feasibility check for nested-exchangeable working correlations.

 The two nested-exchangeable correlation parameters are interpreted as
 rho[0] = within-subgroup correlation
 rho[1] = between-subgroup correlation.

 For the inverse formulas used below we need:
 - 1 - rho[0] > 0
 - (1 - rho[0]) + m (rho[0] - rho[1]) > 0 for every subgroup size m
 - 1 + rho[1] * sum_j v_j m_j > 0 for the Sherman-Morrison correction

 The last condition is algebraically equivalent to requiring that the final
 rank-one correction in the nested-exchangeable inverse remain nonsingular.
 */
static inline bool ngee_nested_rho_feasible(const arma::colvec& rho,
                                            const ngee_hier_index_view& idx,
                                            const double rho_eps) {
  if (rho.n_elem != 2) return false;
  if (!std::isfinite(rho[0]) || !std::isfinite(rho[1])) return false;
  if (rho[0] >= 1.0 - rho_eps) return false;
  if (rho[0] <= rho[1] + rho_eps) return false;

  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  if (idx.sub_size.n_elem == 0) return true;

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    if (idx.cluster_n_sub.n_elem == 0 || idx.cluster_n_sub[i] == 0) continue;

    const arma::uword s0 = idx.cluster_sub_start[i];
    const arma::uword nsub = idx.cluster_n_sub[i];
    arma::vec m = arma::conv_to<arma::vec>::from(idx.sub_size.subvec(s0, s0 + nsub - 1));

    for (arma::uword j = 0; j < m.n_elem; ++j) {
      if (rho0 + m[j] * delta <= rho_eps) return false;
    }

    arma::vec vij = 1.0 / rho0 - delta / (rho0 * (rho0 / m + delta));
    const double denom = 1.0 + rho[1] * arma::dot(vij, m);
    if (denom <= rho_eps) return false;
  }

  return true;
}

/*
 Scalar coefficient used in the nested-exchangeable Sherman-Morrison update.

 Writing
 c = 1 / (1 / rho_between + sum(v))
 = rho_between / (1 + rho_between * sum(v))
 avoids explicit division by rho_between, so the code remains well-defined when
 rho_between is close to or equal to zero.
 */
static inline double ngee_nested_c_value(const double rho_between,
                                         const double sum_v,
                                         const double rho_eps = 1e-12) {
  const double denom = 1.0 + rho_between * sum_v;
  if (std::abs(denom) <= rho_eps) {
    Rcpp::stop("Nested-exchangeable working covariance is numerically singular.");
  }
  return rho_between / denom;
}

/*
 Numerical stabilization for nested-exchangeable working correlations.

 The old implementation effectively assumed rho[1] > 0 because it used 1/rho[1]
 directly.  The updated code allows rho[1] to be negative when the working
 correlation remains admissible, and shrinks infeasible proposals smoothly toward
 zero until all inverse formulas are well-defined.
 */
static inline void ngee_stabilize_nested_rho(arma::colvec& rho,
                                             const ngee_hier_index_view& idx,
                                             const double rho_eps) {
  if (rho.n_elem != 2) Rcpp::stop("nested-exchangeable requires rho0 of length 2.");
  if (!std::isfinite(rho[0])) rho[0] = 0.05;
  if (!std::isfinite(rho[1])) rho[1] = 0.0;

  rho[0] = std::max(-1.0 + rho_eps, std::min(1.0 - rho_eps, rho[0]));
  rho[1] = std::max(-1.0 + rho_eps, std::min(1.0 - rho_eps, rho[1]));

  for (int it = 0; it < 100; ++it) {
    if (rho[0] <= rho[1] + rho_eps) {
      rho[1] = rho[0] - rho_eps;
    }

    if (ngee_nested_rho_feasible(rho, idx, rho_eps)) return;
    rho *= 0.5;
  }

  rho.zeros();
}

/*
 Deterministic hierarchical kernel for binomial outcomes with one correlation
 parameter (independence/simple exchangeable).

 For each cluster i this computes:
 G_beta,i = D_i^T V_i^{-1} (Y_i - mu_i)
 H_beta,i = D_i^T V_i^{-1} D_i
 and the corresponding second-order score/Hessian for rho based on standardized
 residual products.
 */
static inline ngee_hier_totals ngee_kernel_binomial_hier_q1(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(1, 1);
  out.G_rho.zeros(1);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;
  const arma::mat std_design_mat = design_mat.each_col() % U_sqrt;

  const double rho0 = 1.0 - rho[0];

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    const arma::rowvec Xsum = arma::sum(Xtilde, 0);
    const double Esum = arma::sum(Etilde);
    const double c = rho[0] / (rho0 * (rho0 + rho[0] * static_cast<double>(csize)));

    out.G_beta += Xtilde.t() * Etilde / rho0 - c * Xsum.t() * Esum;
    out.H_beta += Xtilde.t() * Xtilde / rho0 - c * Xsum.t() * Xsum;

    out.G_rho[0] += (Esum * Esum - arma::dot(Etilde, Etilde)) / 2.0
    - static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) * rho[0] / 2.0;

    out.H_rho(0, 0) += static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;
  }

  return out;
}

/*
 Gaussian counterpart of the one-parameter hierarchical kernel.

 The working mean is linear in beta, phi is updated through the Gaussian scale
 estimating equation, and the rho block is again built from standardized
 residual products.
 */
static inline ngee_hier_totals ngee_kernel_gaussian_hier_q1(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(1, 1);
  out.G_rho.zeros(1);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;
  const double sqrt_phi = std::sqrt(phi);
  const arma::colvec std_resid = resid / sqrt_phi;
  const arma::mat std_design_mat = design_mat / sqrt_phi;

  const double rho0 = 1.0 - rho[0];

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    const arma::rowvec Xsum = arma::sum(Xtilde, 0);
    const double Esum = arma::sum(Etilde);
    const double c = rho[0] / (rho0 * (rho0 + rho[0] * static_cast<double>(csize)));

    out.G_beta += Xtilde.t() * Etilde / rho0 - c * Xsum.t() * Esum;
    out.H_beta += Xtilde.t() * Xtilde / rho0 - c * Xsum.t() * Xsum;

    out.G_phi += phi * (arma::dot(Etilde, Etilde) - static_cast<double>(csize));
    out.H_phi += static_cast<double>(csize);

    out.G_rho[0] += (Esum * Esum - arma::dot(Etilde, Etilde)) / 2.0
    - static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) * rho[0] / 2.0;

    out.H_rho(0, 0) += static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;
  }

  return out;
}

/*
 Deterministic hierarchical kernel for the nested-exchangeable binomial model.

 The correlation structure is decomposed into within-subgroup and between-
 subgroup pieces using the analytical inverse from the nested-exchangeable
 working covariance.
 */
static inline ngee_hier_totals ngee_kernel_binomial_hier_q2(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(2, 2);
  out.G_rho.zeros(2);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;
  const arma::mat std_design_mat = design_mat.each_col() % U_sqrt;

  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword nsub   = idx.cluster_n_sub[i];

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    arma::colvec v(csize, arma::fill::zeros);
    arma::uword cursor = 0;
    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword len = idx.sub_size[s0 + s];
      const double vij = 1.0 / rho0 - delta / (rho0 * (rho0 / static_cast<double>(len) + delta));
      v.subvec(cursor, cursor + len - 1).fill(vij);
      cursor += len;
    }

    const double c = ngee_nested_c_value(rho[1], arma::sum(v));

    double inner_pairs = 0.0;
    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword subid = s0 + s;
      const arma::uword len = idx.sub_size[subid];
      const arma::uword local_start = idx.sub_start[subid] - cstart;
      const arma::uword local_end   = local_start + len - 1;

      const auto Xsub = Xtilde.rows(local_start, local_end);
      const auto Esub = Etilde.rows(local_start, local_end);

      const arma::rowvec Xsub_sum = arma::sum(Xsub, 0);
      const double Esub_sum = arma::sum(Esub);
      const double denom = rho0 * (rho0 + delta * static_cast<double>(len));
      const double subcorr = delta / denom;

      out.G_beta += -Xsub_sum.t() * Esub_sum * subcorr;
      out.H_beta += -Xsub_sum.t() * Xsub_sum * subcorr;

      const double sumsq_sub = arma::dot(Esub, Esub);
      const double a = sumsq_sub - static_cast<double>(len);
      const double a0 = (Esub_sum * Esub_sum - sumsq_sub) / 2.0;

      out.G_rho[0] += a0 - static_cast<double>(len) * (static_cast<double>(len) - 1.0) * rho[0] / 2.0;
      out.G_rho[1] -= (a + static_cast<double>(len)) / 2.0 + a0;

      inner_pairs += static_cast<double>(len) * (static_cast<double>(len) - 1.0) / 2.0;
    }

    const double total_pairs = static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;

    out.G_beta += Xtilde.t() * Etilde / rho0 - c * (Xtilde.t() * v) * arma::as_scalar(v.t() * Etilde);
    out.H_beta += Xtilde.t() * Xtilde / rho0 - c * (Xtilde.t() * v) * (v.t() * Xtilde);

    out.G_rho[1] += std::pow(arma::sum(Etilde), 2.0) / 2.0 - (total_pairs - inner_pairs) * rho[1];
    out.H_rho(0, 0) += inner_pairs;
    out.H_rho(1, 1) += total_pairs - inner_pairs;
  }

  return out;
}

/*
 Gaussian nested-exchangeable kernel.

 This mirrors the binomial nested kernel but carries an additional phi block.
 The same working inverse is used, with beta and phi updated from Gaussian
 estimating equations.
 */
static inline ngee_hier_totals ngee_kernel_gaussian_hier_q2(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(2, 2);
  out.G_rho.zeros(2);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;
  const double sqrt_phi = std::sqrt(phi);
  const arma::colvec std_resid = resid / sqrt_phi;
  const arma::mat std_design_mat = design_mat / sqrt_phi;

  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword nsub   = idx.cluster_n_sub[i];

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    arma::colvec v(csize, arma::fill::zeros);
    arma::uword cursor = 0;
    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword len = idx.sub_size[s0 + s];
      const double vij = 1.0 / rho0 - delta / (rho0 * (rho0 / static_cast<double>(len) + delta));
      v.subvec(cursor, cursor + len - 1).fill(vij);
      cursor += len;
    }

    const double c = ngee_nested_c_value(rho[1], arma::sum(v));

    double inner_pairs = 0.0;
    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword subid = s0 + s;
      const arma::uword len = idx.sub_size[subid];
      const arma::uword local_start = idx.sub_start[subid] - cstart;
      const arma::uword local_end   = local_start + len - 1;

      const auto Xsub = Xtilde.rows(local_start, local_end);
      const auto Esub = Etilde.rows(local_start, local_end);

      const arma::rowvec Xsub_sum = arma::sum(Xsub, 0);
      const double Esub_sum = arma::sum(Esub);
      const double sumsq_sub = arma::dot(Esub, Esub);
      const double denom = rho0 * (rho0 + delta * static_cast<double>(len));
      const double subcorr = delta / denom;

      out.G_beta += -Xsub_sum.t() * Esub_sum * subcorr;
      out.H_beta += -Xsub_sum.t() * Xsub_sum * subcorr;

      const double a = phi * (sumsq_sub - static_cast<double>(len));
      const double a0 = (Esub_sum * Esub_sum - sumsq_sub) / 2.0;

      out.G_rho[0] += a0 - static_cast<double>(len) * (static_cast<double>(len) - 1.0) * rho[0] / 2.0;
      out.G_rho[1] -= (a / phi + static_cast<double>(len)) / 2.0 + a0;

      inner_pairs += static_cast<double>(len) * (static_cast<double>(len) - 1.0) / 2.0;
    }

    const double total_pairs = static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;

    out.G_beta += Xtilde.t() * Etilde / rho0 - c * (Xtilde.t() * v) * arma::as_scalar(v.t() * Etilde);
    out.H_beta += Xtilde.t() * Xtilde / rho0 - c * (Xtilde.t() * v) * (v.t() * Xtilde);

    out.G_phi += phi * (arma::dot(Etilde, Etilde) - static_cast<double>(csize));
    out.H_phi += static_cast<double>(csize);

    out.G_rho[1] += std::pow(arma::sum(Etilde), 2.0) / 2.0 - (total_pairs - inner_pairs) * rho[1];
    out.H_rho(0, 0) += inner_pairs;
    out.H_rho(1, 1) += total_pairs - inner_pairs;
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_hier_dispatch(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx,
    const std::string& family,
    const std::string& corstr) {

  if (family == "binomial") {
    if (corstr == "independence" || corstr == "simple-exchangeable") {
      return ngee_kernel_binomial_hier_q1(beta, rho, outcome, design_mat, idx);
    } else if (corstr == "nested-exchangeable") {
      return ngee_kernel_binomial_hier_q2(beta, rho, outcome, design_mat, idx);
    }
  } else if (family == "gaussian") {
    if (corstr == "independence" || corstr == "simple-exchangeable") {
      return ngee_kernel_gaussian_hier_q1(beta, phi, rho, outcome, design_mat, idx);
    } else if (corstr == "nested-exchangeable") {
      return ngee_kernel_gaussian_hier_q2(beta, phi, rho, outcome, design_mat, idx);
    }
  }

  Rcpp::stop("Unsupported family/corstr combination in hierarchical kernel.");
  return ngee_hier_totals();
}

/*
 Expose the deterministic hierarchical score/Hessian blocks at a fixed parameter
 value. This is mainly useful for testing and debugging against older code.
 */
// [[Rcpp::export]]
Rcpp::List ngee_kernel_det_hier_cpp(const arma::colvec& beta,
                                    const double phi,
                                    const arma::colvec& rho,
                                    const arma::colvec& outcome,
                                    const arma::mat& design_mat,
                                    const Rcpp::List& index_data,
                                    const std::string& family,
                                    const std::string& corstr,
                                    const double rho_eps = 1e-4) {
  const ngee_hier_index_view idx = ngee_get_hier_index_view(index_data);

  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");

  arma::colvec rho_work = rho;
  if (corstr == "simple-exchangeable") {
    ngee_stabilize_simple_rho(rho_work, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    ngee_stabilize_nested_rho(rho_work, idx, rho_eps);
  }

  ngee_hier_totals out = ngee_kernel_hier_dispatch(
    beta, phi, rho_work, outcome, design_mat, idx, family, corstr
  );

  return Rcpp::List::create(
    Rcpp::Named("G_beta") = out.G_beta,
    Rcpp::Named("H_beta") = out.H_beta,
    Rcpp::Named("G_phi")  = out.G_phi,
    Rcpp::Named("H_phi")  = out.H_phi,
    Rcpp::Named("G_rho")  = out.G_rho,
    Rcpp::Named("H_rho")  = out.H_rho,
    Rcpp::Named("rho_used") = rho_work
  );
}


// Deterministic Newton/Fisher-scoring fit for hierarchical structures.
// The update blocks are separated into beta, phi (Gaussian only), and rho.
// Convergence is monitored through relative parameter changes.

// [[Rcpp::export]]
Rcpp::List ngee_fit_det_hier_cpp(const arma::colvec& outcome,
                                 const arma::mat& design_mat,
                                 const Rcpp::List& index_data,
                                 const std::string& family,
                                 const std::string& corstr,
                                 arma::colvec beta0,
                                 double phi0,
                                 arma::colvec rho0,
                                 const double tol = 1e-8,
                                 const int maxit = 200,
                                 const double phi_min = 1e-8,
                                 const double rho_eps = 1e-4) {
  const ngee_hier_index_view idx = ngee_get_hier_index_view(index_data);

  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta0.n_elem != design_mat.n_cols) Rcpp::stop("beta0 has wrong length.");

  if (!(family == "binomial" || family == "gaussian")) {
    Rcpp::stop("family must be 'binomial' or 'gaussian'.");
  }
  if (!(corstr == "independence" || corstr == "simple-exchangeable" || corstr == "nested-exchangeable")) {
    Rcpp::stop("Unsupported corstr for hierarchical deterministic fit.");
  }

  const std::string corstr_idx = Rcpp::as<std::string>(index_data["corstr"]);
  if (corstr_idx != corstr) {
    Rcpp::stop("corstr does not match index_data$corstr.");
  }

  const bool update_phi = (family == "gaussian");
  const bool update_rho = (corstr == "simple-exchangeable" || corstr == "nested-exchangeable");

  if (corstr == "independence") {
    if (rho0.n_elem < 1) rho0 = arma::zeros<arma::colvec>(1);
  } else if (corstr == "simple-exchangeable") {
    if (rho0.n_elem != 1) Rcpp::stop("simple-exchangeable requires rho0 of length 1.");
    ngee_stabilize_simple_rho(rho0, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    if (rho0.n_elem != 2) Rcpp::stop("nested-exchangeable requires rho0 of length 2.");
    ngee_stabilize_nested_rho(rho0, idx, rho_eps);
  }

  arma::colvec beta = beta0;
  arma::colvec rho = rho0;
  double phi = std::max(phi0, phi_min);

  arma::colvec dbeta(beta.n_elem, arma::fill::zeros);
  arma::colvec drho(rho.n_elem, arma::fill::zeros);
  double dphi = 0.0;

  arma::mat Info_beta(beta.n_elem, beta.n_elem, arma::fill::eye);
  arma::mat Info_phi(1, 1, arma::fill::eye);
  arma::mat Info_rho(rho.n_elem, rho.n_elem, arma::fill::zeros);

  ngee_hier_totals last_blocks;
  double err = arma::datum::inf;
  int iter = 0;
  bool converged = false;

  for (iter = 1; iter <= maxit; ++iter) {
    if (corstr == "simple-exchangeable") {
      ngee_stabilize_simple_rho(rho, idx, rho_eps);
    } else if (corstr == "nested-exchangeable") {
      ngee_stabilize_nested_rho(rho, idx, rho_eps);
    }

    last_blocks = ngee_kernel_hier_dispatch(
      beta, phi, rho, outcome, design_mat, idx, family, corstr
    );

    dbeta = ngee_safe_solve_vec(last_blocks.H_beta, last_blocks.G_beta);
    beta += dbeta;
    err = ngee_rel_change_vec(dbeta, beta);
    Info_beta = ngee_safe_inv_mat(last_blocks.H_beta);

    if (update_phi) {
      dphi = last_blocks.G_phi / last_blocks.H_phi;
      phi = std::max(phi + dphi, phi_min);
      err += ngee_rel_change_scalar(dphi, phi);
      Info_phi(0, 0) = 1.0 / last_blocks.H_phi;
    } else {
      dphi = 0.0;
      Info_phi(0, 0) = 1.0;
    }

    if (update_rho) {
      drho = ngee_safe_solve_vec(last_blocks.H_rho, last_blocks.G_rho);
      rho += drho;

      if (corstr == "simple-exchangeable") {
        ngee_stabilize_simple_rho(rho, idx, rho_eps);
      } else {
        ngee_stabilize_nested_rho(rho, idx, rho_eps);
      }

      err += ngee_rel_change_vec(drho, rho);
      Info_rho = ngee_safe_inv_mat(last_blocks.H_rho);
    } else {
      drho.zeros();
      Info_rho.zeros();
    }

    if (err <= tol) {
      converged = true;
      break;
    }
  }

  // refresh blocks at final values so returned Hessian/gradient correspond to final estimates
  if (corstr == "simple-exchangeable") {
    ngee_stabilize_simple_rho(rho, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    ngee_stabilize_nested_rho(rho, idx, rho_eps);
  }

  last_blocks = ngee_kernel_hier_dispatch(
    beta, phi, rho, outcome, design_mat, idx, family, corstr
  );

  Info_beta = ngee_safe_inv_mat(last_blocks.H_beta);
  if (update_phi) Info_phi(0, 0) = 1.0 / last_blocks.H_phi;
  if (update_rho) Info_rho = ngee_safe_inv_mat(last_blocks.H_rho);

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("Info_beta") = Info_beta,
    Rcpp::Named("Info_phi") = Info_phi,
    Rcpp::Named("Info_rho") = Info_rho,
    Rcpp::Named("G_beta") = last_blocks.G_beta,
    Rcpp::Named("H_beta") = last_blocks.H_beta,
    Rcpp::Named("G_phi") = last_blocks.G_phi,
    Rcpp::Named("H_phi") = last_blocks.H_phi,
    Rcpp::Named("G_rho") = last_blocks.G_rho,
    Rcpp::Named("H_rho") = last_blocks.H_rho,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("final_error") = err
  );
}

/***** DETERMINISTIC HIERARCHICAL SANDWICH LAYER *******************************
 Supports:
 - family: "binomial", "gaussian"
 - corstr: "independence", "simple-exchangeable", "nested-exchangeable"

 This rewrites:
 - binomial_hier_sandwich()
 - gaussian_hier_sandwich()
 - meat_computation() usage for deterministic hierarchical fits

 It recomputes clusterwise H/G blocks at the final parameter values and then
 assembles the full sandwich exactly in the style of the old deterministic code.
 *******************************************************************************/

struct ngee_hier_sandwich_blocks {
  arma::cube H1;
  arma::cube G1;
  arma::cube H1_5;
  arma::cube G1_5;
  arma::cube H2;
  arma::cube G2;
  arma::cube B;
  arma::cube D;
  arma::cube E;
};

/*
 Compute the empirical sandwich meat from clusterwise stacked score vectors.

 For se_adjust = "FG", the Fay-Graubard small-sample correction is applied
 clusterwise using the diagonal of H_i * Info.
 */
static inline arma::mat ngee_meat_from_cubes(const arma::cube& G,
                                             const arma::cube& H,
                                             const arma::mat& Info,
                                             const std::string& se_adjust,
                                             const double fg_cap = 0.85) {
  const arma::uword d = G.n_rows;
  const arma::uword I = G.n_slices;
  arma::mat G_outersum(d, d, arma::fill::zeros);

  if (se_adjust == "unadjusted") {
    for (arma::uword i = 0; i < I; ++i) {
      G_outersum += G.slice(i) * G.slice(i).t();
    }
  } else if (se_adjust == "FG") {
    for (arma::uword i = 0; i < I; ++i) {
      arma::mat Q = H.slice(i) * Info;
      arma::colvec scale(d, arma::fill::ones);

      for (arma::uword j = 0; j < d; ++j) {
        const double qjj = Q(j, j);
        const double cap = std::min(fg_cap, qjj);
        const double denom = std::max(1.0 - cap, 1e-12);
        scale[j] = std::pow(denom, -0.5);
      }

      arma::colvec G_tilde = scale % G.slice(i);
      G_outersum += G_tilde * G_tilde.t();
    }
  } else {
    Rcpp::stop("se_adjust must be 'unadjusted' or 'FG'.");
  }

  return G_outersum;
}

/*
 Clusterwise sandwich blocks for binomial independence/simple exchangeable.

 The D block contains derivatives of the second-order score with respect to the
 mean parameters. B and E are not needed in this binomial one-parameter case.
 */
static inline ngee_hier_sandwich_blocks ngee_blocks_binomial_hier_q1(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  const arma::uword q = rho.n_elem;
  const arma::uword I = idx.n_cluster;

  ngee_hier_sandwich_blocks out;
  out.H1.zeros(p, p, I);
  out.G1.zeros(p, 1, I);
  out.H1_5.zeros(1, 1, I);
  out.G1_5.zeros(1, 1, I);
  out.H2.zeros(q, q, I);
  out.G2.zeros(q, 1, I);
  out.B.zeros(1, p, I); out.B.fill(1e-15);
  out.D.zeros(q, p, I);
  out.E.zeros(q, 1, I); out.E.fill(1e-15);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec v = mu % (1.0 - mu);
  const arma::colvec vsqrt = arma::sqrt(v);
  const arma::colvec dvdm = 1.0 - 2.0 * mu;

  const arma::colvec U_sqrt = vsqrt;
  const arma::colvec std_resid = resid / U_sqrt;
  const arma::mat std_design_mat = design_mat.each_col() % U_sqrt;

  const double rho0 = 1.0 - rho[0];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);
    const double c = rho[0] / (rho0 * (rho0 + rho[0] * static_cast<double>(csize)));

    out.G1.slice(i) = Xtilde.t() * Etilde / rho0 - c * Xtilde_sum.t() * Etilde_sum;
    out.H1.slice(i) = Xtilde.t() * Xtilde / rho0 - c * Xtilde_sum.t() * Xtilde_sum;

    out.G2(0, 0, i) =
      (Etilde_sum * Etilde_sum - arma::dot(Etilde, Etilde)) / 2.0
    - static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) * rho[0] / 2.0;

    out.H2(0, 0, i) = static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;

    arma::mat Xmat = design_mat.rows(cstart, cend);
    arma::colvec Evec = resid.rows(cstart, cend);
    arma::colvec vsqrtsub = vsqrt.rows(cstart, cend);
    arma::colvec dvdmsub = dvdm.rows(cstart, cend);

    arma::mat XE_portion_1 =
      Xmat.each_col() % vsqrtsub +
      Xmat.each_col() % (Evec % dvdmsub / vsqrtsub) / 2.0;
    arma::colvec XE_portion_2 = Evec / vsqrtsub;

    const arma::rowvec temp_vec_XE_portion_1 = arma::sum(XE_portion_1, 0);
    const double temp_double_XE_portion_2 = arma::sum(XE_portion_2);

    out.D.slice(i).row(0) =
      temp_vec_XE_portion_1 * temp_double_XE_portion_2 -
      XE_portion_2.t() * XE_portion_1;
  }

  return out;
}

/*
 Clusterwise sandwich blocks for Gaussian independence/simple exchangeable.

 B links the phi equation to beta, and E is not needed when there is only one
 correlation parameter.
 */
static inline ngee_hier_sandwich_blocks ngee_blocks_gaussian_hier_q1(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  const arma::uword q = rho.n_elem;
  const arma::uword I = idx.n_cluster;

  ngee_hier_sandwich_blocks out;
  out.H1.zeros(p, p, I);
  out.G1.zeros(p, 1, I);
  out.H1_5.zeros(1, 1, I);
  out.G1_5.zeros(1, 1, I);
  out.H2.zeros(q, q, I);
  out.G2.zeros(q, 1, I);
  out.B.zeros(1, p, I);
  out.D.zeros(q, p, I);
  out.E.zeros(q, 1, I);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;

  const double sqrt_phi = std::sqrt(phi);
  const arma::colvec std_resid = resid / sqrt_phi;
  const arma::mat std_design_mat = design_mat / sqrt_phi;

  const double rho0 = 1.0 - rho[0];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);
    const double c = rho[0] / (rho0 * (rho0 + rho[0] * static_cast<double>(csize)));

    out.G1.slice(i) = Xtilde.t() * Etilde / rho0 - c * Xtilde_sum.t() * Etilde_sum;
    out.H1.slice(i) = Xtilde.t() * Xtilde / rho0 - c * Xtilde_sum.t() * Xtilde_sum;

    out.G1_5(0, 0, i) = phi * (arma::dot(Etilde, Etilde) - static_cast<double>(csize));
    out.H1_5(0, 0, i) = static_cast<double>(csize);

    out.G2(0, 0, i) =
      (Etilde_sum * Etilde_sum - arma::dot(Etilde, Etilde)) / 2.0
    - static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) * rho[0] / 2.0;

    out.H2(0, 0, i) = static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;

    const auto Xmat = design_mat.rows(cstart, cend);
    const auto Evec = resid.rows(cstart, cend);

    out.B.slice(i) = 2.0 * Evec.t() * Xmat;
    out.D.slice(i) = (arma::sum(Xmat, 0) * arma::sum(Evec) - Evec.t() * Xmat) / phi;
    out.E.slice(i) =
      (std::pow(arma::sum(Evec), 2.0) - arma::dot(Evec, Evec)) / (2.0 * phi * phi);
  }

  return out;
}

/*
 Clusterwise sandwich blocks for the binomial nested-exchangeable model.

 Row 1 of D corresponds to the within-subgroup correlation block and row 2 to
 between-subgroup correlation.
 */
static inline ngee_hier_sandwich_blocks ngee_blocks_binomial_hier_q2(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  const arma::uword q = rho.n_elem;
  const arma::uword I = idx.n_cluster;

  ngee_hier_sandwich_blocks out;
  out.H1.zeros(p, p, I);
  out.G1.zeros(p, 1, I);
  out.H1_5.zeros(1, 1, I);
  out.G1_5.zeros(1, 1, I);
  out.H2.zeros(q, q, I);
  out.G2.zeros(q, 1, I);
  out.B.zeros(1, p, I); out.B.fill(1e-15);
  out.D.zeros(q, p, I);
  out.E.zeros(q, 1, I); out.E.fill(1e-15);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec v = mu % (1.0 - mu);
  const arma::colvec vsqrt = arma::sqrt(v);
  const arma::colvec dvdm = 1.0 - 2.0 * mu;

  const arma::colvec std_resid = resid / vsqrt;
  const arma::mat std_design_mat = design_mat.each_col() % vsqrt;

  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword nsub   = idx.cluster_n_sub[i];

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    arma::colvec v_rep(csize, arma::fill::zeros);
    arma::uword cursor = 0;
    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword len = idx.sub_size[s0 + s];
      const double vij = 1.0 / rho0 - delta / (rho0 * (rho0 / static_cast<double>(len) + delta));
      v_rep.subvec(cursor, cursor + len - 1).fill(vij);
      cursor += len;
    }

    const double c = ngee_nested_c_value(rho[1], arma::sum(v_rep));
    double inner_pairs = 0.0;

    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword subid = s0 + s;
      const arma::uword len = idx.sub_size[subid];
      const arma::uword local_start = idx.sub_start[subid] - cstart;
      const arma::uword local_end   = local_start + len - 1;

      const auto Xsub = Xtilde.rows(local_start, local_end);
      const auto Esub = Etilde.rows(local_start, local_end);

      const arma::rowvec Xsub_sum = arma::sum(Xsub, 0);
      const double Esub_sum = arma::sum(Esub);
      const double subcorr = delta / (rho0 * (rho0 + delta * static_cast<double>(len)));

      out.G1.slice(i) += -Xsub_sum.t() * Esub_sum * subcorr;
      out.H1.slice(i) += -Xsub_sum.t() * Xsub_sum * subcorr;

      const double sumsq_sub = arma::dot(Esub, Esub);
      const double a = sumsq_sub - static_cast<double>(len);
      const double a0 = (Esub_sum * Esub_sum - sumsq_sub) / 2.0;

      out.G2(0, 0, i) += a0 - static_cast<double>(len) * (static_cast<double>(len) - 1.0) * rho[0] / 2.0;
      out.G2(1, 0, i) -= (a + static_cast<double>(len)) / 2.0 + a0;

      inner_pairs += static_cast<double>(len) * (static_cast<double>(len) - 1.0) / 2.0;
    }

    const double total_pairs = static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;

    out.G1.slice(i) += Xtilde.t() * Etilde / rho0 - c * (Xtilde.t() * v_rep) * arma::as_scalar(v_rep.t() * Etilde);
    out.H1.slice(i) += Xtilde.t() * Xtilde / rho0 - c * (Xtilde.t() * v_rep) * (v_rep.t() * Xtilde);

    out.G2(1, 0, i) += std::pow(arma::sum(Etilde), 2.0) / 2.0 - (total_pairs - inner_pairs) * rho[1];
    out.H2.slice(i) = arma::zeros<arma::mat>(2, 2);
    out.H2(0, 0, i) = inner_pairs;
    out.H2(1, 1, i) = total_pairs - inner_pairs;

    arma::mat Xmat = design_mat.rows(cstart, cend);
    arma::colvec Evec = resid.rows(cstart, cend);
    arma::colvec vsqrtsub = vsqrt.rows(cstart, cend);
    arma::colvec dvdmsub = dvdm.rows(cstart, cend);

    arma::mat XE_portion_1 =
      Xmat.each_col() % vsqrtsub +
      Xmat.each_col() % (Evec % dvdmsub / vsqrtsub) / 2.0;
    arma::colvec XE_portion_2 = Evec / vsqrtsub;

    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword subid = s0 + s;
      const arma::uword len = idx.sub_size[subid];
      const arma::uword local_start = idx.sub_start[subid] - cstart;
      const arma::uword local_end   = local_start + len - 1;

      const auto XE1_sub = XE_portion_1.rows(local_start, local_end);
      const auto XE2_sub = XE_portion_2.rows(local_start, local_end);

      const arma::rowvec XE1_sum = arma::sum(XE1_sub, 0);
      const double XE2_sum = arma::sum(XE2_sub);

      out.D.slice(i).row(0) += XE1_sum * XE2_sum - XE2_sub.t() * XE1_sub;
    }

    const arma::rowvec XE1_sum_all = arma::sum(XE_portion_1, 0);
    const double XE2_sum_all = arma::sum(XE_portion_2);
    out.D.slice(i).row(1) =
      (XE1_sum_all * XE2_sum_all - XE_portion_2.t() * XE_portion_1) -
      out.D.slice(i).row(0);
  }

  return out;
}

/*
 Gaussian nested-exchangeable sandwich blocks.

 B, D, and E encode the cross-derivative pieces needed to assemble the full
 stacked sandwich for (beta, phi, rho).
 */
static inline ngee_hier_sandwich_blocks ngee_blocks_gaussian_hier_q2(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx) {

  const arma::uword p = beta.n_elem;
  const arma::uword q = rho.n_elem;
  const arma::uword I = idx.n_cluster;

  ngee_hier_sandwich_blocks out;
  out.H1.zeros(p, p, I);
  out.G1.zeros(p, 1, I);
  out.H1_5.zeros(1, 1, I);
  out.G1_5.zeros(1, 1, I);
  out.H2.zeros(q, q, I);
  out.G2.zeros(q, 1, I);
  out.B.zeros(1, p, I);
  out.D.zeros(q, p, I);
  out.E.zeros(q, 1, I);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;

  const double sqrt_phi = std::sqrt(phi);
  const arma::colvec std_resid = resid / sqrt_phi;
  const arma::mat std_design_mat = design_mat / sqrt_phi;

  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword nsub   = idx.cluster_n_sub[i];

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    arma::colvec v_rep(csize, arma::fill::zeros);
    arma::uword cursor = 0;
    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword len = idx.sub_size[s0 + s];
      const double vij = 1.0 / rho0 - delta / (rho0 * (rho0 / static_cast<double>(len) + delta));
      v_rep.subvec(cursor, cursor + len - 1).fill(vij);
      cursor += len;
    }

    const double c = ngee_nested_c_value(rho[1], arma::sum(v_rep));
    double inner_pairs = 0.0;

    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword subid = s0 + s;
      const arma::uword len = idx.sub_size[subid];
      const arma::uword local_start = idx.sub_start[subid] - cstart;
      const arma::uword local_end   = local_start + len - 1;

      const auto Xsub = Xtilde.rows(local_start, local_end);
      const auto Esub = Etilde.rows(local_start, local_end);

      const arma::rowvec Xsub_sum = arma::sum(Xsub, 0);
      const double Esub_sum = arma::sum(Esub);
      const double sumsq_sub = arma::dot(Esub, Esub);
      const double subcorr = delta / (rho0 * (rho0 + delta * static_cast<double>(len)));

      out.G1.slice(i) += -Xsub_sum.t() * Esub_sum * subcorr;
      out.H1.slice(i) += -Xsub_sum.t() * Xsub_sum * subcorr;

      out.G1_5(0, 0, i) += phi * (sumsq_sub - static_cast<double>(len));
      out.H1_5(0, 0, i) += static_cast<double>(len);

      const double a0 = (Esub_sum * Esub_sum - sumsq_sub) / 2.0;
      out.G2(0, 0, i) += a0 - static_cast<double>(len) * (static_cast<double>(len) - 1.0) * rho[0] / 2.0;
      out.G2(1, 0, i) -= (sumsq_sub + a0);

      inner_pairs += static_cast<double>(len) * (static_cast<double>(len) - 1.0) / 2.0;
    }

    const double total_pairs = static_cast<double>(csize) * (static_cast<double>(csize) - 1.0) / 2.0;

    out.G1.slice(i) += Xtilde.t() * Etilde / rho0 - c * (Xtilde.t() * v_rep) * arma::as_scalar(v_rep.t() * Etilde);
    out.H1.slice(i) += Xtilde.t() * Xtilde / rho0 - c * (Xtilde.t() * v_rep) * (v_rep.t() * Xtilde);

    out.G2(1, 0, i) += std::pow(arma::sum(Etilde), 2.0) / 2.0 - (total_pairs - inner_pairs) * rho[1];
    out.H2.slice(i) = arma::zeros<arma::mat>(2, 2);
    out.H2(0, 0, i) = inner_pairs;
    out.H2(1, 1, i) = total_pairs - inner_pairs;

    const auto Xmat = design_mat.rows(cstart, cend);
    const auto Evec = resid.rows(cstart, cend);

    out.B.slice(i) = 2.0 * Evec.t() * Xmat;

    for (arma::uword s = 0; s < nsub; ++s) {
      const arma::uword subid = s0 + s;
      const arma::uword len = idx.sub_size[subid];
      const arma::uword local_start = idx.sub_start[subid] - cstart;
      const arma::uword local_end   = local_start + len - 1;

      const auto Xsub = Xmat.rows(local_start, local_end);
      const auto Esub = Evec.rows(local_start, local_end);

      const arma::rowvec Xsub_sum = arma::sum(Xsub, 0);
      const double Esub_sum = arma::sum(Esub);

      out.D.slice(i).row(0) += (Xsub_sum * Esub_sum - Esub.t() * Xsub) / phi;
      out.E(0, 0, i) += (Esub_sum * Esub_sum - arma::dot(Esub, Esub)) / (2.0 * phi * phi);
    }

    const arma::rowvec Xsum_all = arma::sum(Xmat, 0);
    const double Esum_all = arma::sum(Evec);

    out.D.slice(i).row(1) =
      (Xsum_all * Esum_all - Evec.t() * Xmat) / phi -
      out.D.slice(i).row(0);

    out.E(1, 0, i) =
      (Esum_all * Esum_all - arma::dot(Evec, Evec)) / (2.0 * phi * phi) -
      out.E(0, 0, i);
  }

  return out;
}

static inline ngee_hier_sandwich_blocks ngee_hier_blocks_dispatch(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_index_view& idx,
    const std::string& family,
    const std::string& corstr) {

  if (family == "binomial") {
    if (corstr == "independence" || corstr == "simple-exchangeable") {
      return ngee_blocks_binomial_hier_q1(beta, rho, outcome, design_mat, idx);
    } else if (corstr == "nested-exchangeable") {
      return ngee_blocks_binomial_hier_q2(beta, rho, outcome, design_mat, idx);
    }
  } else if (family == "gaussian") {
    if (corstr == "independence" || corstr == "simple-exchangeable") {
      return ngee_blocks_gaussian_hier_q1(beta, phi, rho, outcome, design_mat, idx);
    } else if (corstr == "nested-exchangeable") {
      return ngee_blocks_gaussian_hier_q2(beta, phi, rho, outcome, design_mat, idx);
    }
  }

  Rcpp::stop("Unsupported family/corstr combination in hierarchical sandwich dispatch.");
  return ngee_hier_sandwich_blocks();
}

// [[Rcpp::export]]
Rcpp::List ngee_hier_sandwich_blocks_cpp(const arma::colvec& beta,
                                         const double phi,
                                         const arma::colvec& rho,
                                         const arma::colvec& outcome,
                                         const arma::mat& design_mat,
                                         const Rcpp::List& index_data,
                                         const std::string& family,
                                         const std::string& corstr,
                                         const double rho_eps = 1e-4) {
  const ngee_hier_index_view idx = ngee_get_hier_index_view(index_data);

  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta.n_elem != design_mat.n_cols) Rcpp::stop("beta has wrong length.");

  arma::colvec rho_work = rho;
  if (corstr == "simple-exchangeable") {
    ngee_stabilize_simple_rho(rho_work, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    ngee_stabilize_nested_rho(rho_work, idx, rho_eps);
  }

  ngee_hier_sandwich_blocks out = ngee_hier_blocks_dispatch(
    beta, phi, rho_work, outcome, design_mat, idx, family, corstr
  );

  return Rcpp::List::create(
    Rcpp::Named("rho_used") = rho_work,
    Rcpp::Named("H1") = out.H1,
    Rcpp::Named("G1") = out.G1,
    Rcpp::Named("H1_5") = out.H1_5,
    Rcpp::Named("G1_5") = out.G1_5,
    Rcpp::Named("H2") = out.H2,
    Rcpp::Named("G2") = out.G2,
    Rcpp::Named("B") = out.B,
    Rcpp::Named("D") = out.D,
    Rcpp::Named("E") = out.E
  );
}

/*
 Deterministic robust covariance estimator for hierarchical models.

 This reconstructs the stacked clusterwise score/Hessian blocks at the final
 parameter values and then returns the model-based bread and robust sandwich
 covariance for the active parameter vector.
 */
// [[Rcpp::export]]
Rcpp::List ngee_hier_sandwich_det_cpp(const arma::colvec& beta,
                                      const double phi,
                                      const arma::colvec& rho,
                                      const arma::colvec& outcome,
                                      const arma::mat& design_mat,
                                      const Rcpp::List& index_data,
                                      const std::string& family,
                                      const std::string& corstr,
                                      const std::string& se_adjust = "unadjusted",
                                      const double fg_cap = 0.85,
                                      const double rho_eps = 1e-4) {
  const ngee_hier_index_view idx = ngee_get_hier_index_view(index_data);
  const arma::uword p = beta.n_elem;
  const arma::uword q = rho.n_elem;
  const arma::uword I = idx.n_cluster;

  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta.n_elem != design_mat.n_cols) Rcpp::stop("beta has wrong length.");

  arma::colvec rho_work = rho;
  if (corstr == "simple-exchangeable") {
    ngee_stabilize_simple_rho(rho_work, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    ngee_stabilize_nested_rho(rho_work, idx, rho_eps);
  }

  ngee_hier_sandwich_blocks blocks = ngee_hier_blocks_dispatch(
    beta, phi, rho_work, outcome, design_mat, idx, family, corstr
  );

  arma::mat Info1 = ngee_safe_inv_mat(arma::sum(blocks.H1, 2));
  arma::mat Info1_5(1, 1, arma::fill::zeros);
  arma::mat Info2;
  if (q > 0) {
    Info2 = ngee_safe_inv_mat(arma::sum(blocks.H2, 2));
  } else {
    Info2.zeros(0, 0);
  }

  if (family == "gaussian") {
    const double Hphi_sum = arma::accu(blocks.H1_5);
    if (Hphi_sum <= 0.0) Rcpp::stop("Non-positive H1_5 sum encountered.");
    Info1_5(0, 0) = 1.0 / Hphi_sum;
  }

  arma::mat Info;
  arma::mat G_outersum;
  arma::mat var_sandwich;

  if (family == "binomial") {
    if (corstr == "independence") {
      Info = ngee_safe_inv_mat(arma::sum(blocks.H1, 2));
      G_outersum = ngee_meat_from_cubes(blocks.G1, blocks.H1, Info, se_adjust, fg_cap);
      var_sandwich = Info * G_outersum * Info.t();
    } else {
      const arma::uword d = p + q;
      arma::cube H(d, d, I, arma::fill::zeros);
      arma::cube G(d, 1, I, arma::fill::zeros);

      for (arma::uword i = 0; i < I; ++i) {
        H.slice(i).submat(0, 0, p - 1, p - 1) = blocks.H1.slice(i);
        H.slice(i).submat(p, 0, p + q - 1, p - 1) = blocks.D.slice(i);
        H.slice(i).submat(p, p, p + q - 1, p + q - 1) = blocks.H2.slice(i);

        G.slice(i).submat(0, 0, p - 1, 0) = blocks.G1.slice(i);
        G.slice(i).submat(p, 0, p + q - 1, 0) = blocks.G2.slice(i);
      }

      Info = ngee_safe_inv_mat(arma::sum(H, 2));
      G_outersum = ngee_meat_from_cubes(G, H, Info, se_adjust, fg_cap);
      var_sandwich = Info * G_outersum * Info.t();
    }
  } else if (family == "gaussian") {
    if (corstr == "independence") {
      const arma::uword d = p + 1;
      arma::cube H(d, d, I, arma::fill::zeros);
      arma::cube G(d, 1, I, arma::fill::zeros);

      for (arma::uword i = 0; i < I; ++i) {
        H.slice(i).submat(0, 0, p - 1, p - 1) = blocks.H1.slice(i);
        H.slice(i).submat(p, 0, p, p - 1) = blocks.B.slice(i);
        H.slice(i)(p, p) = blocks.H1_5(0, 0, i);

        G.slice(i).submat(0, 0, p - 1, 0) = blocks.G1.slice(i);
        G.slice(i)(p, 0) = blocks.G1_5(0, 0, i);
      }

      Info = ngee_safe_inv_mat(arma::sum(H, 2));
      G_outersum = ngee_meat_from_cubes(G, H, Info, se_adjust, fg_cap);
      var_sandwich = Info * G_outersum * Info.t();
    } else {
      const arma::uword d = p + 1 + q;
      arma::cube H(d, d, I, arma::fill::zeros);
      arma::cube G(d, 1, I, arma::fill::zeros);

      for (arma::uword i = 0; i < I; ++i) {
        H.slice(i).submat(0, 0, p - 1, p - 1) = blocks.H1.slice(i);
        H.slice(i).submat(p, 0, p, p - 1) = blocks.B.slice(i);
        H.slice(i)(p, p) = blocks.H1_5(0, 0, i);
        H.slice(i).submat(p + 1, 0, p + q, p - 1) = blocks.D.slice(i);
        H.slice(i).submat(p + 1, p, p + q, p) = blocks.E.slice(i);
        H.slice(i).submat(p + 1, p + 1, p + q, p + q) = blocks.H2.slice(i);

        G.slice(i).submat(0, 0, p - 1, 0) = blocks.G1.slice(i);
        G.slice(i)(p, 0) = blocks.G1_5(0, 0, i);
        G.slice(i).submat(p + 1, 0, p + q, 0) = blocks.G2.slice(i);
      }

      Info = ngee_safe_inv_mat(arma::sum(H, 2));
      G_outersum = ngee_meat_from_cubes(G, H, Info, se_adjust, fg_cap);
      var_sandwich = Info * G_outersum * Info.t();
    }
  } else {
    Rcpp::stop("family must be 'binomial' or 'gaussian'.");
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho_work,
    Rcpp::Named("Info1") = Info1,
    Rcpp::Named("Info1_5") = Info1_5,
    Rcpp::Named("Info2") = Info2,
    Rcpp::Named("Info") = Info,
    Rcpp::Named("G_outersum") = G_outersum,
    Rcpp::Named("var_sandwich") = var_sandwich,
    Rcpp::Named("H1") = blocks.H1,
    Rcpp::Named("G1") = blocks.G1,
    Rcpp::Named("H1_5") = blocks.H1_5,
    Rcpp::Named("G1_5") = blocks.G1_5,
    Rcpp::Named("H2") = blocks.H2,
    Rcpp::Named("G2") = blocks.G2,
    Rcpp::Named("B") = blocks.B,
    Rcpp::Named("D") = blocks.D,
    Rcpp::Named("E") = blocks.E
  );
}

/***** DETERMINISTIC HIERARCHICAL FIT + SANDWICH WRAPPER ***********************
 Wires together:
 - ngee_fit_det_hier_cpp()
 - ngee_hier_sandwich_det_cpp()

 Returns:
 - point estimates
 - sandwich variance
 - active parameter vector and SEs
 *******************************************************************************/

static inline arma::uword ngee_active_q_hier(const std::string& corstr) {
  if (corstr == "independence") return 0;
  if (corstr == "simple-exchangeable") return 1;
  if (corstr == "nested-exchangeable") return 2;
  Rcpp::stop("Unsupported corstr in ngee_active_q_hier().");
  return 0;
}

static inline arma::colvec ngee_diag_se(const arma::mat& V) {
  arma::colvec out = V.diag();
  out.transform([](double x) { return std::sqrt(std::max(x, 0.0)); });
  return out;
}

static inline arma::colvec ngee_pack_active_coef_hier(const arma::colvec& beta,
                                                      const double phi,
                                                      const arma::colvec& rho,
                                                      const std::string& family,
                                                      const std::string& corstr) {
  const arma::uword p = beta.n_elem;
  const arma::uword q_active = ngee_active_q_hier(corstr);
  const bool include_phi = (family == "gaussian");

  arma::colvec out(p + static_cast<arma::uword>(include_phi) + q_active,
                   arma::fill::zeros);

  arma::uword pos = 0;
  out.subvec(pos, pos + p - 1) = beta;
  pos += p;

  if (include_phi) {
    out[pos] = phi;
    ++pos;
  }

  if (q_active > 0) {
    out.subvec(pos, pos + q_active - 1) = rho.subvec(0, q_active - 1);
  }

  return out;
}

static inline Rcpp::CharacterVector ngee_active_param_names_hier(const arma::uword p,
                                                                 const std::string& family,
                                                                 const std::string& corstr) {
  const arma::uword q_active = ngee_active_q_hier(corstr);
  const bool include_phi = (family == "gaussian");
  const arma::uword d = p + static_cast<arma::uword>(include_phi) + q_active;

  Rcpp::CharacterVector out(d);
  arma::uword pos = 0;

  for (arma::uword j = 0; j < p; ++j) {
    out[pos] = "beta" + std::to_string(j + 1);
    ++pos;
  }

  if (include_phi) {
    out[pos] = "phi";
    ++pos;
  }

  for (arma::uword j = 0; j < q_active; ++j) {
    out[pos] = "rho" + std::to_string(j + 1);
    ++pos;
  }

  return out;
}

/*
 High-level deterministic hierarchical fit that returns point estimates, the
 robust covariance matrix, and blockwise standard errors.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_det_hier_se_cpp(const arma::colvec& outcome,
                                    const arma::mat& design_mat,
                                    const Rcpp::List& index_data,
                                    const std::string& family,
                                    const std::string& corstr,
                                    arma::colvec beta0,
                                    double phi0,
                                    arma::colvec rho0,
                                    const std::string& se_adjust = "unadjusted",
                                    const double tol = 1e-8,
                                    const int maxit = 200,
                                    const double phi_min = 1e-8,
                                    const double fg_cap = 0.85,
                                    const double rho_eps = 1e-4) {

  Rcpp::List fit = ngee_fit_det_hier_cpp(
    outcome,
    design_mat,
    index_data,
    family,
    corstr,
    beta0,
    phi0,
    rho0,
    tol,
    maxit,
    phi_min,
    rho_eps
  );

  const arma::colvec beta = Rcpp::as<arma::colvec>(fit["beta"]);
  const double phi = Rcpp::as<double>(fit["phi"]);
  const arma::colvec rho = Rcpp::as<arma::colvec>(fit["rho"]);
  const int iter = Rcpp::as<int>(fit["iter"]);
  const bool converged = Rcpp::as<bool>(fit["converged"]);
  const double final_error = Rcpp::as<double>(fit["final_error"]);

  Rcpp::List sand = ngee_hier_sandwich_det_cpp(
    beta,
    phi,
    rho,
    outcome,
    design_mat,
    index_data,
    family,
    corstr,
    se_adjust,
    fg_cap,
    rho_eps
  );

  const arma::mat var_sandwich = Rcpp::as<arma::mat>(sand["var_sandwich"]);
  const arma::mat Info = Rcpp::as<arma::mat>(sand["Info"]);
  const arma::colvec se = ngee_diag_se(var_sandwich);
  const arma::colvec coef = ngee_pack_active_coef_hier(beta, phi, rho, family, corstr);

  if (coef.n_elem != se.n_elem) {
    Rcpp::stop("Internal dimension mismatch: coef and sandwich SE lengths differ.");
  }

  arma::colvec z = coef / se;
  for (arma::uword j = 0; j < z.n_elem; ++j) {
    if (!std::isfinite(z[j])) z[j] = NA_REAL;
  }

  const arma::uword p = beta.n_elem;
  const arma::uword q_active = ngee_active_q_hier(corstr);
  const bool include_phi = (family == "gaussian");

  arma::colvec beta_se = se.subvec(0, p - 1);

  arma::colvec phi_se;
  phi_se.zeros(include_phi ? 1 : 0);
  if (include_phi) {
    phi_se[0] = se[p];
  }

  arma::colvec rho_se;
  rho_se.zeros(q_active);
  if (q_active > 0) {
    const arma::uword rho_start = p + static_cast<arma::uword>(include_phi);
    rho_se = se.subvec(rho_start, rho_start + q_active - 1);
  }

  return Rcpp::List::create(
    Rcpp::Named("family") = family,
    Rcpp::Named("corstr") = corstr,
    Rcpp::Named("se_adjust") = se_adjust,
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("beta_se") = beta_se,
    Rcpp::Named("phi_se") = phi_se,
    Rcpp::Named("rho_se") = rho_se,
    Rcpp::Named("coef") = coef,
    Rcpp::Named("se") = se,
    Rcpp::Named("z") = z,
    Rcpp::Named("param_names") = ngee_active_param_names_hier(p, family, corstr),
    Rcpp::Named("var_sandwich") = var_sandwich,
    Rcpp::Named("Info") = Info,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("final_error") = final_error
  );
}

/***** DETERMINISTIC BLOCK-EXCHANGEABLE (TSCS) **********************************
 Supports:
 - corstr: "block-exchangeable"
 - family: "binomial", "gaussian"

 Assumptions:
 - index_data comes from ngee_precompute_index_cpp(..., "block-exchangeable")
 - within each cluster, rows are sorted by period
 - each period has the same number of individuals within that cluster
 - row order aligns repeated individuals across periods
 *******************************************************************************/

/*
 Lightweight C++ view of the block-exchangeable / TSCS index object.

 cluster_subject_n stores the number of aligned subjects per period inside each
 cluster. The block working covariance formulas assume this balanced layout.
 */
struct ngee_tscs_index_view {
  std::string corstr;
  arma::uvec cluster_start;
  arma::uvec cluster_size;
  arma::uvec cluster_sub_start;
  arma::uvec cluster_n_sub;
  arma::uvec sub_start;
  arma::uvec sub_size;
  arma::uvec cluster_subject_n;
  Rcpp::LogicalVector cluster_balanced;
  arma::uword n_cluster;
  arma::uword n_obs;
};

static inline ngee_tscs_index_view ngee_get_tscs_index_view(const Rcpp::List& index_data) {
  ngee_tscs_index_view idx;
  idx.corstr = Rcpp::as<std::string>(index_data["corstr"]);
  idx.cluster_start = Rcpp::as<arma::uvec>(index_data["cluster_start"]);
  idx.cluster_size = Rcpp::as<arma::uvec>(index_data["cluster_size"]);
  idx.cluster_sub_start = Rcpp::as<arma::uvec>(index_data["cluster_sub_start"]);
  idx.cluster_n_sub = Rcpp::as<arma::uvec>(index_data["cluster_n_sub"]);
  idx.sub_start = Rcpp::as<arma::uvec>(index_data["sub_start"]);
  idx.sub_size = Rcpp::as<arma::uvec>(index_data["sub_size"]);
  idx.n_cluster = idx.cluster_start.n_elem;
  idx.n_obs = Rcpp::as<int>(index_data["n_obs"]);

  if (index_data.containsElementNamed("cluster_subject_n")) {
    idx.cluster_subject_n = Rcpp::as<arma::uvec>(index_data["cluster_subject_n"]);
  } else {
    idx.cluster_subject_n.set_size(idx.n_cluster);
    for (arma::uword i = 0; i < idx.n_cluster; ++i) {
      if (idx.cluster_n_sub[i] == 0) {
        idx.cluster_subject_n[i] = 0;
      } else {
        idx.cluster_subject_n[i] = idx.sub_size[idx.cluster_sub_start[i]];
      }
    }
  }

  if (index_data.containsElementNamed("cluster_balanced")) {
    idx.cluster_balanced = Rcpp::as<Rcpp::LogicalVector>(index_data["cluster_balanced"]);
  } else {
    Rcpp::LogicalVector cb(idx.n_cluster);
    for (arma::uword i = 0; i < idx.n_cluster; ++i) {
      bool ok = true;
      if (idx.cluster_n_sub[i] > 0) {
        const arma::uword s0 = idx.cluster_sub_start[i];
        const arma::uword ref = idx.sub_size[s0];
        for (arma::uword t = 1; t < idx.cluster_n_sub[i]; ++t) {
          if (idx.sub_size[s0 + t] != ref) {
            ok = false;
            break;
          }
        }
      }
      cb[i] = ok;
    }
    idx.cluster_balanced = cb;
  }

  return idx;
}

static inline bool ngee_block_rho_feasible_one(const arma::colvec& rho,
                                               const double J,
                                               const double T,
                                               const double rho_eps) {
  if (rho.n_elem != 3) return false;
  if (arma::max(arma::abs(rho)) >= 1.0 - rho_eps) return false;

  const double lambda1 = 1.0 - rho[0] + rho[1] - rho[2];
  const double lambda2 = 1.0 - rho[0] - (T - 1.0) * (rho[1] - rho[2]);
  const double lambda3 = 1.0 + (J - 1.0) * (rho[0] - rho[1]) - rho[2];
  const double lambda4 = 1.0 + (J - 1.0) * rho[0] + (T - 1.0) * (J - 1.0) * rho[1] + (T - 1.0) * rho[2];

  return (lambda1 > rho_eps &&
          lambda2 > rho_eps &&
          lambda3 > rho_eps &&
          lambda4 > rho_eps);
}

/*
 Numerical stabilization for the three block-exchangeable correlations.

 The four lambda quantities are the eigenvalue-style factors that appear in the
 analytical inverse of the block-exchangeable working correlation matrix. The
 routine shrinks rho toward zero until all clusters satisfy the positivity
 conditions implied by those factors.
 */
static inline void ngee_stabilize_block_rho(arma::colvec& rho,
                                            const ngee_tscs_index_view& idx,
                                            const double rho_eps) {
  if (rho.n_elem != 3) Rcpp::stop("block-exchangeable requires rho0 of length 3.");

  for (arma::uword k = 0; k < rho.n_elem; ++k) {
    if (!std::isfinite(rho[k])) rho[k] = 0.0;
    rho[k] = std::max(-(1.0 - rho_eps), std::min(1.0 - rho_eps, rho[k]));
  }

  for (int it = 0; it < 100; ++it) {
    bool ok = true;
    for (arma::uword i = 0; i < idx.n_cluster; ++i) {
      const double J = static_cast<double>(idx.cluster_subject_n[i]);
      const double T = static_cast<double>(idx.cluster_n_sub[i]);
      if (!ngee_block_rho_feasible_one(rho, J, T, rho_eps)) {
        ok = false;
        break;
      }
    }
    if (ok) return;
    rho *= 0.5;
  }

  rho.zeros();
}

/*
 Compute the lambda factors and the scalar c used in the analytical inverse of
 the block-exchangeable working correlation matrix.
 */
static inline void ngee_block_constants(const arma::colvec& rho,
                                        const double J,
                                        const double T,
                                        double& lambda1,
                                        double& lambda2,
                                        double& lambda3,
                                        double& lambda4,
                                        double& c) {
  lambda1 = 1.0 - rho[0] + rho[1] - rho[2];
  lambda2 = 1.0 - rho[0] - (T - 1.0) * (rho[1] - rho[2]);
  lambda3 = 1.0 + (J - 1.0) * (rho[0] - rho[1]) - rho[2];
  lambda4 = 1.0 + (J - 1.0) * rho[0] + (T - 1.0) * (J - 1.0) * rho[1] + (T - 1.0) * rho[2];
  c = (rho[2] - rho[1]) * (rho[0] - rho[1]) / (lambda1 * lambda2 * lambda3) +
    (rho[2] * rho[0] - rho[1]) / (lambda2 * lambda3 * lambda4);
}

/*
 Deterministic block-exchangeable kernel for binomial outcomes.

 The three correlation parameters correspond to:
 rho1 = same period, different individuals
 rho2 = different periods, different individuals
 rho3 = same individual, different periods

 Rows within each cluster are assumed to be aligned by subject across periods.
 */
static inline ngee_hier_totals ngee_kernel_binomial_tscs(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_index_view& idx) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(3, 3);
  out.G_rho.zeros(3);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;
  const arma::mat std_design_mat = design_mat.each_col() % U_sqrt;

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword T      = idx.cluster_n_sub[i];
    const arma::uword Ji     = idx.cluster_subject_n[i];
    const double dT = static_cast<double>(T);
    const double dJ = static_cast<double>(Ji);
    const double dN = static_cast<double>(csize);

    double lambda1, lambda2, lambda3, lambda4, ccoef;
    ngee_block_constants(rho, dJ, dT, lambda1, lambda2, lambda3, lambda4, ccoef);

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);

    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, arma::fill::zeros);

    double resid2_0 = 0.0;
    double resid2_1 = 0.0;
    double resid2_2 = 0.0;

    for (arma::uword t = 0; t < T; ++t) {
      const arma::uword sid = s0 + t;
      const arma::uword local_start = idx.sub_start[sid] - cstart;
      const arma::uword local_end   = local_start + Ji - 1;

      const auto temp_mat_X = Xtilde.rows(local_start, local_end);
      const auto temp_vec_E = Etilde.rows(local_start, local_end);

      const arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      const double temp_double_E = arma::sum(temp_vec_E);
      const double resid_phi = arma::dot(temp_vec_E, temp_vec_E);

      Xtilde_sub += temp_mat_X;
      Etilde_sub += temp_vec_E;

      out.G_beta += -temp_vec_X.t() * temp_double_E * (rho[0] - rho[1]) / (lambda1 * lambda3);
      out.H_beta += -temp_vec_X.t() * temp_vec_X * (rho[0] - rho[1]) / (lambda1 * lambda3);

      resid2_0 = (temp_double_E * temp_double_E - resid_phi) / 2.0;
      out.G_rho[0] += resid2_0 - dJ * (dJ - 1.0) * rho[0] / 2.0;
      resid2_1 -= resid_phi / 2.0 + resid2_0;
      resid2_2 -= resid_phi / 2.0;
    }

    out.G_beta += Xtilde.t() * Etilde / lambda1 -
      Xtilde_sub.t() * Etilde_sub * (rho[2] - rho[1]) / (lambda1 * lambda2) +
      ccoef * Xtilde_sum.t() * Etilde_sum;

    out.H_beta += Xtilde.t() * Xtilde / lambda1 -
      Xtilde_sub.t() * Xtilde_sub * (rho[2] - rho[1]) / (lambda1 * lambda2) +
      ccoef * Xtilde_sum.t() * Xtilde_sum;

    const double diff_ind_same_period_length = dT * dJ * (dJ - 1.0) / 2.0;
    const double same_ind_diff_period_length = dT * (dT - 1.0) * dJ / 2.0;
    const double diff_ind_and_period_length =
      dN * (dN - 1.0) / 2.0 - diff_ind_same_period_length - same_ind_diff_period_length;

    resid2_2 += arma::dot(Etilde_sub, Etilde_sub) / 2.0;
    resid2_1 += Etilde_sum * Etilde_sum / 2.0 - resid2_2;

    out.G_rho[1] += resid2_1 - diff_ind_and_period_length * rho[1];
    out.G_rho[2] += resid2_2 - same_ind_diff_period_length * rho[2];

    out.H_rho(0, 0) += diff_ind_same_period_length;
    out.H_rho(1, 1) += diff_ind_and_period_length;
    out.H_rho(2, 2) += same_ind_diff_period_length;
  }

  return out;
}

/*
 Gaussian block-exchangeable kernel.

 This mirrors the binomial block kernel and adds the Gaussian phi block.
 */
static inline ngee_hier_totals ngee_kernel_gaussian_tscs(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_index_view& idx) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(3, 3);
  out.G_rho.zeros(3);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;
  const double sqrt_phi = std::sqrt(phi);
  const arma::colvec std_resid = resid / sqrt_phi;
  const arma::mat std_design_mat = design_mat / sqrt_phi;

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword T      = idx.cluster_n_sub[i];
    const arma::uword Ji     = idx.cluster_subject_n[i];
    const double dT = static_cast<double>(T);
    const double dJ = static_cast<double>(Ji);
    const double dN = static_cast<double>(csize);

    double lambda1, lambda2, lambda3, lambda4, ccoef;
    ngee_block_constants(rho, dJ, dT, lambda1, lambda2, lambda3, lambda4, ccoef);

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);

    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);

    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, arma::fill::zeros);

    double resid2_0 = 0.0;
    double resid2_1 = 0.0;
    double resid2_2 = 0.0;

    for (arma::uword t = 0; t < T; ++t) {
      const arma::uword sid = s0 + t;
      const arma::uword local_start = idx.sub_start[sid] - cstart;
      const arma::uword local_end   = local_start + Ji - 1;

      const auto temp_mat_X = Xtilde.rows(local_start, local_end);
      const auto temp_vec_E = Etilde.rows(local_start, local_end);

      const arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      const double temp_double_E = arma::sum(temp_vec_E);
      const double resid_phi = arma::dot(temp_vec_E, temp_vec_E);

      Xtilde_sub += temp_mat_X;
      Etilde_sub += temp_vec_E;

      out.G_beta += -temp_vec_X.t() * temp_double_E * (rho[0] - rho[1]) / (lambda1 * lambda3);
      out.H_beta += -temp_vec_X.t() * temp_vec_X * (rho[0] - rho[1]) / (lambda1 * lambda3);

      resid2_0 = (temp_double_E * temp_double_E - resid_phi) / 2.0;
      out.G_rho[0] += resid2_0 - dJ * (dJ - 1.0) * rho[0] / 2.0;
      resid2_1 -= resid_phi / 2.0 + resid2_0;
      resid2_2 -= resid_phi / 2.0;
    }

    out.G_beta += Xtilde.t() * Etilde / lambda1 -
      Xtilde_sub.t() * Etilde_sub * (rho[2] - rho[1]) / (lambda1 * lambda2) +
      ccoef * Xtilde_sum.t() * Etilde_sum;

    out.H_beta += Xtilde.t() * Xtilde / lambda1 -
      Xtilde_sub.t() * Xtilde_sub * (rho[2] - rho[1]) / (lambda1 * lambda2) +
      ccoef * Xtilde_sum.t() * Xtilde_sum;

    out.G_phi += phi * (arma::dot(Etilde, Etilde) - dN);
    out.H_phi += dN;

    const double diff_ind_same_period_length = dT * dJ * (dJ - 1.0) / 2.0;
    const double same_ind_diff_period_length = dT * (dT - 1.0) * dJ / 2.0;
    const double diff_ind_and_period_length =
      dN * (dN - 1.0) / 2.0 - diff_ind_same_period_length - same_ind_diff_period_length;

    resid2_2 += arma::dot(Etilde_sub, Etilde_sub) / 2.0;
    resid2_1 += Etilde_sum * Etilde_sum / 2.0 - resid2_2;

    out.G_rho[1] += resid2_1 - diff_ind_and_period_length * rho[1];
    out.G_rho[2] += resid2_2 - same_ind_diff_period_length * rho[2];

    out.H_rho(0, 0) += diff_ind_same_period_length;
    out.H_rho(1, 1) += diff_ind_and_period_length;
    out.H_rho(2, 2) += same_ind_diff_period_length;
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_tscs_dispatch(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_index_view& idx,
    const std::string& family) {

  if (family == "binomial") {
    return ngee_kernel_binomial_tscs(beta, rho, outcome, design_mat, idx);
  } else if (family == "gaussian") {
    return ngee_kernel_gaussian_tscs(beta, phi, rho, outcome, design_mat, idx);
  }

  Rcpp::stop("Unsupported family in block-exchangeable kernel.");
  return ngee_hier_totals();
}

// [[Rcpp::export]]
Rcpp::List ngee_kernel_det_tscs_cpp(const arma::colvec& beta,
                                    const double phi,
                                    const arma::colvec& rho,
                                    const arma::colvec& outcome,
                                    const arma::mat& design_mat,
                                    const Rcpp::List& index_data,
                                    const std::string& family,
                                    const double rho_eps = 1e-4) {
  const ngee_tscs_index_view idx = ngee_get_tscs_index_view(index_data);

  if (idx.corstr != "block-exchangeable") Rcpp::stop("index_data$corstr must be 'block-exchangeable'.");
  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta.n_elem != design_mat.n_cols) Rcpp::stop("beta has wrong length.");

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    if (!idx.cluster_balanced[i]) {
      Rcpp::stop("block-exchangeable requires equal period sizes within each cluster.");
    }
  }

  arma::colvec rho_work = rho;
  ngee_stabilize_block_rho(rho_work, idx, rho_eps);

  ngee_hier_totals out = ngee_kernel_tscs_dispatch(
    beta, phi, rho_work, outcome, design_mat, idx, family
  );

  return Rcpp::List::create(
    Rcpp::Named("G_beta") = out.G_beta,
    Rcpp::Named("H_beta") = out.H_beta,
    Rcpp::Named("G_phi")  = out.G_phi,
    Rcpp::Named("H_phi")  = out.H_phi,
    Rcpp::Named("G_rho")  = out.G_rho,
    Rcpp::Named("H_rho")  = out.H_rho,
    Rcpp::Named("rho_used") = rho_work
  );
}

/*
 Deterministic Newton/Fisher-scoring fit for the block-exchangeable model.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_det_tscs_cpp(const arma::colvec& outcome,
                                 const arma::mat& design_mat,
                                 const Rcpp::List& index_data,
                                 const std::string& family,
                                 arma::colvec beta0,
                                 double phi0,
                                 arma::colvec rho0,
                                 const double tol = 1e-8,
                                 const int maxit = 200,
                                 const double phi_min = 1e-8,
                                 const double rho_eps = 1e-4) {
  const ngee_tscs_index_view idx = ngee_get_tscs_index_view(index_data);

  if (idx.corstr != "block-exchangeable") Rcpp::stop("index_data$corstr must be 'block-exchangeable'.");
  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta0.n_elem != design_mat.n_cols) Rcpp::stop("beta0 has wrong length.");
  if (!(family == "binomial" || family == "gaussian")) Rcpp::stop("family must be 'binomial' or 'gaussian'.");
  if (rho0.n_elem != 3) Rcpp::stop("block-exchangeable requires rho0 of length 3.");

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    if (!idx.cluster_balanced[i]) {
      Rcpp::stop("block-exchangeable requires equal period sizes within each cluster.");
    }
  }

  const bool update_phi = (family == "gaussian");

  arma::colvec beta = beta0;
  arma::colvec rho = rho0;
  double phi = std::max(phi0, phi_min);

  ngee_stabilize_block_rho(rho, idx, rho_eps);

  arma::colvec dbeta(beta.n_elem, arma::fill::zeros);
  arma::colvec drho(rho.n_elem, arma::fill::zeros);
  double dphi = 0.0;

  arma::mat Info_beta(beta.n_elem, beta.n_elem, arma::fill::eye);
  arma::mat Info_phi(1, 1, arma::fill::eye);
  arma::mat Info_rho(3, 3, arma::fill::eye);

  ngee_hier_totals last_blocks;
  double err = arma::datum::inf;
  int iter = 0;
  bool converged = false;

  for (iter = 1; iter <= maxit; ++iter) {
    ngee_stabilize_block_rho(rho, idx, rho_eps);

    last_blocks = ngee_kernel_tscs_dispatch(
      beta, phi, rho, outcome, design_mat, idx, family
    );

    dbeta = ngee_safe_solve_vec(last_blocks.H_beta, last_blocks.G_beta);
    beta += dbeta;
    err = ngee_rel_change_vec(dbeta, beta);
    Info_beta = ngee_safe_inv_mat(last_blocks.H_beta);

    if (update_phi) {
      dphi = last_blocks.G_phi / last_blocks.H_phi;
      phi = std::max(phi + dphi, phi_min);
      err += ngee_rel_change_scalar(dphi, phi);
      Info_phi(0, 0) = 1.0 / last_blocks.H_phi;
    } else {
      dphi = 0.0;
      Info_phi(0, 0) = 1.0;
    }

    drho = ngee_safe_solve_vec(last_blocks.H_rho, last_blocks.G_rho);
    rho += drho;
    ngee_stabilize_block_rho(rho, idx, rho_eps);

    err += ngee_rel_change_vec(drho, rho);
    Info_rho = ngee_safe_inv_mat(last_blocks.H_rho);

    if (err <= tol) {
      converged = true;
      break;
    }
  }

  ngee_stabilize_block_rho(rho, idx, rho_eps);
  last_blocks = ngee_kernel_tscs_dispatch(
    beta, phi, rho, outcome, design_mat, idx, family
  );

  Info_beta = ngee_safe_inv_mat(last_blocks.H_beta);
  if (update_phi) Info_phi(0, 0) = 1.0 / last_blocks.H_phi;
  Info_rho = ngee_safe_inv_mat(last_blocks.H_rho);

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("Info_beta") = Info_beta,
    Rcpp::Named("Info_phi") = Info_phi,
    Rcpp::Named("Info_rho") = Info_rho,
    Rcpp::Named("G_beta") = last_blocks.G_beta,
    Rcpp::Named("H_beta") = last_blocks.H_beta,
    Rcpp::Named("G_phi") = last_blocks.G_phi,
    Rcpp::Named("H_phi") = last_blocks.H_phi,
    Rcpp::Named("G_rho") = last_blocks.G_rho,
    Rcpp::Named("H_rho") = last_blocks.H_rho,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("final_error") = err
  );
}

/*
 Block-exchangeable binomial sandwich blocks.

 D contains the cross-derivative terms linking the three correlation classes to
 beta. These are accumulated class by class using the aligned-period layout.
 */
static inline ngee_hier_sandwich_blocks ngee_blocks_binomial_tscs(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_index_view& idx) {

  const arma::uword p = beta.n_elem;
  const arma::uword I = idx.n_cluster;

  ngee_hier_sandwich_blocks out;
  out.H1.zeros(p, p, I);
  out.G1.zeros(p, 1, I);
  out.H1_5.zeros(1, 1, I);
  out.G1_5.zeros(1, 1, I);
  out.H2.zeros(3, 3, I);
  out.G2.zeros(3, 1, I);
  out.B.zeros(1, p, I);
  out.D.zeros(3, p, I);
  out.E.zeros(3, 1, I);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec vsqrt = arma::sqrt(U);
  const arma::colvec dvdm = 1.0 - 2.0 * mu;

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword T      = idx.cluster_n_sub[i];
    const arma::uword Ji     = idx.cluster_subject_n[i];

    arma::mat Xmat = design_mat.rows(cstart, cend);
    arma::colvec Evec = resid.rows(cstart, cend);
    arma::colvec vsqrtsub = vsqrt.rows(cstart, cend);
    arma::colvec dvdmsub = dvdm.rows(cstart, cend);

    arma::mat XE_portion_1 =
      Xmat.each_col() % vsqrtsub +
      Xmat.each_col() % (Evec % dvdmsub / vsqrtsub) / 2.0;
    arma::colvec XE_portion_2 = Evec / vsqrtsub;

    arma::mat XE_sub_1(Ji, p, arma::fill::zeros);
    arma::colvec XE_sub_2(Ji, arma::fill::zeros);
    arma::rowvec D_resid(p, arma::fill::zeros);

    for (arma::uword t = 0; t < T; ++t) {
      const arma::uword sid = s0 + t;
      const arma::uword local_start = idx.sub_start[sid] - cstart;
      const arma::uword local_end   = local_start + Ji - 1;

      const auto temp_mat_XE_portion_1 = XE_portion_1.rows(local_start, local_end);
      const auto temp_vec_XE_portion_2 = XE_portion_2.rows(local_start, local_end);
      const arma::rowvec temp_vec_XE_portion_1 = arma::sum(temp_mat_XE_portion_1, 0);
      const double temp_double_XE_portion_2 = arma::sum(temp_vec_XE_portion_2);

      D_resid += temp_vec_XE_portion_2.t() * temp_mat_XE_portion_1;

      out.D.slice(i).row(0) += temp_vec_XE_portion_1 * temp_double_XE_portion_2 / phi;

      XE_sub_1 += temp_mat_XE_portion_1;
      XE_sub_2 += temp_vec_XE_portion_2;
    }

    const arma::rowvec temp_vec_XE_portion_1 = arma::sum(XE_portion_1, 0);
    const double temp_double_XE_portion_2 = arma::sum(XE_portion_2);

    out.D.slice(i).row(0) -= D_resid / phi;
    out.D.slice(i).row(2) = XE_sub_2.t() * XE_sub_1 / phi - D_resid / phi;
    out.D.slice(i).row(1) =
      temp_vec_XE_portion_1 * temp_double_XE_portion_2 / phi -
      out.D.slice(i).row(2) - out.D.slice(i).row(0) - D_resid / phi;
  }

  return out;
}

/*
 Gaussian block-exchangeable sandwich blocks.

 B, D, and E provide the cross-derivative pieces required for the stacked
 sandwich of (beta, phi, rho1, rho2, rho3).
 */
static inline ngee_hier_sandwich_blocks ngee_blocks_gaussian_tscs(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_index_view& idx) {

  const arma::uword p = beta.n_elem;
  const arma::uword I = idx.n_cluster;

  ngee_hier_sandwich_blocks out;
  out.H1.zeros(p, p, I);
  out.G1.zeros(p, 1, I);
  out.H1_5.zeros(1, 1, I);
  out.G1_5.zeros(1, 1, I);
  out.H2.zeros(3, 3, I);
  out.G2.zeros(3, 1, I);
  out.B.zeros(1, p, I);
  out.D.zeros(3, p, I);
  out.E.zeros(3, 1, I);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword T      = idx.cluster_n_sub[i];
    const arma::uword Ji     = idx.cluster_subject_n[i];

    arma::mat Xmat = design_mat.rows(cstart, cend);
    arma::colvec Evec = resid.rows(cstart, cend);

    out.B.slice(i) = 2.0 * Evec.t() * Xmat;

    arma::mat XE_sub_1(Ji, p, arma::fill::zeros);
    arma::colvec XE_sub_2(Ji, arma::fill::zeros);
    arma::rowvec D_resid(p, arma::fill::zeros);
    double E_resid = 0.0;

    for (arma::uword t = 0; t < T; ++t) {
      const arma::uword sid = s0 + t;
      const arma::uword local_start = idx.sub_start[sid] - cstart;
      const arma::uword local_end   = local_start + Ji - 1;

      const auto temp_mat_X = Xmat.rows(local_start, local_end);
      const auto temp_vec_E = Evec.rows(local_start, local_end);
      const arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      const double temp_double_E = arma::sum(temp_vec_E);

      D_resid += temp_vec_E.t() * temp_mat_X;
      E_resid += arma::dot(temp_vec_E, temp_vec_E);

      out.D.slice(i).row(0) += temp_vec_X * temp_double_E / phi;
      out.E(0, 0, i) += temp_double_E * temp_double_E / (2.0 * phi * phi);

      XE_sub_1 += temp_mat_X;
      XE_sub_2 += temp_vec_E;
    }

    const arma::rowvec temp_vec_X = arma::sum(Xmat, 0);
    const double temp_double_E = arma::sum(Evec);

    out.D.slice(i).row(0) -= D_resid / phi;
    out.D.slice(i).row(2) = XE_sub_2.t() * XE_sub_1 / phi - D_resid / phi;
    out.D.slice(i).row(1) =
      temp_vec_X * temp_double_E / phi -
      out.D.slice(i).row(2) - out.D.slice(i).row(0) - D_resid / phi;

    out.E(0, 0, i) -= E_resid / (2.0 * phi * phi);
    out.E(2, 0, i) = arma::dot(XE_sub_2, XE_sub_2) / (2.0 * phi * phi) - E_resid / (2.0 * phi * phi);
    out.E(1, 0, i) =
      temp_double_E * temp_double_E / (2.0 * phi * phi) -
      out.E(2, 0, i) - out.E(0, 0, i) - E_resid / (2.0 * phi * phi);
  }

  return out;
}

static inline ngee_hier_sandwich_blocks ngee_tscs_blocks_dispatch(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_index_view& idx,
    const std::string& family) {

  if (family == "binomial") {
    return ngee_blocks_binomial_tscs(beta, 1.0, rho, outcome, design_mat, idx);
  } else if (family == "gaussian") {
    return ngee_blocks_gaussian_tscs(beta, phi, rho, outcome, design_mat, idx);
  }

  Rcpp::stop("Unsupported family in block-exchangeable sandwich dispatch.");
  return ngee_hier_sandwich_blocks();
}

// [[Rcpp::export]]
Rcpp::List ngee_tscs_sandwich_blocks_cpp(const arma::colvec& beta,
                                         const double phi,
                                         const arma::colvec& rho,
                                         const arma::colvec& outcome,
                                         const arma::mat& design_mat,
                                         const Rcpp::List& index_data,
                                         const std::string& family,
                                         const double rho_eps = 1e-4) {
  const ngee_tscs_index_view idx = ngee_get_tscs_index_view(index_data);

  if (idx.corstr != "block-exchangeable") Rcpp::stop("index_data$corstr must be 'block-exchangeable'.");
  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta.n_elem != design_mat.n_cols) Rcpp::stop("beta has wrong length.");

  arma::colvec rho_work = rho;
  ngee_stabilize_block_rho(rho_work, idx, rho_eps);

  ngee_hier_sandwich_blocks out = ngee_tscs_blocks_dispatch(
    beta, phi, rho_work, outcome, design_mat, idx, family
  );

  return Rcpp::List::create(
    Rcpp::Named("rho_used") = rho_work,
    Rcpp::Named("B") = out.B,
    Rcpp::Named("D") = out.D,
    Rcpp::Named("E") = out.E
  );
}

/*
 Deterministic robust covariance estimator for the block-exchangeable model.
 */
// [[Rcpp::export]]
Rcpp::List ngee_tscs_sandwich_det_cpp(const arma::colvec& beta,
                                      const double phi,
                                      const arma::colvec& rho,
                                      const arma::colvec& outcome,
                                      const arma::mat& design_mat,
                                      const Rcpp::List& index_data,
                                      const std::string& family,
                                      const std::string& se_adjust = "unadjusted",
                                      const double fg_cap = 0.85,
                                      const double rho_eps = 1e-4) {
  const ngee_tscs_index_view idx = ngee_get_tscs_index_view(index_data);
  const arma::uword p = beta.n_elem;
  const arma::uword q = 3;
  const arma::uword I = idx.n_cluster;

  if (idx.corstr != "block-exchangeable") Rcpp::stop("index_data$corstr must be 'block-exchangeable'.");
  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta.n_elem != design_mat.n_cols) Rcpp::stop("beta has wrong length.");
  if (!(family == "binomial" || family == "gaussian")) Rcpp::stop("family must be 'binomial' or 'gaussian'.");

  arma::colvec rho_work = rho;
  ngee_stabilize_block_rho(rho_work, idx, rho_eps);

  ngee_hier_sandwich_blocks sblocks = ngee_tscs_blocks_dispatch(
    beta, phi, rho_work, outcome, design_mat, idx, family
  );

  arma::cube H1(p, p, I, arma::fill::zeros);
  arma::cube G1(p, 1, I, arma::fill::zeros);
  arma::cube H1_5(1, 1, I, arma::fill::zeros);
  arma::cube G1_5(1, 1, I, arma::fill::zeros);
  arma::cube H2(q, q, I, arma::fill::zeros);
  arma::cube G2(q, 1, I, arma::fill::zeros);

  // Recompute clusterwise pieces explicitly for sandwich, matching old layout
  const arma::colvec mu = (family == "binomial") ? ngee_invlogit_vec(design_mat * beta) : design_mat * beta;
  const arma::colvec resid = outcome - mu;
  arma::colvec U;
  if (family == "binomial") {
    U = mu % (1.0 - mu);
  } else {
    U = arma::ones<arma::colvec>(mu.n_elem) * phi;
  }
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;
  const arma::mat std_design_mat =
    (family == "binomial") ? (design_mat.each_col() % U_sqrt) : (design_mat / std::sqrt(phi));

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    const arma::uword cstart = idx.cluster_start[i];
    const arma::uword csize  = idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = idx.cluster_sub_start[i];
    const arma::uword T      = idx.cluster_n_sub[i];
    const arma::uword Ji     = idx.cluster_subject_n[i];
    const double dT = static_cast<double>(T);
    const double dJ = static_cast<double>(Ji);
    const double dN = static_cast<double>(csize);

    double lambda1, lambda2, lambda3, lambda4, ccoef;
    ngee_block_constants(rho_work, dJ, dT, lambda1, lambda2, lambda3, lambda4, ccoef);

    const auto Xtilde = std_design_mat.rows(cstart, cend);
    const auto Etilde = std_resid.rows(cstart, cend);
    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);

    arma::mat Xtilde_sub(Ji, p, arma::fill::zeros);
    arma::colvec Etilde_sub(Ji, arma::fill::zeros);

    double resid2_0 = 0.0;
    double resid2_1 = 0.0;
    double resid2_2 = 0.0;

    for (arma::uword t = 0; t < T; ++t) {
      const arma::uword sid = s0 + t;
      const arma::uword local_start = idx.sub_start[sid] - cstart;
      const arma::uword local_end   = local_start + Ji - 1;

      const auto temp_mat_X = Xtilde.rows(local_start, local_end);
      const auto temp_vec_E = Etilde.rows(local_start, local_end);

      const arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      const double temp_double_E = arma::sum(temp_vec_E);
      const double resid_phi = arma::dot(temp_vec_E, temp_vec_E);

      Xtilde_sub += temp_mat_X;
      Etilde_sub += temp_vec_E;

      G1.slice(i) += -temp_vec_X.t() * temp_double_E * (rho_work[0] - rho_work[1]) / (lambda1 * lambda3);
      H1.slice(i) += -temp_vec_X.t() * temp_vec_X * (rho_work[0] - rho_work[1]) / (lambda1 * lambda3);

      resid2_0 = (temp_double_E * temp_double_E - resid_phi) / 2.0;
      G2(0, 0, i) += resid2_0 - dJ * (dJ - 1.0) * rho_work[0] / 2.0;
      resid2_1 -= resid_phi / 2.0 + resid2_0;
      resid2_2 -= resid_phi / 2.0;
    }

    G1.slice(i) += Xtilde.t() * Etilde / lambda1 -
      Xtilde_sub.t() * Etilde_sub * (rho_work[2] - rho_work[1]) / (lambda1 * lambda2) +
      ccoef * Xtilde_sum.t() * Etilde_sum;
    H1.slice(i) += Xtilde.t() * Xtilde / lambda1 -
      Xtilde_sub.t() * Xtilde_sub * (rho_work[2] - rho_work[1]) / (lambda1 * lambda2) +
      ccoef * Xtilde_sum.t() * Xtilde_sum;

    if (family == "gaussian") {
      G1_5(0, 0, i) = phi * (arma::dot(Etilde, Etilde) - dN);
      H1_5(0, 0, i) = dN;
    }

    const double diff_ind_same_period_length = dT * dJ * (dJ - 1.0) / 2.0;
    const double same_ind_diff_period_length = dT * (dT - 1.0) * dJ / 2.0;
    const double diff_ind_and_period_length =
      dN * (dN - 1.0) / 2.0 - diff_ind_same_period_length - same_ind_diff_period_length;

    resid2_2 += arma::dot(Etilde_sub, Etilde_sub) / 2.0;
    resid2_1 += Etilde_sum * Etilde_sum / 2.0 - resid2_2;

    G2(1, 0, i) += resid2_1 - diff_ind_and_period_length * rho_work[1];
    G2(2, 0, i) += resid2_2 - same_ind_diff_period_length * rho_work[2];

    H2(0, 0, i) = diff_ind_same_period_length;
    H2(1, 1, i) = diff_ind_and_period_length;
    H2(2, 2, i) = same_ind_diff_period_length;
  }

  arma::mat Info1 = ngee_safe_inv_mat(arma::sum(H1, 2));
  arma::mat Info1_5(1, 1, arma::fill::zeros);
  arma::mat Info2 = ngee_safe_inv_mat(arma::sum(H2, 2));

  arma::mat Info;
  arma::mat G_outersum;
  arma::mat var_sandwich;

  if (family == "binomial") {
    const arma::uword d = p + q;
    arma::cube H(d, d, I, arma::fill::zeros);
    arma::cube G(d, 1, I, arma::fill::zeros);

    for (arma::uword i = 0; i < I; ++i) {
      H.slice(i).submat(0, 0, p - 1, p - 1) = H1.slice(i);
      H.slice(i).submat(p, 0, p + q - 1, p - 1) = sblocks.D.slice(i);
      H.slice(i).submat(p, p, p + q - 1, p + q - 1) = H2.slice(i);

      G.slice(i).submat(0, 0, p - 1, 0) = G1.slice(i);
      G.slice(i).submat(p, 0, p + q - 1, 0) = G2.slice(i);
    }

    Info = ngee_safe_inv_mat(arma::sum(H, 2));
    G_outersum = ngee_meat_from_cubes(G, H, Info, se_adjust, fg_cap);
    var_sandwich = Info * G_outersum * Info.t();
  } else {
    const arma::uword d = p + 1 + q;
    arma::cube H(d, d, I, arma::fill::zeros);
    arma::cube G(d, 1, I, arma::fill::zeros);

    for (arma::uword i = 0; i < I; ++i) {
      H.slice(i).submat(0, 0, p - 1, p - 1) = H1.slice(i);
      H.slice(i).submat(p, 0, p, p - 1) = sblocks.B.slice(i);
      H.slice(i)(p, p) = H1_5(0, 0, i);
      H.slice(i).submat(p + 1, 0, p + q, p - 1) = sblocks.D.slice(i);
      H.slice(i).submat(p + 1, p, p + q, p) = sblocks.E.slice(i);
      H.slice(i).submat(p + 1, p + 1, p + q, p + q) = H2.slice(i);

      G.slice(i).submat(0, 0, p - 1, 0) = G1.slice(i);
      G.slice(i)(p, 0) = G1_5(0, 0, i);
      G.slice(i).submat(p + 1, 0, p + q, 0) = G2.slice(i);
    }

    Info1_5(0, 0) = 1.0 / arma::accu(H1_5);
    Info = ngee_safe_inv_mat(arma::sum(H, 2));
    G_outersum = ngee_meat_from_cubes(G, H, Info, se_adjust, fg_cap);
    var_sandwich = Info * G_outersum * Info.t();
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho_work,
    Rcpp::Named("Info1") = Info1,
    Rcpp::Named("Info1_5") = Info1_5,
    Rcpp::Named("Info2") = Info2,
    Rcpp::Named("Info") = Info,
    Rcpp::Named("G_outersum") = G_outersum,
    Rcpp::Named("var_sandwich") = var_sandwich,
    Rcpp::Named("B") = sblocks.B,
    Rcpp::Named("D") = sblocks.D,
    Rcpp::Named("E") = sblocks.E,
    Rcpp::Named("H1") = H1,
    Rcpp::Named("G1") = G1,
    Rcpp::Named("H1_5") = H1_5,
    Rcpp::Named("G1_5") = G1_5,
    Rcpp::Named("H2") = H2,
    Rcpp::Named("G2") = G2
  );
}

static inline arma::colvec ngee_pack_active_coef_tscs(const arma::colvec& beta,
                                                      const double phi,
                                                      const arma::colvec& rho,
                                                      const std::string& family) {
  const arma::uword p = beta.n_elem;
  const bool include_phi = (family == "gaussian");
  arma::colvec out(p + static_cast<arma::uword>(include_phi) + 3, arma::fill::zeros);

  arma::uword pos = 0;
  out.subvec(pos, pos + p - 1) = beta;
  pos += p;

  if (include_phi) {
    out[pos] = phi;
    ++pos;
  }

  out.subvec(pos, pos + 2) = rho.subvec(0, 2);
  return out;
}

static inline Rcpp::CharacterVector ngee_active_param_names_tscs(const arma::uword p,
                                                                 const std::string& family) {
  const bool include_phi = (family == "gaussian");
  const arma::uword d = p + static_cast<arma::uword>(include_phi) + 3;
  Rcpp::CharacterVector out(d);

  arma::uword pos = 0;
  for (arma::uword j = 0; j < p; ++j) {
    out[pos] = "beta" + std::to_string(j + 1);
    ++pos;
  }

  if (include_phi) {
    out[pos] = "phi";
    ++pos;
  }

  out[pos++] = "rho1";
  out[pos++] = "rho2";
  out[pos++] = "rho3";
  return out;
}

/*
 High-level deterministic block-exchangeable fit with robust covariance and
 blockwise standard errors.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_det_tscs_se_cpp(const arma::colvec& outcome,
                                    const arma::mat& design_mat,
                                    const Rcpp::List& index_data,
                                    const std::string& family,
                                    arma::colvec beta0,
                                    double phi0,
                                    arma::colvec rho0,
                                    const std::string& se_adjust = "unadjusted",
                                    const double tol = 1e-8,
                                    const int maxit = 200,
                                    const double phi_min = 1e-8,
                                    const double fg_cap = 0.85,
                                    const double rho_eps = 1e-4) {
  Rcpp::List fit = ngee_fit_det_tscs_cpp(
    outcome,
    design_mat,
    index_data,
    family,
    beta0,
    phi0,
    rho0,
    tol,
    maxit,
    phi_min,
    rho_eps
  );

  const arma::colvec beta = Rcpp::as<arma::colvec>(fit["beta"]);
  const double phi = Rcpp::as<double>(fit["phi"]);
  const arma::colvec rho = Rcpp::as<arma::colvec>(fit["rho"]);
  const int iter = Rcpp::as<int>(fit["iter"]);
  const bool converged = Rcpp::as<bool>(fit["converged"]);
  const double final_error = Rcpp::as<double>(fit["final_error"]);

  Rcpp::List sand = ngee_tscs_sandwich_det_cpp(
    beta,
    phi,
    rho,
    outcome,
    design_mat,
    index_data,
    family,
    se_adjust,
    fg_cap,
    rho_eps
  );

  const arma::mat var_sandwich = Rcpp::as<arma::mat>(sand["var_sandwich"]);
  const arma::mat Info = Rcpp::as<arma::mat>(sand["Info"]);
  const arma::colvec se = ngee_diag_se(var_sandwich);
  const arma::colvec coef = ngee_pack_active_coef_tscs(beta, phi, rho, family);

  if (coef.n_elem != se.n_elem) {
    Rcpp::stop("Internal dimension mismatch: coef and sandwich SE lengths differ.");
  }

  arma::colvec z = coef / se;
  for (arma::uword j = 0; j < z.n_elem; ++j) {
    if (!std::isfinite(z[j])) z[j] = NA_REAL;
  }

  const arma::uword p = beta.n_elem;
  const bool include_phi = (family == "gaussian");

  arma::colvec beta_se = se.subvec(0, p - 1);

  arma::colvec phi_se;
  phi_se.zeros(include_phi ? 1 : 0);
  if (include_phi) {
    phi_se[0] = se[p];
  }

  arma::colvec rho_se(3, arma::fill::zeros);
  rho_se = se.subvec(p + static_cast<arma::uword>(include_phi),
                     p + static_cast<arma::uword>(include_phi) + 2);

  return Rcpp::List::create(
    Rcpp::Named("family") = family,
    Rcpp::Named("corstr") = "block-exchangeable",
    Rcpp::Named("se_adjust") = se_adjust,
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("beta_se") = beta_se,
    Rcpp::Named("phi_se") = phi_se,
    Rcpp::Named("rho_se") = rho_se,
    Rcpp::Named("coef") = coef,
    Rcpp::Named("se") = se,
    Rcpp::Named("z") = z,
    Rcpp::Named("param_names") = ngee_active_param_names_tscs(p, family),
    Rcpp::Named("var_sandwich") = var_sandwich,
    Rcpp::Named("Info") = Info,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("final_error") = final_error
  );
}

/***** FAST INTERNAL SAMPLERS FOR STOCHASTIC FITS ******************************
 Purpose:
 - avoid Rcpp::List creation/unpacking inside stochastic loops
 - return flat sampled index metadata directly as C++ structs
 *******************************************************************************/

/*
 Internal sampled hierarchical subproblem used inside stochastic loops.

 obs_index0 identifies the selected rows in the full outcome/design matrices,
 while idx describes the sampled cluster/subcluster layout in zero-based form.
 The extra full_* fields preserve the original cluster/subcluster sizes attached
 to the sampled units.  These are required for the weighted stochastic NGEE
 updates, whose coefficients depend on the ratio of full-sample to subsample
 cluster sizes.
 */
struct ngee_hier_sampled {
  arma::uvec obs_index0;
  ngee_hier_index_view idx;
  arma::uword full_n_cluster;
  arma::uvec full_cluster_size;
  arma::uvec full_cluster_n_sub;
  arma::uvec full_sub_size;
};

struct ngee_tscs_sampled {
  arma::uvec obs_index0;
  ngee_tscs_index_view idx;
  arma::uword full_n_cluster;
  arma::uvec full_cluster_subject_n;
};

/*
 Fast internal sampler for stochastic hierarchical fitting.

 This reproduces the same sampling scheme as the exported sampler but avoids
 constructing and unpacking Rcpp lists inside the hot stochastic loops.
 */
/*
 Fast internal sampler for stochastic hierarchical fitting.

 This reproduces the same sampling scheme as the exported sampler but avoids
 constructing and unpacking Rcpp lists inside the hot stochastic loops.  In
 addition to the sampled index layout, it keeps the corresponding full-sample
 cluster/subcluster sizes for the sampled units so that the weighted stochastic
 estimating equations match the original implementation.
 */
static inline ngee_hier_sampled ngee_sample_hier_fast(const ngee_hier_index_view& full_idx,
                                                      const arma::uvec& batch_size,
                                                      const bool replace = false) {
  ngee_hier_sampled out;

  const std::string corstr = full_idx.corstr;
  const arma::uword I_full = full_idx.n_cluster;
  const arma::uvec all_clusters = ngee_seq0(static_cast<unsigned int>(I_full));

  if (I_full == 0) Rcpp::stop("Full hierarchical index has zero clusters.");
  if (batch_size.n_elem == 0) Rcpp::stop("batch_size must have positive length.");

  arma::uvec cluster_sel = ngee_sample_values(all_clusters, batch_size[0], replace);

  std::vector<unsigned int> obs_index_v;
  std::vector<unsigned int> cluster_start_v;
  std::vector<unsigned int> cluster_size_v;
  std::vector<unsigned int> cluster_sub_start_v;
  std::vector<unsigned int> cluster_n_sub_v;
  std::vector<unsigned int> sub_start_v;
  std::vector<unsigned int> sub_size_v;
  std::vector<unsigned int> full_cluster_size_v;
  std::vector<unsigned int> full_cluster_n_sub_v;
  std::vector<unsigned int> full_sub_size_v;

  unsigned int obs_cursor = 0;
  unsigned int sub_cursor = 0;

  if (corstr == "independence" || corstr == "simple-exchangeable") {
    if (batch_size.n_elem < 2) {
      Rcpp::stop("For independence/simple-exchangeable, batch_size must have length >= 2.");
    }

    const unsigned int n_obs_target = batch_size[1];
    obs_index_v.reserve(cluster_sel.n_elem * n_obs_target);
    cluster_start_v.reserve(cluster_sel.n_elem);
    cluster_size_v.reserve(cluster_sel.n_elem);
    full_cluster_size_v.reserve(cluster_sel.n_elem);

    for (arma::uword ii = 0; ii < cluster_sel.n_elem; ++ii) {
      const unsigned int ci = cluster_sel[ii];
      const unsigned int cstart = full_idx.cluster_start[ci];
      const unsigned int csize = full_idx.cluster_size[ci];

      arma::uvec obs_pos = ngee_seq0(csize);
      obs_pos = ngee_sample_values(obs_pos, n_obs_target, replace);
      obs_pos += cstart;

      cluster_start_v.push_back(obs_cursor);
      cluster_size_v.push_back(obs_pos.n_elem);
      full_cluster_size_v.push_back(csize);

      ngee_append_uvec(obs_index_v, obs_pos);
      obs_cursor += obs_pos.n_elem;
    }

    out.idx.corstr = corstr;
    out.idx.cluster_start = ngee_vec_to_uvec(cluster_start_v);
    out.idx.cluster_size = ngee_vec_to_uvec(cluster_size_v);
    out.idx.cluster_sub_start = arma::uvec();
    out.idx.cluster_n_sub = arma::uvec();
    out.idx.sub_start = arma::uvec();
    out.idx.sub_size = arma::uvec();
    out.idx.n_cluster = out.idx.cluster_start.n_elem;
    out.idx.n_obs = obs_index_v.size();

    out.full_cluster_size = ngee_vec_to_uvec(full_cluster_size_v);
    out.full_cluster_n_sub = arma::uvec();
    out.full_sub_size = arma::uvec();
  } else if (corstr == "nested-exchangeable") {
    if (batch_size.n_elem < 3) {
      Rcpp::stop("For nested-exchangeable, batch_size must have length >= 3.");
    }

    const unsigned int n_sub_target = batch_size[1];
    const unsigned int n_obs_target = batch_size[2];

    cluster_start_v.reserve(cluster_sel.n_elem);
    cluster_size_v.reserve(cluster_sel.n_elem);
    cluster_sub_start_v.reserve(cluster_sel.n_elem);
    cluster_n_sub_v.reserve(cluster_sel.n_elem);
    full_cluster_n_sub_v.reserve(cluster_sel.n_elem);

    for (arma::uword ii = 0; ii < cluster_sel.n_elem; ++ii) {
      const unsigned int ci = cluster_sel[ii];
      const unsigned int sub0 = full_idx.cluster_sub_start[ci];
      const unsigned int nsub_full = full_idx.cluster_n_sub[ci];

      arma::uvec sub_ids = ngee_seq0(nsub_full);
      sub_ids += sub0;
      sub_ids = ngee_sample_values(sub_ids, n_sub_target, replace);

      const unsigned int cluster_obs_start = obs_cursor;

      cluster_start_v.push_back(cluster_obs_start);
      cluster_sub_start_v.push_back(sub_cursor);
      cluster_n_sub_v.push_back(sub_ids.n_elem);
      full_cluster_n_sub_v.push_back(nsub_full);

      for (arma::uword jj = 0; jj < sub_ids.n_elem; ++jj) {
        const unsigned int sid = sub_ids[jj];
        const unsigned int sstart = full_idx.sub_start[sid];
        const unsigned int ssize = full_idx.sub_size[sid];

        arma::uvec obs_pos = ngee_seq0(ssize);
        obs_pos = ngee_sample_values(obs_pos, n_obs_target, replace);
        obs_pos += sstart;

        sub_start_v.push_back(obs_cursor);
        sub_size_v.push_back(obs_pos.n_elem);
        full_sub_size_v.push_back(ssize);

        ngee_append_uvec(obs_index_v, obs_pos);
        obs_cursor += obs_pos.n_elem;
        ++sub_cursor;
      }

      cluster_size_v.push_back(obs_cursor - cluster_obs_start);
    }

    out.idx.corstr = corstr;
    out.idx.cluster_start = ngee_vec_to_uvec(cluster_start_v);
    out.idx.cluster_size = ngee_vec_to_uvec(cluster_size_v);
    out.idx.cluster_sub_start = ngee_vec_to_uvec(cluster_sub_start_v);
    out.idx.cluster_n_sub = ngee_vec_to_uvec(cluster_n_sub_v);
    out.idx.sub_start = ngee_vec_to_uvec(sub_start_v);
    out.idx.sub_size = ngee_vec_to_uvec(sub_size_v);
    out.idx.n_cluster = out.idx.cluster_start.n_elem;
    out.idx.n_obs = obs_index_v.size();

    out.full_cluster_size = arma::uvec();
    out.full_cluster_n_sub = ngee_vec_to_uvec(full_cluster_n_sub_v);
    out.full_sub_size = ngee_vec_to_uvec(full_sub_size_v);
  } else {
    Rcpp::stop("Unsupported corstr in ngee_sample_hier_fast().");
  }

  out.full_n_cluster = I_full;
  out.obs_index0 = ngee_vec_to_uvec(obs_index_v);
  return out;
}

static inline ngee_tscs_sampled ngee_sample_tscs_fast(const ngee_tscs_index_view& full_idx,
                                                      const arma::uvec& batch_size,
                                                      const bool replace = false) {
  ngee_tscs_sampled out;

  if (full_idx.corstr != "block-exchangeable") {
    Rcpp::stop("ngee_sample_tscs_fast() requires block-exchangeable index.");
  }
  if (batch_size.n_elem < 2) {
    Rcpp::stop("For block-exchangeable, batch_size must have length >= 2.");
  }

  const arma::uword I_full = full_idx.n_cluster;
  const arma::uvec all_clusters = ngee_seq0(static_cast<unsigned int>(I_full));
  arma::uvec cluster_sel = ngee_sample_values(all_clusters, batch_size[0], replace);

  std::vector<unsigned int> obs_index_v;
  std::vector<unsigned int> cluster_start_v;
  std::vector<unsigned int> cluster_size_v;
  std::vector<unsigned int> cluster_sub_start_v;
  std::vector<unsigned int> cluster_n_sub_v;
  std::vector<unsigned int> sub_start_v;
  std::vector<unsigned int> sub_size_v;
  std::vector<unsigned int> cluster_subject_n_v;
  std::vector<unsigned int> full_cluster_subject_n_v;

  unsigned int obs_cursor = 0;
  unsigned int sub_cursor = 0;

  for (arma::uword ii = 0; ii < cluster_sel.n_elem; ++ii) {
    const unsigned int ci = cluster_sel[ii];

    if (!full_idx.cluster_balanced[ci]) {
      Rcpp::stop("Encountered unbalanced cluster in block-exchangeable sampling.");
    }

    const unsigned int sub0 = full_idx.cluster_sub_start[ci];
    const unsigned int nsub_full = full_idx.cluster_n_sub[ci];
    const unsigned int subject_n_full = full_idx.cluster_subject_n[ci];

    arma::uvec subject_pos = ngee_seq0(subject_n_full);
    subject_pos = ngee_sample_values(subject_pos, batch_size[1], replace);

    const unsigned int cluster_obs_start = obs_cursor;

    cluster_start_v.push_back(cluster_obs_start);
    cluster_sub_start_v.push_back(sub_cursor);
    cluster_n_sub_v.push_back(nsub_full);
    cluster_subject_n_v.push_back(subject_pos.n_elem);
    full_cluster_subject_n_v.push_back(subject_n_full);

    for (unsigned int t = 0; t < nsub_full; ++t) {
      const unsigned int sid = sub0 + t;
      const unsigned int sstart = full_idx.sub_start[sid];

      arma::uvec obs_pos = subject_pos;
      obs_pos += sstart;

      sub_start_v.push_back(obs_cursor);
      sub_size_v.push_back(obs_pos.n_elem);

      ngee_append_uvec(obs_index_v, obs_pos);
      obs_cursor += obs_pos.n_elem;
      ++sub_cursor;
    }

    cluster_size_v.push_back(obs_cursor - cluster_obs_start);
  }

  out.obs_index0 = ngee_vec_to_uvec(obs_index_v);
  out.full_n_cluster = I_full;
  out.full_cluster_subject_n = ngee_vec_to_uvec(full_cluster_subject_n_v);

  out.idx.corstr = "block-exchangeable";
  out.idx.cluster_start = ngee_vec_to_uvec(cluster_start_v);
  out.idx.cluster_size = ngee_vec_to_uvec(cluster_size_v);
  out.idx.cluster_sub_start = ngee_vec_to_uvec(cluster_sub_start_v);
  out.idx.cluster_n_sub = ngee_vec_to_uvec(cluster_n_sub_v);
  out.idx.sub_start = ngee_vec_to_uvec(sub_start_v);
  out.idx.sub_size = ngee_vec_to_uvec(sub_size_v);
  out.idx.cluster_subject_n = ngee_vec_to_uvec(cluster_subject_n_v);
  out.idx.cluster_balanced = Rcpp::LogicalVector(out.idx.cluster_start.n_elem, true);
  out.idx.n_cluster = out.idx.cluster_start.n_elem;
  out.idx.n_obs = out.obs_index0.n_elem;

  return out;
}

static inline double ngee_stoch_lr(const int iter) {
  return 1.0 / std::pow(static_cast<double>(iter) + 1.0, 0.51);
}

static inline void ngee_det_one_step_hier(arma::colvec& beta,
                                          double& phi,
                                          arma::colvec& rho,
                                          const arma::colvec& outcome,
                                          const arma::mat& design_mat,
                                          const ngee_hier_index_view& idx,
                                          const std::string& family,
                                          const std::string& corstr,
                                          const double phi_min,
                                          const double rho_eps) {
  const bool update_phi = (family == "gaussian");
  const bool update_rho = (corstr == "simple-exchangeable" || corstr == "nested-exchangeable");

  if (corstr == "simple-exchangeable") {
    ngee_stabilize_simple_rho(rho, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    ngee_stabilize_nested_rho(rho, idx, rho_eps);
  }

  ngee_hier_totals blocks = ngee_kernel_hier_dispatch(
    beta, phi, rho, outcome, design_mat, idx, family, corstr
  );

  beta += ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);

  if (update_phi) {
    phi = std::max(phi + blocks.G_phi / blocks.H_phi, phi_min);
  }

  if (update_rho) {
    rho += ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
    if (corstr == "simple-exchangeable") {
      ngee_stabilize_simple_rho(rho, idx, rho_eps);
    } else {
      ngee_stabilize_nested_rho(rho, idx, rho_eps);
    }
  }
}


/*
 Weighted stochastic hierarchical kernels.

 These are faithful transcriptions of the old stochastic NGEE updates.  The
 subsampled score/Hessian contributions are reweighted using the full-versus-
 subsample size factors derived in Chen et al. (2020) so that the resulting
 subsampled estimating equations target the full-data equations.

 For independence/simple-exchangeable this reduces to the old one-level
 stochastic hierarchical solver.  For nested-exchangeable it reproduces the old
 three-level weighting factors IJK, K_minus_1, and J_minus_1_K.
 */
static inline ngee_hier_totals ngee_kernel_stoch_binomial_hier_q1(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_sampled& samp,
    const std::string& corstr) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(1, 1);
  out.G_rho.zeros(1);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;

  const arma::uword I = samp.idx.n_cluster;
  const double I_full = static_cast<double>(samp.full_n_cluster);
  const double rho0 = 1.0 - rho[0];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = samp.idx.cluster_start[i];
    const arma::uword csize  = samp.idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const double Ji = static_cast<double>(csize);
    const double J_full = static_cast<double>(samp.full_cluster_size[i]);

    arma::mat Xtilde = design_mat.rows(cstart, cend);
    Xtilde.each_col() %= U_sqrt.rows(cstart, cend);
    const arma::colvec Etilde = std_resid.rows(cstart, cend);

    if (corstr == "independence") {
      const double weight = (I_full * J_full) / (static_cast<double>(I) * Ji);
      out.G_beta += weight * Xtilde.t() * Etilde;
      out.H_beta += weight * Xtilde.t() * Xtilde;
      continue;
    }

    if (Ji <= 1.0) {
      Rcpp::stop("simple-exchangeable stochastic fitting requires at least 2 sampled observations per cluster.");
    }

    const arma::rowvec Xsum = arma::sum(Xtilde, 0);
    const double Esum = arma::sum(Etilde);

    const double const1 = (I_full * J_full) / (static_cast<double>(I) * Ji) *
      (1.0 / rho0 + rho[0] / (rho0 * (rho0 + rho[0] * J_full)) * (J_full - Ji) / (Ji - 1.0));
    const double const2 = (I_full * J_full * (J_full - 1.0)) /
      (static_cast<double>(I) * Ji * (Ji - 1.0)) * rho[0] / (rho0 * (rho0 + rho[0] * J_full));

    out.G_beta += const1 * Xtilde.t() * Etilde - const2 * Xsum.t() * Esum;
    out.H_beta += const1 * Xtilde.t() * Xtilde - const2 * Xsum.t() * Xsum;

    const double pair_weight = (I_full * J_full * (J_full - 1.0)) /
      (static_cast<double>(I) * Ji * (Ji - 1.0));
    out.G_rho[0] += pair_weight *
      ((Esum * Esum - arma::dot(Etilde, Etilde)) / 2.0 - Ji * (Ji - 1.0) * rho[0] / 2.0);
    out.H_rho(0, 0) += (I_full * J_full * (J_full - 1.0)) / (2.0 * static_cast<double>(I));
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_stoch_gaussian_hier_q1(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_sampled& samp,
    const std::string& corstr) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(1, 1);
  out.G_rho.zeros(1);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;
  const arma::colvec std_resid = resid / std::sqrt(phi);

  const arma::uword I = samp.idx.n_cluster;
  const double I_full = static_cast<double>(samp.full_n_cluster);
  const double rho0 = 1.0 - rho[0];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = samp.idx.cluster_start[i];
    const arma::uword csize  = samp.idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const double Ji = static_cast<double>(csize);
    const double J_full = static_cast<double>(samp.full_cluster_size[i]);

    arma::mat Xtilde = design_mat.rows(cstart, cend);
    Xtilde /= std::sqrt(phi);
    const arma::colvec Etilde = std_resid.rows(cstart, cend);

    if (corstr == "independence") {
      const double weight = (I_full * J_full) / (static_cast<double>(I) * Ji);
      out.G_beta += weight * Xtilde.t() * Etilde;
      out.H_beta += weight * Xtilde.t() * Xtilde;
      out.G_phi += weight * phi * arma::sum(arma::square(Etilde) - 1.0);
      out.H_phi += (I_full * J_full) / static_cast<double>(I);
      continue;
    }

    if (Ji <= 1.0) {
      Rcpp::stop("simple-exchangeable stochastic fitting requires at least 2 sampled observations per cluster.");
    }

    const arma::rowvec Xsum = arma::sum(Xtilde, 0);
    const double Esum = arma::sum(Etilde);

    const double const1 = (I_full * J_full) / (static_cast<double>(I) * Ji) *
      (1.0 / rho0 + rho[0] / (rho0 * (rho0 + rho[0] * J_full)) * (J_full - Ji) / (Ji - 1.0));
    const double const2 = (I_full * J_full * (J_full - 1.0)) /
      (static_cast<double>(I) * Ji * (Ji - 1.0)) * rho[0] / (rho0 * (rho0 + rho[0] * J_full));

    out.G_beta += const1 * Xtilde.t() * Etilde - const2 * Xsum.t() * Esum;
    out.H_beta += const1 * Xtilde.t() * Xtilde - const2 * Xsum.t() * Xsum;

    const arma::colvec std_resid_sub = std_resid.rows(cstart, cend);
    out.G_phi += (I_full * J_full) / (static_cast<double>(I) * Ji) * phi * arma::sum(arma::square(std_resid_sub) - 1.0);
    out.H_phi += (I_full * J_full) / static_cast<double>(I);

    const double pair_weight = (I_full * J_full * (J_full - 1.0)) /
      (static_cast<double>(I) * Ji * (Ji - 1.0));
    out.G_rho[0] += pair_weight *
      ((Esum * Esum - arma::dot(Etilde, Etilde)) / 2.0 - Ji * (Ji - 1.0) * rho[0] / 2.0);
    out.H_rho(0, 0) += (I_full * J_full * (J_full - 1.0)) / (2.0 * static_cast<double>(I));
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_stoch_binomial_hier_q2(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_sampled& samp) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(2, 2);
  out.G_rho.zeros(2);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;

  const arma::uword I = samp.idx.n_cluster;
  const double I_full = static_cast<double>(samp.full_n_cluster);
  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = samp.idx.cluster_start[i];
    const arma::uword csize  = samp.idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = samp.idx.cluster_sub_start[i];
    const arma::uword Ji_u   = samp.idx.cluster_n_sub[i];
    const double Ji = static_cast<double>(Ji_u);
    const double J_full = static_cast<double>(samp.full_cluster_n_sub[i]);

    if (Ji <= 1.0) {
      Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 sampled subclusters per cluster.");
    }

    arma::uvec J_sub_u = samp.idx.sub_size.subvec(s0, s0 + Ji_u - 1);
    arma::uvec Kij_u = samp.full_sub_size.subvec(s0, s0 + Ji_u - 1);
    arma::vec J_sub = arma::conv_to<arma::vec>::from(J_sub_u);
    arma::vec Kij = arma::conv_to<arma::vec>::from(Kij_u);

    arma::vec vij = 1.0 / rho0 - delta / (rho0 * (rho0 / Kij + delta));
    arma::vec v = arma::zeros<arma::vec>(csize);
    arma::uword cursor = 0;
    for (arma::uword j = 0; j < Ji_u; ++j) {
      v.subvec(cursor, cursor + J_sub_u[j] - 1).fill(vij[j]);
      cursor += J_sub_u[j];
    }
    const double c = ngee_nested_c_value(rho[1], arma::dot(vij, Kij));
    const double sign_c = (c >= 0.0 ? 1.0 : -1.0);

    arma::mat Xtilde = design_mat.rows(cstart, cend);
    Xtilde.each_col() %= U_sqrt.rows(cstart, cend);
    const arma::colvec Etilde = std_resid.rows(cstart, cend);
    arma::mat Xtildev = Xtilde.each_col() % v;
    arma::colvec Etildev = Etilde % v;

    double a = 0.0;
    double a0 = 0.0;
    double a1 = 0.0;
    double b1 = 0.0;
    arma::uword idx_val2 = 0;
    double G1temp = 0.0;
    arma::rowvec H1temp(p, arma::fill::zeros);

    for (arma::uword j = 0; j < Ji_u; ++j) {
      const double Jsubj = J_sub[j];
      const double Kijj = Kij[j];
      if (Jsubj <= 1.0) {
        Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 sampled observations per sampled subcluster.");
      }

      const double IJK = I_full * J_full * Kijj / (static_cast<double>(I) * Ji * Jsubj);
      const double K_minus_1 = (Kijj - 1.0) / (Jsubj - 1.0);
      const double J_minus_1_K = ((J_full - 1.0) * Kijj) / ((Ji - 1.0) * Jsubj);

      const double m11 = IJK / rho0 + delta / (rho0 * (rho0 + delta * Kijj)) * (K_minus_1 - 1.0) * IJK;
      const double m12 = c * (K_minus_1 - 1.0) * IJK;
      const double m21 = delta / (rho0 * (rho0 + delta * Kijj)) * K_minus_1 * IJK;
      const double m22 = c * (K_minus_1 - J_minus_1_K) * IJK;
      const double m32 = std::sqrt(std::abs(c) * J_minus_1_K * IJK);

      arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      arma::mat temp_mat_v_X = Xtildev.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      arma::rowvec temp_vec_v_X = arma::sum(temp_mat_v_X, 0);
      arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      arma::colvec temp_vec_v_E = Etildev.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      double temp_double_E = arma::sum(temp_vec_E);
      double temp_double_v_E = arma::sum(temp_vec_v_E);

      out.G_beta += m11 * temp_mat_X.t() * temp_vec_E +
        m12 * temp_mat_v_X.t() * temp_vec_v_E -
        m21 * temp_vec_X.t() * temp_double_E -
        m22 * temp_vec_v_X.t() * temp_double_v_E;
      G1temp += m32 * temp_double_v_E;

      out.H_beta += m11 * temp_mat_X.t() * temp_mat_X +
        m12 * temp_mat_v_X.t() * temp_mat_v_X -
        m21 * temp_vec_X.t() * temp_vec_X -
        m22 * temp_vec_v_X.t() * temp_vec_v_X;
      H1temp += m32 * temp_vec_v_X;

      a = IJK * arma::sum(arma::square(temp_vec_E) - 1.0);
      a0 = IJK * K_minus_1 * (temp_double_E * temp_double_E - arma::dot(temp_vec_E, temp_vec_E)) / 2.0;
      out.G_rho[0] += a0 - IJK * K_minus_1 * Jsubj * (Jsubj - 1.0) * rho[0] / 2.0;
      out.G_rho[1] -= ((a / IJK + Jsubj) / 2.0 + a0 / (IJK * K_minus_1)) * IJK * J_minus_1_K;

      out.H_rho(0, 0) += IJK * K_minus_1 * Jsubj * (Jsubj - 1.0) / 2.0;
      out.H_rho(1, 1) += -(Jsubj / 2.0 + Jsubj * (Jsubj - 1.0) / 2.0) * IJK * J_minus_1_K;

      a1 += std::sqrt(IJK * J_minus_1_K) * arma::sum(temp_vec_E);
      b1 += std::sqrt(IJK * J_minus_1_K) * Jsubj;

      idx_val2 += J_sub_u[j];
    }

    out.G_beta += -sign_c * H1temp.t() * G1temp;
    out.H_beta += -sign_c * H1temp.t() * H1temp;

    out.H_rho(1, 1) += b1 * b1 / 2.0;
    out.G_rho[1] += a1 * a1 / 2.0 - out.H_rho(1, 1) * rho[1];
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_stoch_gaussian_hier_q2(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_sampled& samp) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(2, 2);
  out.G_rho.zeros(2);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;
  const arma::colvec std_resid = resid / std::sqrt(phi);

  const arma::uword I = samp.idx.n_cluster;
  const double I_full = static_cast<double>(samp.full_n_cluster);
  const double rho0 = 1.0 - rho[0];
  const double delta = rho[0] - rho[1];

  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword cstart = samp.idx.cluster_start[i];
    const arma::uword csize  = samp.idx.cluster_size[i];
    const arma::uword cend   = cstart + csize - 1;
    const arma::uword s0     = samp.idx.cluster_sub_start[i];
    const arma::uword Ji_u   = samp.idx.cluster_n_sub[i];
    const double Ji = static_cast<double>(Ji_u);
    const double J_full = static_cast<double>(samp.full_cluster_n_sub[i]);

    if (Ji <= 1.0) {
      Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 sampled subclusters per cluster.");
    }

    arma::uvec J_sub_u = samp.idx.sub_size.subvec(s0, s0 + Ji_u - 1);
    arma::uvec Kij_u = samp.full_sub_size.subvec(s0, s0 + Ji_u - 1);
    arma::vec J_sub = arma::conv_to<arma::vec>::from(J_sub_u);
    arma::vec Kij = arma::conv_to<arma::vec>::from(Kij_u);

    arma::vec vij = 1.0 / rho0 - delta / (rho0 * (rho0 / Kij + delta));
    arma::vec v = arma::zeros<arma::vec>(csize);
    arma::uword cursor = 0;
    for (arma::uword j = 0; j < Ji_u; ++j) {
      v.subvec(cursor, cursor + J_sub_u[j] - 1).fill(vij[j]);
      cursor += J_sub_u[j];
    }
    const double c = ngee_nested_c_value(rho[1], arma::dot(vij, Kij));
    const double sign_c = (c >= 0.0 ? 1.0 : -1.0);

    arma::mat Xtilde = design_mat.rows(cstart, cend);
    Xtilde /= std::sqrt(phi);
    const arma::colvec Etilde = std_resid.rows(cstart, cend);
    arma::mat Xtildev = Xtilde.each_col() % v;
    arma::colvec Etildev = Etilde % v;

    double a = 0.0;
    double a0 = 0.0;
    double a1 = 0.0;
    double b1 = 0.0;
    arma::uword idx_val2 = 0;
    double G1temp = 0.0;
    arma::rowvec H1temp(p, arma::fill::zeros);

    for (arma::uword j = 0; j < Ji_u; ++j) {
      const double Jsubj = J_sub[j];
      const double Kijj = Kij[j];
      if (Jsubj <= 1.0) {
        Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 sampled observations per sampled subcluster.");
      }

      const double IJK = I_full * J_full * Kijj / (static_cast<double>(I) * Ji * Jsubj);
      const double K_minus_1 = (Kijj - 1.0) / (Jsubj - 1.0);
      const double J_minus_1_K = ((J_full - 1.0) * Kijj) / ((Ji - 1.0) * Jsubj);

      const double m11 = IJK / rho0 + delta / (rho0 * (rho0 + delta * Kijj)) * (K_minus_1 - 1.0) * IJK;
      const double m12 = c * (K_minus_1 - 1.0) * IJK;
      const double m21 = delta / (rho0 * (rho0 + delta * Kijj)) * K_minus_1 * IJK;
      const double m22 = c * (K_minus_1 - J_minus_1_K) * IJK;
      const double m32 = std::sqrt(std::abs(c) * J_minus_1_K * IJK);

      arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      arma::mat temp_mat_v_X = Xtildev.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      arma::rowvec temp_vec_v_X = arma::sum(temp_mat_v_X, 0);
      arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      arma::colvec temp_vec_v_E = Etildev.rows(idx_val2, idx_val2 + J_sub_u[j] - 1);
      double temp_double_E = arma::sum(temp_vec_E);
      double temp_double_v_E = arma::sum(temp_vec_v_E);

      out.G_beta += m11 * temp_mat_X.t() * temp_vec_E +
        m12 * temp_mat_v_X.t() * temp_vec_v_E -
        m21 * temp_vec_X.t() * temp_double_E -
        m22 * temp_vec_v_X.t() * temp_double_v_E;
      G1temp += m32 * temp_double_v_E;

      out.H_beta += m11 * temp_mat_X.t() * temp_mat_X +
        m12 * temp_mat_v_X.t() * temp_mat_v_X -
        m21 * temp_vec_X.t() * temp_vec_X -
        m22 * temp_vec_v_X.t() * temp_vec_v_X;
      H1temp += m32 * temp_vec_v_X;

      a = IJK * phi * arma::sum(arma::square(temp_vec_E) - 1.0);
      out.G_phi += a;
      out.H_phi += IJK * Jsubj;

      a0 = IJK * K_minus_1 * (temp_double_E * temp_double_E - arma::dot(temp_vec_E, temp_vec_E)) / 2.0;
      out.G_rho[0] += a0 - IJK * K_minus_1 * Jsubj * (Jsubj - 1.0) * rho[0] / 2.0;
      out.G_rho[1] -= ((a / (IJK * phi) + Jsubj) / 2.0 + a0 / (IJK * K_minus_1)) * IJK * J_minus_1_K;

      out.H_rho(0, 0) += IJK * K_minus_1 * Jsubj * (Jsubj - 1.0) / 2.0;
      out.H_rho(1, 1) += -(Jsubj / 2.0 + Jsubj * (Jsubj - 1.0) / 2.0) * IJK * J_minus_1_K;

      a1 += std::sqrt(IJK * J_minus_1_K) * arma::sum(temp_vec_E);
      b1 += std::sqrt(IJK * J_minus_1_K) * Jsubj;

      idx_val2 += J_sub_u[j];
    }

    out.G_beta += -sign_c * H1temp.t() * G1temp;
    out.H_beta += -sign_c * H1temp.t() * H1temp;

    out.H_rho(1, 1) += b1 * b1 / 2.0;
    out.G_rho[1] += a1 * a1 / 2.0 - out.H_rho(1, 1) * rho[1];
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_stoch_hier_dispatch(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_hier_sampled& samp,
    const std::string& family,
    const std::string& corstr) {

  if (family == "binomial") {
    if (corstr == "independence" || corstr == "simple-exchangeable") {
      return ngee_kernel_stoch_binomial_hier_q1(beta, rho, outcome, design_mat, samp, corstr);
    }
    return ngee_kernel_stoch_binomial_hier_q2(beta, rho, outcome, design_mat, samp);
  }

  if (corstr == "independence" || corstr == "simple-exchangeable") {
    return ngee_kernel_stoch_gaussian_hier_q1(beta, phi, rho, outcome, design_mat, samp, corstr);
  }
  return ngee_kernel_stoch_gaussian_hier_q2(beta, phi, rho, outcome, design_mat, samp);
}

/*
 Weighted stochastic block-exchangeable kernels.

 These preserve the original stochastic TSCS weighting scheme.  If J_full is the
 full number of aligned subjects per period in a sampled cluster and J_sub is the
 number retained in the stochastic subsample, then
 IK        = I_full * J_full / (I_sub * J_sub)
 K_minus_1 = (J_full - 1) / (J_sub - 1)
 enter the weighted score/Hessian blocks exactly as in the legacy code.
 */
static inline ngee_hier_totals ngee_kernel_stoch_binomial_tscs(
    const arma::colvec& beta,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_sampled& samp) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(3, 3);
  out.G_rho.zeros(3);

  const arma::colvec mu = ngee_invlogit_vec(design_mat * beta);
  const arma::colvec resid = outcome - mu;
  const arma::colvec U = mu % (1.0 - mu);
  const arma::colvec U_sqrt = arma::sqrt(U);
  const arma::colvec std_resid = resid / U_sqrt;

  const arma::uword I = samp.idx.n_cluster;
  const double I_full = static_cast<double>(samp.full_n_cluster);

  int idx_val = 0;
  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword T = samp.idx.cluster_n_sub[i];
    const double Ji = static_cast<double>(samp.idx.cluster_subject_n[i]);
    const double J_full = static_cast<double>(samp.full_cluster_subject_n[i]);
    const double TJi = static_cast<double>(samp.idx.cluster_size[i]);

    if (Ji <= 1.0) {
      Rcpp::stop("stochastic block-exchangeable fitting requires at least 2 sampled subjects per period.");
    }

    const double lambda1 = 1.0 - rho[0] + rho[1] - rho[2];
    const double lambda2 = 1.0 - rho[0] - (static_cast<double>(T) - 1.0) * (rho[1] - rho[2]);
    const double lambda3 = 1.0 + (J_full - 1.0) * (rho[0] - rho[1]) - rho[2];
    const double lambda4 = 1.0 + (J_full - 1.0) * rho[0] +
      (static_cast<double>(T) - 1.0) * (J_full - 1.0) * rho[1] +
      (static_cast<double>(T) - 1.0) * rho[2];

    const double c1 = 1.0 / lambda1;
    const double c2 = (rho[2] - rho[1]) / (lambda1 * lambda2);
    const double c3 = (rho[0] - rho[1]) / (lambda1 * lambda3);
    const double c4 = (rho[2] - rho[1]) * (rho[0] - rho[1]) / (lambda1 * lambda2 * lambda3) +
      (rho[2] * rho[0] - rho[1]) / (lambda2 * lambda3 * lambda4);

    const double IK = I_full * J_full / (static_cast<double>(I) * Ji);
    const double K_minus_1 = (J_full - 1.0) / (Ji - 1.0);

    const double c1_tilde = c1 * IK + c3 * IK * (K_minus_1 - 1.0);
    const double c2_tilde = c2 * IK + c4 * IK * (K_minus_1 - 1.0);
    const double c3_tilde = c3 * IK * K_minus_1;
    const double c4_tilde = c4 * IK * K_minus_1;

    arma::mat Xtilde = design_mat.rows(idx_val, idx_val + static_cast<int>(TJi) - 1);
    Xtilde.each_col() %= U_sqrt.rows(idx_val, idx_val + static_cast<int>(TJi) - 1);
    const arma::colvec Etilde = std_resid.rows(idx_val, idx_val + static_cast<int>(TJi) - 1);

    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);

    arma::mat Xtilde_sub(static_cast<arma::uword>(Ji), p, arma::fill::zeros);
    arma::colvec Etilde_sub(static_cast<arma::uword>(Ji), arma::fill::zeros);
    int idx_val2 = 0;

    double resid2_0 = 0.0;
    double resid2_1 = 0.0;
    double resid2_2 = 0.0;

    for (arma::uword j = 0; j < T; ++j) {
      const arma::uword Jsubj_u = samp.idx.sub_size[samp.idx.cluster_sub_start[i] + j];
      const double Jsubj = static_cast<double>(Jsubj_u);

      arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + static_cast<int>(Jsubj) - 1);
      arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + static_cast<int>(Jsubj) - 1);
      double temp_double_E = arma::sum(temp_vec_E);

      Xtilde_sub += temp_mat_X;
      Etilde_sub += temp_vec_E;

      out.G_beta += -c3_tilde * temp_vec_X.t() * temp_double_E;
      out.H_beta += -c3_tilde * temp_vec_X.t() * temp_vec_X;

      const double resid_phi = IK * arma::dot(temp_vec_E, temp_vec_E);
      resid2_0 = (temp_double_E * temp_double_E - resid_phi / IK) / 2.0;
      out.G_rho[0] += IK * K_minus_1 * (resid2_0 - Jsubj * (Jsubj - 1.0) * rho[0] / 2.0);
      resid2_1 -= resid_phi / IK / 2.0 + resid2_0;
      resid2_2 -= resid_phi / IK / 2.0;

      idx_val2 += static_cast<int>(Jsubj);
    }

    out.G_beta += c1_tilde * Xtilde.t() * Etilde -
      c2_tilde * Xtilde_sub.t() * Etilde_sub +
      c4_tilde * Xtilde_sum.t() * Etilde_sum;
    out.H_beta += c1_tilde * Xtilde.t() * Xtilde -
      c2_tilde * Xtilde_sub.t() * Xtilde_sub +
      c4_tilde * Xtilde_sum.t() * Xtilde_sum;

    const double diff_ind_same_period_length = arma::sum(arma::conv_to<arma::vec>::from(
      samp.idx.sub_size.subvec(samp.idx.cluster_sub_start[i], samp.idx.cluster_sub_start[i] + T - 1)) %
        (arma::conv_to<arma::vec>::from(
            samp.idx.sub_size.subvec(samp.idx.cluster_sub_start[i], samp.idx.cluster_sub_start[i] + T - 1)) - 1.0) / 2.0);
    const double same_ind_diff_period_length = static_cast<double>(T) * (static_cast<double>(T) - 1.0) * Ji / 2.0;
    const double diff_ind_and_period_length = TJi * (TJi - 1.0) / 2.0 - diff_ind_same_period_length - same_ind_diff_period_length;

    resid2_2 += arma::dot(Etilde_sub, Etilde_sub) / 2.0;
    resid2_1 += Etilde_sum * Etilde_sum / 2.0 - resid2_2;

    out.G_rho[1] += IK * K_minus_1 * (resid2_1 - diff_ind_and_period_length * rho[1]);
    out.G_rho[2] += IK * (resid2_2 - same_ind_diff_period_length * rho[2]);

    out.H_rho(0, 0) += IK * K_minus_1 * diff_ind_same_period_length;
    out.H_rho(1, 1) += IK * K_minus_1 * diff_ind_and_period_length;
    out.H_rho(2, 2) += IK * same_ind_diff_period_length;

    idx_val += static_cast<int>(TJi);
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_stoch_gaussian_tscs(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_sampled& samp) {

  const arma::uword p = beta.n_elem;
  ngee_hier_totals out;
  out.H_beta.zeros(p, p);
  out.G_beta.zeros(p);
  out.H_phi = 0.0;
  out.G_phi = 0.0;
  out.H_rho.zeros(3, 3);
  out.G_rho.zeros(3);

  const arma::colvec mu = design_mat * beta;
  const arma::colvec resid = outcome - mu;
  const arma::colvec std_resid = resid / std::sqrt(phi);

  const arma::uword I = samp.idx.n_cluster;
  const double I_full = static_cast<double>(samp.full_n_cluster);

  int idx_val = 0;
  for (arma::uword i = 0; i < I; ++i) {
    const arma::uword T = samp.idx.cluster_n_sub[i];
    const double Ji = static_cast<double>(samp.idx.cluster_subject_n[i]);
    const double J_full = static_cast<double>(samp.full_cluster_subject_n[i]);
    const double TJi = static_cast<double>(samp.idx.cluster_size[i]);

    if (Ji <= 1.0) {
      Rcpp::stop("stochastic block-exchangeable fitting requires at least 2 sampled subjects per period.");
    }

    const double lambda1 = 1.0 - rho[0] + rho[1] - rho[2];
    const double lambda2 = 1.0 - rho[0] - (static_cast<double>(T) - 1.0) * (rho[1] - rho[2]);
    const double lambda3 = 1.0 + (J_full - 1.0) * (rho[0] - rho[1]) - rho[2];
    const double lambda4 = 1.0 + (J_full - 1.0) * rho[0] +
      (static_cast<double>(T) - 1.0) * (J_full - 1.0) * rho[1] +
      (static_cast<double>(T) - 1.0) * rho[2];

    const double c1 = 1.0 / lambda1;
    const double c2 = (rho[2] - rho[1]) / (lambda1 * lambda2);
    const double c3 = (rho[0] - rho[1]) / (lambda1 * lambda3);
    const double c4 = (rho[2] - rho[1]) * (rho[0] - rho[1]) / (lambda1 * lambda2 * lambda3) +
      (rho[2] * rho[0] - rho[1]) / (lambda2 * lambda3 * lambda4);

    const double IK = I_full * J_full / (static_cast<double>(I) * Ji);
    const double K_minus_1 = (J_full - 1.0) / (Ji - 1.0);

    const double c1_tilde = c1 * IK + c3 * IK * (K_minus_1 - 1.0);
    const double c2_tilde = c2 * IK + c4 * IK * (K_minus_1 - 1.0);
    const double c3_tilde = c3 * IK * K_minus_1;
    const double c4_tilde = c4 * IK * K_minus_1;

    arma::mat Xtilde = design_mat.rows(idx_val, idx_val + static_cast<int>(TJi) - 1);
    Xtilde /= std::sqrt(phi);
    const arma::colvec Etilde = std_resid.rows(idx_val, idx_val + static_cast<int>(TJi) - 1);

    const arma::rowvec Xtilde_sum = arma::sum(Xtilde, 0);
    const double Etilde_sum = arma::sum(Etilde);

    arma::mat Xtilde_sub(static_cast<arma::uword>(Ji), p, arma::fill::zeros);
    arma::colvec Etilde_sub(static_cast<arma::uword>(Ji), arma::fill::zeros);
    int idx_val2 = 0;

    double resid2_0 = 0.0;
    double resid2_1 = 0.0;
    double resid2_2 = 0.0;

    for (arma::uword j = 0; j < T; ++j) {
      const arma::uword Jsubj_u = samp.idx.sub_size[samp.idx.cluster_sub_start[i] + j];
      const double Jsubj = static_cast<double>(Jsubj_u);

      arma::mat temp_mat_X = Xtilde.rows(idx_val2, idx_val2 + static_cast<int>(Jsubj) - 1);
      arma::rowvec temp_vec_X = arma::sum(temp_mat_X, 0);
      arma::colvec temp_vec_E = Etilde.rows(idx_val2, idx_val2 + static_cast<int>(Jsubj) - 1);
      double temp_double_E = arma::sum(temp_vec_E);

      Xtilde_sub += temp_mat_X;
      Etilde_sub += temp_vec_E;

      out.G_beta += -c3_tilde * temp_vec_X.t() * temp_double_E;
      out.H_beta += -c3_tilde * temp_vec_X.t() * temp_vec_X;

      const double resid_phi = IK * arma::dot(temp_vec_E, temp_vec_E);
      resid2_0 = (temp_double_E * temp_double_E - resid_phi / IK) / 2.0;
      out.G_rho[0] += IK * K_minus_1 * (resid2_0 - Jsubj * (Jsubj - 1.0) * rho[0] / 2.0);
      resid2_1 -= resid_phi / IK / 2.0 + resid2_0;
      resid2_2 -= resid_phi / IK / 2.0;

      idx_val2 += static_cast<int>(Jsubj);
    }

    out.G_beta += c1_tilde * Xtilde.t() * Etilde -
      c2_tilde * Xtilde_sub.t() * Etilde_sub +
      c4_tilde * Xtilde_sum.t() * Etilde_sum;
    out.H_beta += c1_tilde * Xtilde.t() * Xtilde -
      c2_tilde * Xtilde_sub.t() * Xtilde_sub +
      c4_tilde * Xtilde_sum.t() * Xtilde_sum;

    out.G_phi += IK * phi * arma::sum(arma::square(Etilde) - 1.0);
    out.H_phi += IK * TJi;

    const double diff_ind_same_period_length = arma::sum(arma::conv_to<arma::vec>::from(
      samp.idx.sub_size.subvec(samp.idx.cluster_sub_start[i], samp.idx.cluster_sub_start[i] + T - 1)) %
        (arma::conv_to<arma::vec>::from(
            samp.idx.sub_size.subvec(samp.idx.cluster_sub_start[i], samp.idx.cluster_sub_start[i] + T - 1)) - 1.0) / 2.0);
    const double same_ind_diff_period_length = static_cast<double>(T) * (static_cast<double>(T) - 1.0) * Ji / 2.0;
    const double diff_ind_and_period_length = TJi * (TJi - 1.0) / 2.0 - diff_ind_same_period_length - same_ind_diff_period_length;

    resid2_2 += arma::dot(Etilde_sub, Etilde_sub) / 2.0;
    resid2_1 += Etilde_sum * Etilde_sum / 2.0 - resid2_2;

    out.G_rho[1] += IK * K_minus_1 * (resid2_1 - diff_ind_and_period_length * rho[1]);
    out.G_rho[2] += IK * (resid2_2 - same_ind_diff_period_length * rho[2]);

    out.H_rho(0, 0) += IK * K_minus_1 * diff_ind_same_period_length;
    out.H_rho(1, 1) += IK * K_minus_1 * diff_ind_and_period_length;
    out.H_rho(2, 2) += IK * same_ind_diff_period_length;

    idx_val += static_cast<int>(TJi);
  }

  return out;
}

static inline ngee_hier_totals ngee_kernel_stoch_tscs_dispatch(
    const arma::colvec& beta,
    const double phi,
    const arma::colvec& rho,
    const arma::colvec& outcome,
    const arma::mat& design_mat,
    const ngee_tscs_sampled& samp,
    const std::string& family) {
  if (family == "binomial") {
    return ngee_kernel_stoch_binomial_tscs(beta, rho, outcome, design_mat, samp);
  }
  return ngee_kernel_stoch_gaussian_tscs(beta, phi, rho, outcome, design_mat, samp);
}

/*
 Stochastic hierarchical fit.

 Unlike the deterministic fit, the stochastic updates use a weighted subsampled
 estimating equation whose coefficients depend on the original full-sample
 cluster/subcluster sizes attached to the sampled units.  This preserves the
 original stochastic NGEE construction rather than simply running deterministic
 kernels on a subsample.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_stoch_hier_cpp(const arma::colvec& outcome,
                                   const arma::mat& design_mat,
                                   const Rcpp::List& index_data,
                                   const std::string& family,
                                   const std::string& corstr,
                                   arma::colvec beta0,
                                   double phi0,
                                   arma::colvec rho0,
                                   const arma::uvec& batch_size,
                                   const int burnin = 50,
                                   const int avgiter = 50,
                                   const double phi_min = 1e-8,
                                   const double rho_eps = 1e-4,
                                   const bool final_refine = true) {
  const ngee_hier_index_view idx = ngee_get_hier_index_view(index_data);

  if (idx.n_obs != outcome.n_elem) Rcpp::stop("index_data$n_obs does not match outcome length.");
  if (design_mat.n_rows != outcome.n_elem) Rcpp::stop("design_mat and outcome have incompatible sizes.");
  if (beta0.n_elem != design_mat.n_cols) Rcpp::stop("beta0 has wrong length.");
  if (!(family == "binomial" || family == "gaussian")) {
    Rcpp::stop("family must be 'binomial' or 'gaussian'.");
  }
  if (!(corstr == "independence" || corstr == "simple-exchangeable" || corstr == "nested-exchangeable")) {
    Rcpp::stop("Unsupported corstr for stochastic hierarchical fit.");
  }
  if (Rcpp::as<std::string>(index_data["corstr"]) != corstr) {
    Rcpp::stop("corstr does not match index_data$corstr.");
  }
  if (batch_size.n_elem == 0) {
    Rcpp::stop("batch_size must have positive length.");
  }
  if (burnin < 1 || avgiter < 1) {
    Rcpp::stop("burnin and avgiter must both be >= 1.");
  }
  if (corstr == "simple-exchangeable" && batch_size.n_elem >= 2 && batch_size[1] < 2) {
    Rcpp::stop("simple-exchangeable stochastic fitting requires batch_size[2] >= 2.");
  }
  if (corstr == "simple-exchangeable") {
    if (idx.cluster_size.n_elem > 0 && idx.cluster_size.min() < 2) {
      Rcpp::stop("simple-exchangeable stochastic fitting requires at least 2 observations in every full-data cluster.");
    }
  }
  if (corstr == "nested-exchangeable") {
    if (batch_size.n_elem < 3 || batch_size[1] < 2 || batch_size[2] < 2) {
      Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 sampled subclusters and 2 sampled observations per sampled subcluster.");
    }
    if (idx.cluster_n_sub.n_elem > 0 && idx.cluster_n_sub.min() < 2) {
      Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 subclusters in every full-data cluster.");
    }
    if (idx.sub_size.n_elem > 0 && idx.sub_size.min() < 2) {
      Rcpp::stop("nested-exchangeable stochastic fitting requires at least 2 observations in every full-data subcluster.");
    }
  }

  const bool update_phi = (family == "gaussian");
  const bool update_rho = (corstr == "simple-exchangeable" || corstr == "nested-exchangeable");

  if (corstr == "independence") {
    if (rho0.n_elem < 1) rho0 = arma::zeros<arma::colvec>(1);
  } else if (corstr == "simple-exchangeable") {
    if (rho0.n_elem != 1) Rcpp::stop("simple-exchangeable requires rho0 of length 1.");
    ngee_stabilize_simple_rho(rho0, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    if (rho0.n_elem != 2) Rcpp::stop("nested-exchangeable requires rho0 of length 2.");
    ngee_stabilize_nested_rho(rho0, idx, rho_eps);
  }

  arma::colvec beta = beta0;
  arma::colvec rho = rho0;
  double phi = std::max(phi0, phi_min);

  arma::colvec dbeta(beta.n_elem, arma::fill::zeros);
  arma::colvec drho(rho.n_elem, arma::fill::zeros);
  double dphi = 0.0;

  // initial subsampled Newton step
  {
    if (corstr == "simple-exchangeable") {
      ngee_stabilize_simple_rho(rho, idx, rho_eps);
    } else if (corstr == "nested-exchangeable") {
      ngee_stabilize_nested_rho(rho, idx, rho_eps);
    }

    ngee_hier_sampled samp = ngee_sample_hier_fast(idx, batch_size, false);
    arma::colvec outcome_sub = outcome.elem(samp.obs_index0);
    arma::mat design_sub = design_mat.rows(samp.obs_index0);

    ngee_hier_totals blocks = ngee_kernel_stoch_hier_dispatch(
      beta, phi, rho, outcome_sub, design_sub, samp, family, corstr
    );

    dbeta = ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);
    beta += dbeta;

    if (update_phi && blocks.H_phi > 0.0) {
      dphi = blocks.G_phi / blocks.H_phi;
      phi = std::max(phi + dphi, phi_min);
    }

    if (update_rho) {
      drho = ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
      rho += drho;

      if (corstr == "simple-exchangeable") {
        ngee_stabilize_simple_rho(rho, idx, rho_eps);
      } else {
        ngee_stabilize_nested_rho(rho, idx, rho_eps);
      }
    }
  }

  // burn-in
  for (int iter = 1; iter < burnin; ++iter) {
    if (corstr == "simple-exchangeable") {
      ngee_stabilize_simple_rho(rho, idx, rho_eps);
    } else if (corstr == "nested-exchangeable") {
      ngee_stabilize_nested_rho(rho, idx, rho_eps);
    }

    ngee_hier_sampled samp = ngee_sample_hier_fast(idx, batch_size, false);
    arma::colvec outcome_sub = outcome.elem(samp.obs_index0);
    arma::mat design_sub = design_mat.rows(samp.obs_index0);

    ngee_hier_totals blocks = ngee_kernel_stoch_hier_dispatch(
      beta, phi, rho, outcome_sub, design_sub, samp, family, corstr
    );

    const double lr = ngee_stoch_lr(iter);

    dbeta = ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);
    beta += lr * dbeta;

    if (update_phi && blocks.H_phi > 0.0) {
      dphi = blocks.G_phi / blocks.H_phi;
      phi = std::max(phi + lr * dphi, phi_min);
    }

    if (update_rho) {
      drho = ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
      rho += lr * drho;

      if (corstr == "simple-exchangeable") {
        ngee_stabilize_simple_rho(rho, idx, rho_eps);
      } else {
        ngee_stabilize_nested_rho(rho, idx, rho_eps);
      }
    }
  }

  // averaging phase
  arma::colvec beta_avg(beta.n_elem, arma::fill::zeros);
  arma::colvec rho_avg(rho.n_elem, arma::fill::zeros);
  double phi_avg = 0.0;

  for (int iter = burnin; iter < burnin + avgiter; ++iter) {
    if (corstr == "simple-exchangeable") {
      ngee_stabilize_simple_rho(rho, idx, rho_eps);
    } else if (corstr == "nested-exchangeable") {
      ngee_stabilize_nested_rho(rho, idx, rho_eps);
    }

    ngee_hier_sampled samp = ngee_sample_hier_fast(idx, batch_size, false);
    arma::colvec outcome_sub = outcome.elem(samp.obs_index0);
    arma::mat design_sub = design_mat.rows(samp.obs_index0);

    ngee_hier_totals blocks = ngee_kernel_stoch_hier_dispatch(
      beta, phi, rho, outcome_sub, design_sub, samp, family, corstr
    );

    const double lr = ngee_stoch_lr(iter);

    dbeta = ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);
    beta += lr * dbeta;
    beta_avg += beta;

    if (update_phi && blocks.H_phi > 0.0) {
      dphi = blocks.G_phi / blocks.H_phi;
      phi = std::max(phi + lr * dphi, phi_min);
      phi_avg += phi;
    }

    if (update_rho) {
      drho = ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
      rho += lr * drho;

      if (corstr == "simple-exchangeable") {
        ngee_stabilize_simple_rho(rho, idx, rho_eps);
      } else {
        ngee_stabilize_nested_rho(rho, idx, rho_eps);
      }

      rho_avg += rho;
    }
  }

  beta = beta_avg / static_cast<double>(avgiter);
  if (update_phi) {
    phi = phi_avg / static_cast<double>(avgiter);
  }
  if (update_rho) {
    rho = rho_avg / static_cast<double>(avgiter);
  }

  if (corstr == "simple-exchangeable") {
    ngee_stabilize_simple_rho(rho, idx, rho_eps);
  } else if (corstr == "nested-exchangeable") {
    ngee_stabilize_nested_rho(rho, idx, rho_eps);
  }

  arma::colvec beta_prerefine = beta;
  double phi_prerefine = phi;
  arma::colvec rho_prerefine = rho;

  if (final_refine) {
    ngee_det_one_step_hier(
      beta, phi, rho, outcome, design_mat, idx, family, corstr, phi_min, rho_eps
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("beta_prerefine") = beta_prerefine,
    Rcpp::Named("phi_prerefine") = phi_prerefine,
    Rcpp::Named("rho_prerefine") = rho_prerefine,
    Rcpp::Named("burnin") = burnin,
    Rcpp::Named("avgiter") = avgiter,
    Rcpp::Named("n_iter_total") = burnin + avgiter,
    Rcpp::Named("final_refine") = final_refine,
    Rcpp::Named("completed") = true
  );
}

/*
 High-level stochastic hierarchical fit with deterministic sandwich evaluation at
 its final returned parameter values.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_stoch_hier_se_cpp(const arma::colvec& outcome,
                                      const arma::mat& design_mat,
                                      const Rcpp::List& index_data,
                                      const std::string& family,
                                      const std::string& corstr,
                                      arma::colvec beta0,
                                      double phi0,
                                      arma::colvec rho0,
                                      const arma::uvec& batch_size,
                                      const std::string& se_adjust = "unadjusted",
                                      const int burnin = 50,
                                      const int avgiter = 50,
                                      const double phi_min = 1e-8,
                                      const double fg_cap = 0.85,
                                      const double rho_eps = 1e-4,
                                      const bool final_refine = true) {

  Rcpp::List fit = ngee_fit_stoch_hier_cpp(
    outcome,
    design_mat,
    index_data,
    family,
    corstr,
    beta0,
    phi0,
    rho0,
    batch_size,
    burnin,
    avgiter,
    phi_min,
    rho_eps,
    final_refine
  );

  const arma::colvec beta = Rcpp::as<arma::colvec>(fit["beta"]);
  const double phi = Rcpp::as<double>(fit["phi"]);
  const arma::colvec rho = Rcpp::as<arma::colvec>(fit["rho"]);

  const arma::colvec beta_prerefine = Rcpp::as<arma::colvec>(fit["beta_prerefine"]);
  const double phi_prerefine = Rcpp::as<double>(fit["phi_prerefine"]);
  const arma::colvec rho_prerefine = Rcpp::as<arma::colvec>(fit["rho_prerefine"]);

  Rcpp::List sand = ngee_hier_sandwich_det_cpp(
    beta,
    phi,
    rho,
    outcome,
    design_mat,
    index_data,
    family,
    corstr,
    se_adjust,
    fg_cap,
    rho_eps
  );

  const arma::mat var_sandwich = Rcpp::as<arma::mat>(sand["var_sandwich"]);
  const arma::mat Info = Rcpp::as<arma::mat>(sand["Info"]);
  const arma::colvec se = ngee_diag_se(var_sandwich);
  const arma::colvec coef = ngee_pack_active_coef_hier(beta, phi, rho, family, corstr);

  if (coef.n_elem != se.n_elem) {
    Rcpp::stop("Internal dimension mismatch: coef and sandwich SE lengths differ.");
  }

  arma::colvec z = coef / se;
  for (arma::uword j = 0; j < z.n_elem; ++j) {
    if (!std::isfinite(z[j])) z[j] = NA_REAL;
  }

  const arma::uword p = beta.n_elem;
  const arma::uword q_active = ngee_active_q_hier(corstr);
  const bool include_phi = (family == "gaussian");

  arma::colvec beta_se = se.subvec(0, p - 1);

  arma::colvec phi_se;
  phi_se.zeros(include_phi ? 1 : 0);
  if (include_phi) {
    phi_se[0] = se[p];
  }

  arma::colvec rho_se;
  rho_se.zeros(q_active);
  if (q_active > 0) {
    const arma::uword rho_start = p + static_cast<arma::uword>(include_phi);
    rho_se = se.subvec(rho_start, rho_start + q_active - 1);
  }

  return Rcpp::List::create(
    Rcpp::Named("family") = family,
    Rcpp::Named("corstr") = corstr,
    Rcpp::Named("se_adjust") = se_adjust,
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("beta_prerefine") = beta_prerefine,
    Rcpp::Named("phi_prerefine") = phi_prerefine,
    Rcpp::Named("rho_prerefine") = rho_prerefine,
    Rcpp::Named("beta_se") = beta_se,
    Rcpp::Named("phi_se") = phi_se,
    Rcpp::Named("rho_se") = rho_se,
    Rcpp::Named("coef") = coef,
    Rcpp::Named("se") = se,
    Rcpp::Named("z") = z,
    Rcpp::Named("param_names") = ngee_active_param_names_hier(p, family, corstr),
    Rcpp::Named("var_sandwich") = var_sandwich,
    Rcpp::Named("Info") = Info,
    Rcpp::Named("burnin") = burnin,
    Rcpp::Named("avgiter") = avgiter,
    Rcpp::Named("n_iter_total") = burnin + avgiter,
    Rcpp::Named("final_refine") = final_refine,
    Rcpp::Named("completed") = true
  );
}

/***** STOCHASTIC BLOCK-EXCHANGEABLE FIT + SANDWICH ****************************
 Supports:
 - family: "binomial", "gaussian"
 - corstr fixed as "block-exchangeable"

 Uses:
 - ngee_sample_index_cpp() with block-exchangeable sampling
 - ngee_kernel_tscs_dispatch() for stochastic score/Hessian blocks
 - optional final full-data deterministic refinement
 - deterministic sandwich at final parameter values
 *******************************************************************************/

static inline void ngee_det_one_step_tscs(arma::colvec& beta,
                                          double& phi,
                                          arma::colvec& rho,
                                          const arma::colvec& outcome,
                                          const arma::mat& design_mat,
                                          const ngee_tscs_index_view& idx,
                                          const std::string& family,
                                          const double phi_min,
                                          const double rho_eps) {
  const bool update_phi = (family == "gaussian");

  ngee_stabilize_block_rho(rho, idx, rho_eps);

  ngee_hier_totals blocks = ngee_kernel_tscs_dispatch(
    beta, phi, rho, outcome, design_mat, idx, family
  );

  beta += ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);

  if (update_phi) {
    phi = std::max(phi + blocks.G_phi / blocks.H_phi, phi_min);
  }

  rho += ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
  ngee_stabilize_block_rho(rho, idx, rho_eps);
}

/*
 Stochastic block-exchangeable fit.

 The stochastic update uses the weighted TSCS estimating equations from the
 legacy implementation, but now against the flat sampled index representation.
 For block-exchangeable fits, batch_size is interpreted as
 batch_size[0] = number of sampled clusters
 batch_size[1] = number of sampled aligned subjects per sampled cluster
 All periods are retained for each sampled cluster.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_stoch_tscs_cpp(const arma::colvec& outcome,
                                   const arma::mat& design_mat,
                                   const Rcpp::List& index_data,
                                   const std::string& family,
                                   arma::colvec beta0,
                                   double phi0,
                                   arma::colvec rho0,
                                   const arma::uvec& batch_size,
                                   const int burnin = 50,
                                   const int avgiter = 50,
                                   const double phi_min = 1e-8,
                                   const double rho_eps = 1e-4,
                                   const bool final_refine = true) {
  const ngee_tscs_index_view idx = ngee_get_tscs_index_view(index_data);

  if (idx.corstr != "block-exchangeable") {
    Rcpp::stop("index_data$corstr must be 'block-exchangeable'.");
  }
  if (idx.n_obs != outcome.n_elem) {
    Rcpp::stop("index_data$n_obs does not match outcome length.");
  }
  if (design_mat.n_rows != outcome.n_elem) {
    Rcpp::stop("design_mat and outcome have incompatible sizes.");
  }
  if (beta0.n_elem != design_mat.n_cols) {
    Rcpp::stop("beta0 has wrong length.");
  }
  if (!(family == "binomial" || family == "gaussian")) {
    Rcpp::stop("family must be 'binomial' or 'gaussian'.");
  }
  if (rho0.n_elem != 3) {
    Rcpp::stop("block-exchangeable requires rho0 of length 3.");
  }
  if (batch_size.n_elem < 2) {
    Rcpp::stop("For stochastic block-exchangeable, batch_size must have length >= 2.");
  }
  if (batch_size[1] < 2) {
    Rcpp::stop("stochastic block-exchangeable fitting requires at least 2 sampled subjects per period.");
  }
  if (burnin < 1 || avgiter < 1) {
    Rcpp::stop("burnin and avgiter must both be >= 1.");
  }

  for (arma::uword i = 0; i < idx.n_cluster; ++i) {
    if (!idx.cluster_balanced[i]) {
      Rcpp::stop("block-exchangeable requires equal period sizes within each cluster.");
    }
    if (idx.cluster_subject_n[i] < 2) {
      Rcpp::stop("stochastic block-exchangeable fitting requires at least 2 subjects per period in every full-data cluster.");
    }
  }

  const bool update_phi = (family == "gaussian");

  arma::colvec beta = beta0;
  arma::colvec rho = rho0;
  double phi = std::max(phi0, phi_min);

  ngee_stabilize_block_rho(rho, idx, rho_eps);

  arma::colvec dbeta(beta.n_elem, arma::fill::zeros);
  arma::colvec drho(rho.n_elem, arma::fill::zeros);
  double dphi = 0.0;

  // initial subsampled Newton step
  {
    ngee_stabilize_block_rho(rho, idx, rho_eps);

    ngee_tscs_sampled samp = ngee_sample_tscs_fast(idx, batch_size, false);
    arma::colvec outcome_sub = outcome.elem(samp.obs_index0);
    arma::mat design_sub = design_mat.rows(samp.obs_index0);

    ngee_hier_totals blocks = ngee_kernel_stoch_tscs_dispatch(
      beta, phi, rho, outcome_sub, design_sub, samp, family
    );

    dbeta = ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);
    beta += dbeta;

    if (update_phi && blocks.H_phi > 0.0) {
      dphi = blocks.G_phi / blocks.H_phi;
      phi = std::max(phi + dphi, phi_min);
    }

    drho = ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
    rho += drho;
    ngee_stabilize_block_rho(rho, idx, rho_eps);
  }

  // burn-in
  for (int iter = 1; iter < burnin; ++iter) {
    ngee_stabilize_block_rho(rho, idx, rho_eps);

    ngee_tscs_sampled samp = ngee_sample_tscs_fast(idx, batch_size, false);
    arma::colvec outcome_sub = outcome.elem(samp.obs_index0);
    arma::mat design_sub = design_mat.rows(samp.obs_index0);

    ngee_hier_totals blocks = ngee_kernel_stoch_tscs_dispatch(
      beta, phi, rho, outcome_sub, design_sub, samp, family
    );

    const double lr = ngee_stoch_lr(iter);

    dbeta = ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);
    beta += lr * dbeta;

    if (update_phi && blocks.H_phi > 0.0) {
      dphi = blocks.G_phi / blocks.H_phi;
      phi = std::max(phi + lr * dphi, phi_min);
    }

    drho = ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
    rho += lr * drho;
    ngee_stabilize_block_rho(rho, idx, rho_eps);
  }

  // averaging phase
  arma::colvec beta_avg(beta.n_elem, arma::fill::zeros);
  arma::colvec rho_avg(rho.n_elem, arma::fill::zeros);
  double phi_avg = 0.0;

  for (int iter = burnin; iter < burnin + avgiter; ++iter) {
    ngee_stabilize_block_rho(rho, idx, rho_eps);

    ngee_tscs_sampled samp = ngee_sample_tscs_fast(idx, batch_size, false);
    arma::colvec outcome_sub = outcome.elem(samp.obs_index0);
    arma::mat design_sub = design_mat.rows(samp.obs_index0);

    ngee_hier_totals blocks = ngee_kernel_stoch_tscs_dispatch(
      beta, phi, rho, outcome_sub, design_sub, samp, family
    );

    const double lr = ngee_stoch_lr(iter);

    dbeta = ngee_safe_solve_vec(blocks.H_beta, blocks.G_beta);
    beta += lr * dbeta;
    beta_avg += beta;

    if (update_phi && blocks.H_phi > 0.0) {
      dphi = blocks.G_phi / blocks.H_phi;
      phi = std::max(phi + lr * dphi, phi_min);
      phi_avg += phi;
    }

    drho = ngee_safe_solve_vec(blocks.H_rho, blocks.G_rho);
    rho += lr * drho;
    ngee_stabilize_block_rho(rho, idx, rho_eps);
    rho_avg += rho;
  }

  beta = beta_avg / static_cast<double>(avgiter);
  if (update_phi) phi = phi_avg / static_cast<double>(avgiter);
  rho = rho_avg / static_cast<double>(avgiter);

  ngee_stabilize_block_rho(rho, idx, rho_eps);

  arma::colvec beta_prerefine = beta;
  double phi_prerefine = phi;
  arma::colvec rho_prerefine = rho;

  if (final_refine) {
    ngee_det_one_step_tscs(
      beta, phi, rho, outcome, design_mat, idx, family, phi_min, rho_eps
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("beta_prerefine") = beta_prerefine,
    Rcpp::Named("phi_prerefine") = phi_prerefine,
    Rcpp::Named("rho_prerefine") = rho_prerefine,
    Rcpp::Named("burnin") = burnin,
    Rcpp::Named("avgiter") = avgiter,
    Rcpp::Named("n_iter_total") = burnin + avgiter,
    Rcpp::Named("final_refine") = final_refine,
    Rcpp::Named("completed") = true
  );
}

/*
 High-level stochastic block-exchangeable fit with deterministic sandwich
 covariance evaluated at its final returned parameter values.
 */
// [[Rcpp::export]]
Rcpp::List ngee_fit_stoch_tscs_se_cpp(const arma::colvec& outcome,
                                      const arma::mat& design_mat,
                                      const Rcpp::List& index_data,
                                      const std::string& family,
                                      arma::colvec beta0,
                                      double phi0,
                                      arma::colvec rho0,
                                      const arma::uvec& batch_size,
                                      const std::string& se_adjust = "unadjusted",
                                      const int burnin = 50,
                                      const int avgiter = 50,
                                      const double phi_min = 1e-8,
                                      const double fg_cap = 0.85,
                                      const double rho_eps = 1e-4,
                                      const bool final_refine = true) {

  Rcpp::List fit = ngee_fit_stoch_tscs_cpp(
    outcome,
    design_mat,
    index_data,
    family,
    beta0,
    phi0,
    rho0,
    batch_size,
    burnin,
    avgiter,
    phi_min,
    rho_eps,
    final_refine
  );

  const arma::colvec beta = Rcpp::as<arma::colvec>(fit["beta"]);
  const double phi = Rcpp::as<double>(fit["phi"]);
  const arma::colvec rho = Rcpp::as<arma::colvec>(fit["rho"]);

  const arma::colvec beta_prerefine = Rcpp::as<arma::colvec>(fit["beta_prerefine"]);
  const double phi_prerefine = Rcpp::as<double>(fit["phi_prerefine"]);
  const arma::colvec rho_prerefine = Rcpp::as<arma::colvec>(fit["rho_prerefine"]);

  Rcpp::List sand = ngee_tscs_sandwich_det_cpp(
    beta,
    phi,
    rho,
    outcome,
    design_mat,
    index_data,
    family,
    se_adjust,
    fg_cap,
    rho_eps
  );

  const arma::mat var_sandwich = Rcpp::as<arma::mat>(sand["var_sandwich"]);
  const arma::mat Info = Rcpp::as<arma::mat>(sand["Info"]);
  const arma::colvec se = ngee_diag_se(var_sandwich);
  const arma::colvec coef = ngee_pack_active_coef_tscs(beta, phi, rho, family);

  if (coef.n_elem != se.n_elem) {
    Rcpp::stop("Internal dimension mismatch: coef and sandwich SE lengths differ.");
  }

  arma::colvec z = coef / se;
  for (arma::uword j = 0; j < z.n_elem; ++j) {
    if (!std::isfinite(z[j])) z[j] = NA_REAL;
  }

  const arma::uword p = beta.n_elem;
  const bool include_phi = (family == "gaussian");

  arma::colvec beta_se = se.subvec(0, p - 1);

  arma::colvec phi_se;
  phi_se.zeros(include_phi ? 1 : 0);
  if (include_phi) {
    phi_se[0] = se[p];
  }

  arma::colvec rho_se(3, arma::fill::zeros);
  rho_se = se.subvec(p + static_cast<arma::uword>(include_phi),
                     p + static_cast<arma::uword>(include_phi) + 2);

  return Rcpp::List::create(
    Rcpp::Named("family") = family,
    Rcpp::Named("corstr") = "block-exchangeable",
    Rcpp::Named("se_adjust") = se_adjust,
    Rcpp::Named("beta") = beta,
    Rcpp::Named("phi") = phi,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("beta_prerefine") = beta_prerefine,
    Rcpp::Named("phi_prerefine") = phi_prerefine,
    Rcpp::Named("rho_prerefine") = rho_prerefine,
    Rcpp::Named("beta_se") = beta_se,
    Rcpp::Named("phi_se") = phi_se,
    Rcpp::Named("rho_se") = rho_se,
    Rcpp::Named("coef") = coef,
    Rcpp::Named("se") = se,
    Rcpp::Named("z") = z,
    Rcpp::Named("param_names") = ngee_active_param_names_tscs(p, family),
    Rcpp::Named("var_sandwich") = var_sandwich,
    Rcpp::Named("Info") = Info,
    Rcpp::Named("burnin") = burnin,
    Rcpp::Named("avgiter") = avgiter,
    Rcpp::Named("n_iter_total") = burnin + avgiter,
    Rcpp::Named("final_refine") = final_refine,
    Rcpp::Named("completed") = true
  );
}
