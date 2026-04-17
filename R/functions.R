#' Fit network generalized estimating equations
#'
#' Fits generalized estimating equations for correlated data under independence,
#' simple exchangeable, nested exchangeable, and block-exchangeable working
#' correlation structures.
#'
#' @param formula A model formula.
#' @param id Character string or character vector giving the clustering
#'   variables in `dat`. Use a single variable for `"independence"` and
#'   `"simple-exchangeable"`, and two variables for `"nested-exchangeable"` and
#'   `"block-exchangeable"`.
#' @param dat A data frame containing the variables in `formula` and `id`.
#' @param family Marginal outcome family. Currently `"gaussian"` and
#'   `"binomial"` are supported.
#' @param corstr Working correlation structure. One of `"independence"`,
#'   `"simple-exchangeable"`, `"nested-exchangeable"`, or
#'   `"block-exchangeable"`.
#' @param se_adjust Standard error adjustment. One of `"unadjusted"` or `"FG"`.
#' @param optim Fitting routine. One of `"deterministic"` or `"stochastic"`.
#' @param batch_size Integer vector controlling the subsample size at each
#'   stochastic iteration. Its interpretation depends on `corstr`.
#' @param burnin Number of initial stochastic iterations discarded before
#'   averaging.
#' @param avgiter Number of subsequent stochastic iterations whose parameter
#'   values are averaged (Polyak-Ruppert averaging).
#' @param tol Convergence tolerance for deterministic fitting.
#' @param maxit Maximum number of deterministic iterations.
#' @param phi_min Lower bound enforced on the Gaussian scale parameter.
#' @param fg_cap Upper cap used in the Fay-Graubard standard error adjustment.
#' @param rho_eps Small numerical tolerance used when enforcing admissible
#'   working correlation values.
#' @param final_refine Logical; if `TRUE`, stochastic fits take one final
#'   deterministic full-data update before returning.
#' @param compute_se Logical; if `FALSE`, skips sandwich variance estimation and
#'   returns only point estimates.
#'
#' @details
#' Data should already be sorted by the variables supplied in `id`.
#'
#' For stochastic fitting, `batch_size` is interpreted as follows:
#' \itemize{
#'   \item independence / simple exchangeable:
#'     \code{c(n_clusters, n_observations_per_cluster)}
#'   \item nested exchangeable:
#'     \code{c(n_clusters, n_subclusters_per_cluster, n_observations_per_subcluster)}
#'   \item block exchangeable:
#'     \code{c(n_clusters, n_subjects_per_period)}
#' }
#'
#' For block-exchangeable fits, rows must also be arranged so that within each
#' cluster, periods are contiguous, period sizes are equal, and repeated
#' individuals are aligned by row position across periods.
#'
#' @return
#' A list containing fitted regression coefficients and working correlation
#' parameters. When `compute_se = TRUE`, the returned object also includes the
#' sandwich covariance matrix and componentwise standard errors.
#'
#' @export
ngee = function(formula,
                id,
                dat,
                family = "gaussian",
                corstr = "independence",
                se_adjust = "unadjusted",
                optim = "deterministic",
                batch_size = c(30, 5, 2),
                burnin = 50,
                avgiter = 50,
                tol = 1e-6,
                maxit = 200,
                phi_min = 1e-8,
                fg_cap = 0.85,
                rho_eps = 1e-4,
                final_refine = TRUE,
                compute_se = TRUE) {

  if (!(family %in% c("binomial", "gaussian"))) stop("family not recognized")
  if (!(corstr %in% c("independence",
                      "simple-exchangeable",
                      "nested-exchangeable",
                      "block-exchangeable"))) {
    stop("corstr not recognized")
  }
  if (!(optim %in% c("deterministic", "stochastic"))) stop("optim not recognized")
  if (!(se_adjust %in% c("unadjusted", "FG"))) stop("se_adjust not recognized")

  # Build the model frame and design matrix implied by the user formula.
  mf <- model.frame(formula = as.formula(formula), data = dat)
  outcome <- model.response(mf)
  design_mat <- model.matrix(attr(mf, "terms"), data = mf)
  # Pass cluster identifiers as a numeric matrix to C++.  The row order must
  # already satisfy the sorting assumptions required by the chosen structure.
  clusterid <- as.matrix(dat[id, drop = FALSE])

  if (corstr %in% c("independence", "simple-exchangeable") && ncol(clusterid) != 1) {
    stop("independence/simple-exchangeable currently require a 1-column id.")
  }
  if (corstr %in% c("nested-exchangeable", "block-exchangeable") && ncol(clusterid) != 2) {
    stop("nested-exchangeable/block-exchangeable currently require a 2-column id.")
  }

  beta0 <- cbind(numeric(ncol(design_mat)))
  phi0 <- 1

  if (corstr == "independence") {
    rho0 <- 0
  } else if (corstr == "simple-exchangeable") {
    rho0 <- 0
  } else if (corstr == "nested-exchangeable") {
    rho0 <- c(0, 0)
  } else {
    rho0 <- c(0, 0, 0)
  }

  make_param_names <- function() {
    beta_names <- colnames(design_mat)
    if (is.null(beta_names)) {
      beta_names <- paste0("beta", seq_len(ncol(design_mat)))
    }

    if (family == "binomial" && corstr == "independence") {
      return(beta_names)
    }
    if (family == "binomial" && corstr == "simple-exchangeable") {
      return(c(beta_names, "rho1"))
    }
    if (family == "binomial" && corstr == "nested-exchangeable") {
      return(c(beta_names, "rho1", "rho2"))
    }
    if (family == "binomial" && corstr == "block-exchangeable") {
      return(c(beta_names, "rho1", "rho2", "rho3"))
    }

    if (family == "gaussian" && corstr == "independence") {
      return(c(beta_names, "phi"))
    }
    if (family == "gaussian" && corstr == "simple-exchangeable") {
      return(c(beta_names, "phi", "rho1"))
    }
    if (family == "gaussian" && corstr == "nested-exchangeable") {
      return(c(beta_names, "phi", "rho1", "rho2"))
    }
    c(beta_names, "phi", "rho1", "rho2", "rho3")
  }

  # Deterministic hierarchical route.
  if (optim == "deterministic" && corstr %in% c("independence", "simple-exchangeable", "nested-exchangeable")) {
    index_data <- ngee_precompute_index_cpp(
      cluster_id = clusterid,
      corstr = corstr,
      validate_sorted = TRUE,
      validate_balanced = TRUE
    )

    if (compute_se) {
      out <- ngee_fit_det_hier_se_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        corstr = corstr,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        se_adjust = se_adjust,
        tol = tol,
        maxit = maxit,
        phi_min = phi_min,
        fg_cap = fg_cap,
        rho_eps = rho_eps
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          serho = out$rho_se,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          sephi = as.numeric(out$phi_se),
          serho = out$rho_se,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      }
    } else {
      out <- ngee_fit_det_hier_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        corstr = corstr,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        tol = tol,
        maxit = maxit,
        phi_min = phi_min,
        rho_eps = rho_eps
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          vbetarho = NULL,
          sebeta = NULL,
          serho = NULL,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          vbetarho = NULL,
          sebeta = NULL,
          sephi = NULL,
          serho = NULL,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      }
    }
  }

  # Deterministic block-exchangeable route.
  if (optim == "deterministic" && corstr == "block-exchangeable") {
    index_data <- ngee_precompute_index_cpp(
      cluster_id = clusterid,
      corstr = "block-exchangeable",
      validate_sorted = TRUE,
      validate_balanced = TRUE
    )

    if (compute_se) {
      out <- ngee_fit_det_tscs_se_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        se_adjust = se_adjust,
        tol = tol,
        maxit = maxit,
        phi_min = phi_min,
        fg_cap = fg_cap,
        rho_eps = rho_eps
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          serho = out$rho_se,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          sephi = as.numeric(out$phi_se),
          serho = out$rho_se,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      }
    } else {
      out <- ngee_fit_det_tscs_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        tol = tol,
        maxit = maxit,
        phi_min = phi_min,
        rho_eps = rho_eps
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          vbetarho = NULL,
          sebeta = NULL,
          serho = NULL,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          vbetarho = NULL,
          sebeta = NULL,
          sephi = NULL,
          serho = NULL,
          iter = out$iter,
          converged = out$converged,
          final_error = out$final_error,
          param_names = make_param_names()
        ))
      }
    }
  }

  # Stochastic hierarchical route.
  # final_refine controls whether a final full-data deterministic Newton step is
  # taken after stochastic averaging.
  if (optim == "stochastic" && corstr %in% c("independence", "simple-exchangeable", "nested-exchangeable")) {
    index_data <- ngee_precompute_index_cpp(
      cluster_id = clusterid,
      corstr = corstr,
      validate_sorted = TRUE,
      validate_balanced = TRUE
    )

    levels_num <- if (corstr == "nested-exchangeable") 2 else 1
    batch_use <- batch_size[1:(levels_num + 1)]

    if (compute_se) {
      out <- ngee_fit_stoch_hier_se_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        corstr = corstr,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        batch_size = batch_use,
        se_adjust = se_adjust,
        burnin = burnin,
        avgiter = avgiter,
        phi_min = phi_min,
        fg_cap = fg_cap,
        rho_eps = rho_eps,
        final_refine = final_refine
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          rho_prerefine = out$rho_prerefine,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          serho = out$rho_se,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          phi_prerefine = out$phi_prerefine,
          rho_prerefine = out$rho_prerefine,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          sephi = as.numeric(out$phi_se),
          serho = out$rho_se,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      }
    } else {
      out <- ngee_fit_stoch_hier_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        corstr = corstr,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        batch_size = batch_use,
        burnin = burnin,
        avgiter = avgiter,
        phi_min = phi_min,
        rho_eps = rho_eps,
        final_refine = final_refine
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          rho_prerefine = out$rho_prerefine,
          vbetarho = NULL,
          sebeta = NULL,
          serho = NULL,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          phi_prerefine = out$phi_prerefine,
          rho_prerefine = out$rho_prerefine,
          vbetarho = NULL,
          sebeta = NULL,
          sephi = NULL,
          serho = NULL,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      }
    }
  }

  # Stochastic block-exchangeable route.
  if (optim == "stochastic" && corstr == "block-exchangeable") {
    index_data <- ngee_precompute_index_cpp(
      cluster_id = clusterid,
      corstr = "block-exchangeable",
      validate_sorted = TRUE,
      validate_balanced = TRUE
    )

    batch_use <- batch_size[1:2]

    if (compute_se) {
      out <- ngee_fit_stoch_tscs_se_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        batch_size = batch_use,
        se_adjust = se_adjust,
        burnin = burnin,
        avgiter = avgiter,
        phi_min = phi_min,
        fg_cap = fg_cap,
        rho_eps = rho_eps,
        final_refine = final_refine
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          rho_prerefine = out$rho_prerefine,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          serho = out$rho_se,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          phi_prerefine = out$phi_prerefine,
          rho_prerefine = out$rho_prerefine,
          Info = out$Info,
          vbetarho = out$var_sandwich,
          sebeta = out$beta_se,
          sephi = as.numeric(out$phi_se),
          serho = out$rho_se,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      }
    } else {
      out <- ngee_fit_stoch_tscs_cpp(
        outcome = outcome,
        design_mat = design_mat,
        index_data = index_data,
        family = family,
        beta0 = beta0,
        phi0 = phi0,
        rho0 = rho0,
        batch_size = batch_use,
        burnin = burnin,
        avgiter = avgiter,
        phi_min = phi_min,
        rho_eps = rho_eps,
        final_refine = final_refine
      )

      if (family == "binomial") {
        return(list(
          beta = out$beta,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          rho_prerefine = out$rho_prerefine,
          vbetarho = NULL,
          sebeta = NULL,
          serho = NULL,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      } else {
        return(list(
          beta = out$beta,
          phi = out$phi,
          rho = out$rho,
          beta_prerefine = out$beta_prerefine,
          phi_prerefine = out$phi_prerefine,
          rho_prerefine = out$rho_prerefine,
          vbetarho = NULL,
          sebeta = NULL,
          sephi = NULL,
          serho = NULL,
          burnin = out$burnin,
          avgiter = out$avgiter,
          n_iter_total = out$n_iter_total,
          final_refine = out$final_refine,
          completed = out$completed,
          param_names = make_param_names()
        ))
      }
    }
  }

  stop("No valid route found.")
}
