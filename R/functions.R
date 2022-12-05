library(Rcpp)
library(RcppArmadillo)

# formula = Y ~ X1
# formula = outcome ~ tx
# dat = mydat
# id = c("Level3", "Level2")
# id = c("cluster", "period")
# link = "logit"
# family = "gaussian"
# corstr = "exchangeable"
# design = "hierarchical"
# design = "step-wedge"
# optim = "stochastic"
# tol = 1e-6
# batch_size = NULL

hgee = function(formula,
                id,
                dat,
                family = "gaussian",
                corstr = "independence",
                design = "hierarchical",
                se_adjust = "unadjusted",
                optim = "deterministic",
                batch_size = c(30, 5, 2),
                burnin = 25,
                avgiter = 25,
                tol = 1e-6) {

  #A bunch of checks to ensure inputs are valid
  if (!(family %in% c("binomial","gaussian"))) {stop("family not recognized")}
  if (!(corstr %in% c("independence", "exchangeable"))) {stop("corstr not recognized")}
  if (!(design %in% c("hierarchical", "stepped-wedge"))) {stop("design not recognized")}
  if (!(optim %in% c("deterministic", "stochastic"))) {stop("optim not recognized")}

  outcome = unlist(dat[all.vars(formula)[1]])              #Outcome
  clusterid = as.matrix(dat[id])                           #Cluster ids (levels for hierarchical, (cluster, period) for step-wedge)
  phi = 1                                                  #Dispersion parameter (variance of errors in continuous model)

  #Following if statement is initializing
  #   Design (hierarchical vs step-wedge)
  #   Cluster levels, if hierarchical
  if (design == "hierarchical") {

    design_mat = model.matrix(formula, dat)      #Design matrix
    beta = cbind(numeric(ncol(design_mat)))        #Linear coefficients
    levels_num = length(id)                        #Number of levels
    rho = numeric(levels_num)                      #Association parameters

  } else if (design == "stepped-wedge") {  #(cluster, period) column order for clusterid

    formula_fixed = as.formula(paste(all.vars(formula)[1], paste(-1, all.vars(formula)[2], paste("factor(", id[2], ")"), sep = " + "), sep = " ~ "))
    design_mat = model.matrix(formula_fixed, dat)
    beta = numeric(ncol(design_mat))
    levels_num = 3
    rho = numeric(levels_num)

  }


  ###########################
  #Newton-Raphson iterations#
  ###########################
  if (optim == "deterministic") {
    solver_output = NewRaph(beta,
                            phi,
                            rho,
                            outcome,
                            design_mat,
                            clusterid,
                            family,
                            #link,
                            corstr,
                            design,
                            se_adjust,
                            tol)
  } else if (optim == "stochastic") {
    solver_output = StochNewRaph(beta,
                                 phi,
                                 rho,
                                 outcome,
                                 design_mat,
                                 clusterid,
                                 family,
                                 #link,
                                 corstr,
                                 design,
                                 se_adjust,
                                 batch_size[1:(levels_num+1)],
                                 burnin,
                                 avgiter)
  }

  #################
  #Standard errors#
  #################
  # Info = solve(rbind(cbind(solve(solver_output$Info1), 0, numeric(length(rho))),
  #              cbind(solver_output$B, solve(solver_output$Info1_5), numeric(length(rho))),
  #              cbind(solver_output$D, solver_output$E, solve(solver_output$Info2))))
  #Info = bdiag(solver_output$Info1, solver_output$Info1_5, solver_output$Info2)
  # G_outer_sum = matrix(apply(apply(abind(solver_output$G1, solver_output$G1_5, solver_output$G2, along = 1),
  #                      3, tcrossprod, SIMPLIFY = F), 1, sum),
  #                      nrow = length(beta)+length(rho)+1, ncol = length(beta)+length(rho)+1)
  var_sandwich = solver_output$var_sandwich
  se_sandwich = sqrt(diag(var_sandwich))

  if (family == "binomial") {
    return(list(beta = solver_output$beta,
                rho = solver_output$rho,
                Info1 = solver_output$Info1,
                Info2 = solver_output$Info2,
                vbetarho = var_sandwich,
                sebeta = se_sandwich[1:length(beta)],
                serho = se_sandwich[-(1:(length(beta)))],
                iter = solver_output$iter
    ))
  } else if (family == "gaussian") {
    return(list(beta = solver_output$beta,
                phi = solver_output$phi,
                rho = solver_output$rho,
                Info1 = solver_output$Info1,
                Info1_5 = solver_output$Info1_5,
                Info2 = solver_output$Info2,
                vbetarho = var_sandwich,
                sebeta = se_sandwich[1:length(beta)],
                sephi = se_sandwich[length(beta)+1],
                serho = se_sandwich[-(1:(length(beta)+1))],
                iter = solver_output$iter
    ))
  }
}

##########
#ARCHIVED#
##########
##Newton-Raphson solver (in R)
# mysolver = function(beta, rho, outcome, design_mat, id, K) {
#   mu = expit(design_mat%*%beta)
#   rho0 = 1 - sum(rho)
#   resid1 = outcome - mu
#   resid1_list = split(resid1, id[,1])
#   U_sqrt = sqrt(mu*(1-mu))
#   U_sqrt_list = split(U_sqrt, id[,1])
#   std_resid1 = resid1/U_sqrt
#   std_resid2 = tapply(std_resid1, id[,1], function(r) combn(r, m = 2, FUN = prod))
#   I = length(resid1_list)
#   p = length(beta)
#
#   HG1 = array(0, c(p, p+1, I))
#   HG2 = array(0, c(2,3, I))
#
#   for (i in seq(length(resid1_list))) {
#     v = rep(1/rho0 - rho[1]/{rho0*(rho0/K[[i]] + rho[1])}, K[[i]])
#     c = 1/{1/rho[2] + sum(v)}; vv = outer(v, v); A_inv = bdiag(sapply(K[[i]], function(k) diag(1/rho0,k) - matrix(rho[1]/{rho0*(rho0 + rho[1]*k)}, nrow = k, ncol = k)))
#     C1_inv = A_inv - c*vv
#
#     common_factor = t(design_mat[id[,1] == i,])%*%as.matrix(U_sqrt_list[[i]]*{C1_inv*(1/U_sqrt_list[[i]])})
#     G1 = common_factor%*%resid1_list[[i]]
#     H1 = common_factor%*%(U_sqrt_list[[i]]*design_mat[id[,1] == i,])
#     HG1[,,i] = cbind(H1, G1)
#
#     idx = seq(length(std_resid2[[i]]))
#     outer_idx = index_fun(K[[i]]); inner_idx = setdiff(idx, outer_idx); inner_idx_length = length(inner_idx)
#     a = sum(std_resid2[[i]][inner_idx] - rho[1] - rho[2]); b = sum(std_resid2[[i]][outer_idx] - rho[2])
#     G2 = rbind(a, a+b)
#     H2 = matrix(c(inner_idx_length,inner_idx_length,inner_idx_length,length(std_resid2[[i]])), nrow = 2, ncol = 2)
#     HG2[,,i] = cbind(H2, G2)
#   }
#   return(list(HG1, HG2))
# }

##########
#ARCHIVED#
##########
#Compute indices of off-block diagonal blocks corresponding to same first-level, but different second-level
#Same first & second level correlation is then the complement of this index vector
# index_fun = function(K) {
#   m = length(K)
#   K_rev = rev(K)
#   K_seq = unlist(sapply(K[-m], function(s) (s-1):0))
#   K_rev_cumsum = rep(rev(cumsum(K_rev[-m])), K[-m])
#
#   idx_range = matrix(cumsum(c(rbind(K_seq, K_rev_cumsum))), nrow = 2)
#   return(unlist(apply(idx_range, 2, function(x) seq(x[1]+1, x[2]))))
# }
