# formula = Y ~ X1
# formula = outcome ~ tx
# dat = mydat
# id = c("Level3", "Level2")
# id = c("cluster", "period")
# link = "logit"
# family = "gaussian"
# corstr = "independence"
# corstr = "nested-exchangeable"
# corstr = "block-exchangeable"
# optim = "stochastic"
# tol = 1e-6
# batch_size = NULL

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
                tol = 1e-6) {
  
  #A bunch of checks to ensure inputs are valid
  if (!(family %in% c("binomial","gaussian"))) {stop("family not recognized")}
  if (!(corstr %in% c("independence", "nested-exchangeable", "block-exchangeable"))) {stop("corstr not recognized")}
  if (!(optim %in% c("deterministic", "stochastic"))) {stop("optim not recognized")}
  
  mf <- model.frame(formula = as.formula(formula), data = dat)
  outcome = model.response(mf)                             #Outcome
  clusterid = as.matrix(dat[id])                           #Cluster ids [levels for nested-exchangeable, (cluster, period) for block-exchangeable]

  if (corstr == "nested-exchangeable" | corstr == "independence") {
    levels_num = length(id)                        
  } else if (corstr == "block-exchangeable") {  #(cluster, period) column order for clusterid
    levels_num = 3
  }
  
  design_mat = model.matrix(attr(mf, "terms"), data = mf)        #Design matrix
  beta = cbind(numeric(ncol(design_mat)))                        #Linear coefficients
  phi = 1                                                        #Dispersion parameter
  rho = numeric(levels_num)                                      #Exchangeable correlation coefficients
  
  
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