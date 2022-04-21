#' @export
nn_qp_control<-function(solver = c('admm','osqp','int','quadprog'),
                        max_iters = 1000,
                        tol = 10^-4,
                        tol_primal = 10^-4,
                        tol_dual = 10^-4,
                        tol_relative = 10^-8,
                        tol_method = 'mean',
                        rho = 0.01,#somewhere between 0.001 and 0.01 tends to work well
                        rho_eq_scale = 10^3,
                        sigma = 10^-6,
                        alpha = 1,
                        verbose = FALSE,
                        digits = 8,
                        do_D_crossprod = TRUE,
                        is_E_diag = TRUE,
                        is_G_diag =TRUE,
                        lb_default = -10^10,
                        ub_default = 10^10,
                        output_as_list = FALSE,
                        unroll_grad = FALSE,
                        warm_start = TRUE,
                        int_reg = 10^-6,
                        backprop = c('fixed_point','kkt'),
                        hybrid = FALSE,
                        hybrid_iter = 90,
                        hybrid_solver = 'int',
                        hybrid_max_iters = 10,
                        ...)
{
  # rho
  # flags for box con
  lst = list(...)
  solver = tolower(solver[1])
  if(solver == 'quadprog'){
    #output_as_list = TRUE
    unroll_grad = FALSE
  }
  control = list(solver = solver[1],
                 max_iters = max_iters,
                 tol = tol,
                 tol_primal = tol_primal,
                 tol_dual = tol_dual,
                 tol_relative = tol_relative,
                 tol_method = tol_method,
                 rho = rho,
                 rho_eq_scale = rho_eq_scale,
                 sigma = sigma,
                 alpha = alpha,
                 verbose = verbose,
                 digits = digits,
                 do_D_crossprod = do_D_crossprod,
                 is_E_diag = is_E_diag,
                 is_G_diag = is_G_diag,
                 lb_default = lb_default,
                 ub_default = ub_default,
                 output_as_list = output_as_list,
                 unroll_grad = unroll_grad,
                 warm_start = warm_start,
                 int_reg = int_reg,
                 backprop = backprop[1],
                 hybrid = hybrid,
                 hybrid_iter = hybrid_iter,
                 hybrid_solver = hybrid_solver,
                 hybrid_max_iters = hybrid_max_iters
                 )
  control = c(control,lst)
  return(control)
}
