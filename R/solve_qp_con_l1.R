#' @export
torch_solve_qp_con_l1_approx<-function(Q,
                                  p,
                                  A = NULL,
                                  b = NULL,
                                  G = NULL,
                                  h = NULL,
                                  lb = NULL,
                                  ub = NULL,
                                  E,
                                  lambda_1,
                                  control)
{
  # --- with no grad determine z_star
  with_no_grad({
    # --- default to admm ==todo make this accessible to outside calls:
    control$solver='admm'

    model = nn_qp(Q = Q,
                  p = p,
                  A = A,
                  b = b,
                  G = G,
                  h = h,
                  lb = lb,
                  ub = ub,
                  control = control)

    z_star = model()
  })

  # --- convert the L1 to the approximate L-2 regularization:
  if(!is.null(E)){
    E_t = torch_transpose(E,2,3)
    mat = torch_matmul(E_t,E)
    Ez = torch_matmul(E,z_star)#torch_matmul(z,E$t())
    Ez[Ez>=0] = torch_clamp(Ez[Ez>=0],min=10^-8)
    Ez[Ez<0] = torch_clamp(Ez[Ez<0],max = -10^-8)

    reg = torch_diag_embed(1/abs(Ez$squeeze(3)))
    D = torch_matmul(mat,reg)

  }
  else{
    z_star[z_star>=0] = torch_clamp(z_star[z_star>=0],min = 10^-8)
    z_star[z_star<0] = torch_clamp(z_star[z_star<0],max = -10^-8)

    D = torch_diag_embed(1/abs(z_star$squeeze(3)))
  }

  # --- solve a L2 constrained QP and perform backwardiff:
  model$info$any_l_2 = TRUE
  z_star = torch_solve_qp(Q = Q,
                          p = p,
                          A = A,
                          b = b,
                          G = G,
                          h = h,
                          lb = lb,
                          ub = ub,
                          D = D,
                          E = NULL,
                          lambda_1 = NULL,
                          lambda_2 = 0.5*lambda_1,
                          control = control,
                          info = model$info,
                          solver_method = model$solver_method,
                          sol_index_list = model$sol_index_list)

  return(z_star)


}
