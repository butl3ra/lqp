#' @export
is_static_mat<-function(is_param,
                        any_l_1,
                        any_l_2,
                        is_G_diag,
                        is_E_diag)
{
  is_Q_static = !is_param['Q']
  is_A_static = !is_param['A']
  is_G_static = !is_param['G'] | is_G_diag
  is_D_static = !is_param['D'] | !any_l_2
  is_E_static = !is_param['E'] | !any_l_1 | is_E_diag
  is_mat_static = is_Q_static & is_A_static & is_G_static & is_D_static & is_E_static
  return(is_mat_static)
}

#' @export
get_osqp_mat<-function(Q,
                       A,
                       G,
                       D,
                       E,
                       lambda_1,
                       lambda_2,
                       is_param,
                       is_G_diag,
                       is_E_diag,
                       control)
{
  null_size = c(0,0,0)
  any_A = get_any(A)
  A_size = get_size(A,null_size)
  any_G = get_any(G)
  G_size = get_size(G,null_size)
  any_l_1 = get_any(lambda_1)
  any_l_2 = get_any(lambda_2)
  if(!any_l_1){
    is_E_diag = TRUE
  }
  # --- simples case:


  # --- there is redundancy here but it is easier to view the cases:
  # --- add l_2 regularization
  if(any_l_2){
    Q = Q + lambda_2 * D
  }

  # ---  case 1: unconstrained (uncon solver)
  if(!any_A & !any_G & !any_l_1){
    mat = torch_solve(Q)
  }
  # --- case 2: eqcon only (eqcon solver)
  if(any_A & !any_G & !any_l_1){
    d_zero = c(d_A[c(1,2,2)])
    zero = torch_zeros(d_zero)
    lhs_u = torch_cat( list(Q,A$view(d_A[c(1,3,2)] )), 3)
    lhs_l = torch_cat(list(A, zero ) ,3)
    lhs = torch_cat(list(lhs_u,lhs_l), 2)
  }

  # --- case 3: uncon with G_diag and E diag (ADMM solver)

  # --- case 3: eqcon with G_diag and E diag (ADMM solver)

  # --- case 4: eqcon with l_1 and E diag (OSQP solver)

  # --- case 4:




}

