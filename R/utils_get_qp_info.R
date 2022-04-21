#' @export
get_qp_info<-function(p,
                      A,
                      G,
                      lb,
                      ub,
                      D,
                      lambda_2,
                      E,
                      lambda_1)
{
  # - n_x
  n_x = get_n_x(p)

  # - equalities
  n_eq = get_ncon(A)
  any_eq = n_eq > 0

  # - inequalities
  n_G = get_ncon(G)
  any_G = n_G > 0


  # - lb and ub
  n_lb = get_ncon(lb)
  n_ub = get_ncon(ub)
  any_lb = get_any_lb(lb)
  any_ub = get_any_ub(ub)
  n_ineq = max(n_G,n_lb,n_ub)
  any_ineq = (any_lb | any_ub | any_G)

  # - l1
  n_E = get_ncon(E)
  any_l_1 = get_any(lambda_1, threshold = 0)
  n_E = ifelse(any_l_1,n_E,0)

  # - l2
  n_D = get_ncon(D)
  any_l_2 = get_any(lambda_2, threshold = 0)



  info = list(n_x = n_x,
              n_eq = n_eq,
              any_eq = any_eq,
              n_G = n_G,
              any_G = any_G,
              n_lb = n_lb,
              n_ub = n_ub,
              any_lb = any_lb,
              any_ub = any_ub,
              n_ineq = n_ineq,
              any_ineq = any_ineq,
              n_E = n_E,
              any_l_1 = any_l_1,
              n_D = n_D,
              any_l_2 = any_l_2)

  return(info)
}
