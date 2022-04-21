#' @export
torch_solve_qp_eqcon<-function(Q,
                               p,
                               A,
                               b,
                               mat_inv = NULL)
{

  rhs = torch_cat(list(-p, b), 2)

  if(is.null(mat_inv)){
    lhs = torch_qp_eqcon_mat(Q = Q, A = A)
    sol = linalg_solve(lhs,rhs)
  }
  else{
    sol = torch_matmul(mat_inv,rhs)
  }

  return(sol)
}

#' @export
torch_qp_eqcon_mat<-function(Q,
                             A,
                             bottom_right = NULL)
{
  A_size = get_size(A)
  # --- default zero
  if(is.null(bottom_right)){
    bottom_right = torch_zeros(A_size[c(1,2,2)])
  }
  AT = torch_transpose(A,2,3)
  lhs_u = torch_cat(list(Q,AT),3)
  lhs_l = torch_cat(list(A, bottom_right ) ,3)
  lhs = torch_cat(list(lhs_u,lhs_l), 2)
  return(lhs)
}
