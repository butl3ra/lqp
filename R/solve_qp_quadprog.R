#' @export
torch_solve_qp_quadprog<-function(Q,
                                  p,
                                  A,
                                  b,
                                  G,
                                  h,
                                  control,
                                  ...)
{
  # --- control:
  output_as_list = control$output_as_list

  # note: n_batch prep should be done outside this function
  # --- prep:
  x_size = get_size(p)
  n_batch = x_size[1]
  n_x = x_size[2]

  n_eq = get_ncon(A)
  any_eq = n_eq > 0
  n_ineq = get_ncon(G)
  any_ineq = n_ineq > 0

  # --- if any eq:
  if(any_eq){
    AT = torch_transpose(A,2,3)
    Amat = torch_cat(list(AT,-torch_transpose(G,2,3)),3)
    bvec = torch_cat(list(b,-h),2)
    meq = n_eq
  }
  else{
    Amat = -torch_transpose(G,2,3)
    bvec = -h
    meq = 0
    AT = NULL
  }

  # --- convert to arrays:
  Q_a = as_array(Q)
  p_a = as_array(p)
  Amat_a = as_array(Amat)
  bvec_a = as_array(bvec)

  x = array(0,c(n_batch,n_x,1))
  lams = array(0,c(n_batch,n_ineq,1))
  slacks = array(0,c(n_batch,n_ineq,1))
  nus = NULL
  if(any_eq){
    nus = array(0,c(n_batch,n_eq,1))
  }
  # --- main loop: sequential
  for(i in 1:n_batch){
    sol = torch_solve_qp_quadprog_core(Q = Q_a[i,,],
                                      p = p_a[i,,],
                                      Amat = Amat_a[i,,],
                                      bvec = bvec_a[i,,],
                                      meq = meq,
                                      factorized = FALSE)
    x[i,,] = sol$x
    lams[i,,] = sol$lams
    slacks[i,,] = sol$slacks
    if(any_eq){
      nus[i,,] = sol$nus
    }

  }
  # --- convert to tensor:
  x = as_torch_tensor(x)
  lams = as_torch_tensor(lams)
  slacks = as_torch_tensor(slacks)
  if(any_eq){
    nus = as_torch_tensor(nus)
  }

  # --- make output list:
  out = list(x = x,
             lams = lams,
             slacks = slacks)
  if(any_eq){
    out$nus = nus
  }
  if(!output_as_list){
    out = torch_cat(out,dim = 2)
  }

  return(out)
}

#' @export
torch_solve_qp_quadprog_core<-function(Q,
                                       p,
                                       Amat,
                                       bvec,
                                       meq,
                                       factorized = FALSE)
{
  #Note solve.QP solves:
  #      -z^Tp + 0.5*z^TQz
  #       A^Tz >= b
  # we pass -y_hat (i.e. -p values) so we need to undo the negative value here
  sol = quadprog::solve.QP(Dmat = Q,
                           dvec  = -p,
                           Amat = Amat,
                           bvec = bvec,
                           meq = meq,
                           factorized = factorized)

  x = round(sol$solution,8)
  nus = NULL
  idx = 1:ncol(Amat)
  # --- nus:
  if(meq>0){
    idx_meq = 1:meq
    idx = idx[-idx_meq]
    nus = sol$Lagrangian[idx_meq]
  }
  lams = sol$Lagrangian[idx]

  slacks = t(Amat[,idx,drop=F])%*%x - bvec[idx]

  out = list(x = x,
             lams = lams,
             nus = nus,
             slacks = slacks[,1])

  return(out)


}

#' @export
torch_qp_make_sol_mat<-function(Q,
                                G,
                                A,
                                lams,
                                nus,
                                slacks,
                                E = NULL,
                                lambda_1 = NULL,
                                x = NULL,
                                tau = 10^-6)
{

  # --- prep:
  x_size = get_size(Q)
  n_batch = x_size[1]
  n_x = x_size[2]

  n_eq = get_ncon(A)
  any_eq = n_eq > 0
  n_ineq = get_ncon(G)
  any_ineq = n_ineq > 0
  n_E = get_ncon(E)
  any_l_1 = n_E > 0

  if(any_eq){
    AT = torch_transpose(A,2,3)
  }
  if(any_ineq){
    GT = torch_transpose(G,2,3)
  }
  # --- Modify Q to account for quadratic approximation
  if( any_l_1 ){
    EE = torch_crossprod(E)
    Ex = torch_matmul(E,x)
    Ex_abs = torch_abs(Ex) + tau
    Ex_abs_inv = 1/Ex_abs
    denom = torch_diag_embed(Ex_abs_inv$squeeze(3))
    mat = torch_matmul(denom,EE)
    Q = Q + lambda_1 * mat
  }


  # --- if !any eq
  if(!any_eq){
    lhs_1 = torch_cat(list(Q, GT*torch_transpose(lams,2,3)),3)
    lhs_2 = torch_cat(list(G, torch_diag_embed(-slacks[,,1]) ),3)
    lhs = torch_cat(list(lhs_1,lhs_2),2)
  }
  else if(!any_ineq){
    lhs_1 = torch_cat(list(Q, AT),3)
    lhs_2 = torch_cat(list(A,torch_zeros(c(n_batch,n_eq,n_eq)) ),3)
    lhs = torch_cat(list(lhs_1,lhs_2),2)
  }
  else{
    lhs_1 = torch_cat(list(Q, GT*torch_transpose(lams,2,3), AT),3)
    lhs_2 = torch_cat(list(G, torch_diag_embed(-slacks[,,1]), torch_zeros( c(n_batch,n_ineq,n_eq )  ) ),3)
    lhs_3 = torch_cat(list(A,torch_zeros( c(n_batch,n_eq,n_ineq )  ), torch_zeros(c(n_batch,n_eq,n_eq)) ),3)
    lhs = torch_cat(list(lhs_1,lhs_2,lhs_3),2)
  }


  return(lhs)

}

#' @export
torch_solve_qp_backwards<-function(dl_dx,
                                   sol_mats,
                                   n_eq,
                                   n_ineq)
{

  # --- prep:
  dl_dx = dl_dx#$unsqueeze(3)
  dim_dl_dx = get_size(dl_dx)
  n_batch = dim_dl_dx[1]
  n_x = dim_dl_dx[2]
  n_con = n_eq + n_ineq
  zeros = torch_zeros(c(n_batch,n_con,1))

  # --- rhs:
  rhs = torch_cat(list(-dl_dx,zeros),dim=2)
  back_sol = linalg_solve(sol_mats,rhs)

  d_x = back_sol[,1:n_x,,drop=F]
  d_lam = NULL
  if(n_ineq > 0){
    idx_lam = (n_x+1):(n_x+n_ineq)
    d_lam = back_sol[,idx_lam,,drop=F]
  }
  d_nu = NULL
  if(n_eq >0){
    idx_nu = (n_x+n_ineq+1):(n_x+n_ineq+n_eq)
    d_nu = back_sol[,idx_nu,,drop=F]
  }

  diff_list = list(d_x = d_x,
                   d_lam = d_lam,
                   d_nu = d_nu)

  return(diff_list)

}
