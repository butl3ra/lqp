#' @export
torch_solve_qp_admm<-function(Q,
                              p,
                              A,
                              b,
                              G,
                              lb,
                              ub,
                              E,
                              lambda_1,
                              control,
                              x = NULL,
                              u = NULL,
                              z = NULL,
                              mat_data = NULL,
                              mat_pivots = NULL,
                              mat_inv = NULL,
                              ...)
{
  # --- unpacking control:
  rho = control$rho
  tol_primal = 0.5*control$tol_primal
  tol_dual = 0.5*control$tol_dual
  tol_relative = control$tol_relative
  verbose = control$verbose
  max_iters = control$max_iters
  tol_method  = control$tol_method
  output_as_list = control$output_as_list
  warm_start = control$warm_start
  unroll_grad = control$unroll_grad
  lb_default = control$lb_default
  ub_default = control$ub_default

  # --- check for warm start
  warm_start_vars = check_warm_start(x = x,
                                     u = u,
                                     z = z)
  warm_start_vars = warm_start_vars & warm_start

  if(unroll_grad){
    warm_start_mats = !is.null(mat_inv)
  }
  else{
    warm_start_mats = check_warm_start(mat_data = mat_data,
                                       mat_pivots = mat_pivots)
  }
  warm_start_mats = warm_start_mats & warm_start



  # --- prep:
  lb = torch_clamp(lb,min = lb_default)
  ub = torch_clamp(ub, max = ub_default)
  x_size = get_size(p)
  n_x = x_size[2]
  idx_x = 1:x_size[2]

  #idx_x = torch_tensor(idx_x,dtype=torch_int32())
  any_A = get_any(A)
  n_eq = get_ncon(A)
  idx_eq = (x_size[2]+1):(x_size[2] + n_eq)
  any_G = get_any(G)
  any_lb = as.logical(torch_max(lb) > -Inf)
  any_ub = as.logical(torch_min(ub) < Inf)
  any_l_1 = get_any(lambda_1, threshold = 0)
  any_ineq = any_lb | any_ub

  # --- if any G then just adjust the bounds:
  if(any_G & FALSE){
    diag_G = torch_diagonal(G,dim1=2,dim2=3)
    diag_G = prep_torch_tensor(diag_G,target_dim=3)
    lb = lb / diag_G
    ub = ub / diag_G
  }

  # --- create G and h for primal/dual residual computation
  if(!any_G){
    G = torch_make_G_bound(n_x = n_x,
                           any_lb = any_lb,
                           any_ub = any_ub)
  }
  h = torch_cat(list(-lb,ub),dim=2)

  # --- if any l_1 then convert to a vector
  if(any_l_1){
    E = torch_diagonal(E,dim1=2,dim2=3)
    E= prep_torch_tensor(E,target_dim=3)
    lambda_1_E = lambda_1 * E
    lambda_1_E_rho = lambda_1_E/rho
  }

  # --- factorize and cache inverse if not supplied
  if(!warm_start_mats){
    Id = torch_eye(n_x)$unsqueeze(1)
    Q_I = Q + rho*Id
    if(any_A){
      mat = torch_qp_eqcon_mat(Q = Q_I, A = A)
    }
    else{
      mat = Q_I
    }
    #lu  factorization is not differentiable:
    if(unroll_grad){
      if(n_x <= 100){# --- this is efficient for small problems
        mat_inv = linalg_inv(mat)
      }
      else if(n_x > 100){
        with_no_grad({mat_lu = torch_lu(mat)})
        mat_inv = mat # --- placeholder
      }
      else{# --- placeholder until cleanup possible
        mat_inv = mat
      }
    }
    else{
      with_no_grad({mat_lu = torch_lu(mat)})
      mat_data = mat_lu[[1]]
      mat_pivots = mat_lu[[2]]
    }

  }
  else{
    if(!unroll_grad){
      mat_pivots = change_tensor_dtype(mat_pivots,dtype = torch_int())
    }
  }


  # --- initialize z and u
  if(!warm_start_vars){
    z = torch_zeros(x_size)
    u = torch_zeros(x_size)
  }
  # --- main loop
  errors = NULL
  iters = 1:max_iters
  for(iter in iters){

    # --- projection to sub-space:
    y = z - u
    # --- rhs matrix
    if(any_A){
      rhs = torch_cat(list(-p + rho*y, b), 2)
    }
    else{
      rhs = -p + rho*y
    }
    if(unroll_grad){
      if(n_x <= 100){
        xv = torch_matmul(mat_inv,rhs)
      }
      else if(n_x > 100){
        xv = lu_layer(A = mat, b = rhs, A_lu = mat_lu) # --- this is having some memory leakage
      }
      else{
        xv = linalg_solve(mat,rhs)
      }
    }
    else{
      xv = torch_lu_solve(rhs,mat_data,mat_pivots)
    }
    x = xv[,idx_x,]

    # --- proximal projection:
    z_prev = z
    z = x + u
    if(any_l_1){
      z = torch_soft_threshold(z, lambda_1_E_rho)
    }
    if(any_ineq){
      z = torch_proj_box(z,
                         lb = lb,
                         ub = ub,
                         any_lb = any_lb,
                         any_ub = any_ub)
    }
    # --- update residuals
    r = x - z
    s = rho * (z - z_prev)
    # --- running sum of residuals and dual variables
    u = u + r

    # --- compute lambdas and nus:
    primal_error = torch_norm(r,dim=2,keepdim=T)#/n_x
    dual_error = torch_norm(s,dim=2,keepdim=T)#/n_x

    primal_error = torch_reduce(primal_error,method = tol_method)
    dual_error = torch_reduce(dual_error,method = tol_method)
    # --- history of errors:
    error = (primal_error+dual_error)$item()
    errors = c(errors,error )

    # --- verbose
    if(verbose){
      cat('iteration: ', iter, '\n')
      cat('|| primal_error||_2 = ', primal_error$item(),'\n')
      cat('|| dual_error||_2 = ', dual_error$item(),'\n')
    }

    stop_1 = as.logical(primal_error < tol_primal) & as.logical(dual_error < tol_dual)
    stop_2 = FALSE
    # --- check for improvements
    if(iter > 10){
      prev_error = errors[iter - 9]
      stop_2 = as.logical(abs(prev_error - error) < tol_relative )# --- potential roundoff errors for float32
    }
    do_stop = stop_1 | stop_2
    if(do_stop){
      break
    }


  }

  # --- solution is [x, z, z_prev, u]
  # --- residuals can be computed using x-z, z-z_prev
  lams = u*rho
  lams_neg = torch_threshold_(-lams,0,0)
  lams_pos = torch_threshold_(lams,0,0)
  lams = torch_cat(list(lams_neg,lams_pos),2)

  # this first statement will almost always be true because ub and lb
  # are defaulted to not be numerically +/- Inf
  if(any_lb & any_ub){
    lams = torch_cat(list(lams_neg,lams_pos),2)
  }
  else if(any_lb){
    lams = lams_neg
  }
  else if(any_ub){
    lams = lams_pos
  }

  nus = NULL
  if(any_A){
    n_eq = get_ncon(A)
    idx_eq = (x_size[2]+1):(x_size[2] + n_eq)
    nus = xv[,idx_eq,,drop=F]
  }

  # --- concatenate if not output as list...
  if(!output_as_list){
    if(unroll_grad){
      mat_inv = torch_reshape_mat(mat_inv,forward = TRUE)
    }
    else{
      mat_data = torch_reshape_mat(mat_data,forward = TRUE)
      mat_pivots = mat_pivots$unsqueeze(3)
    }
  }
  # --- make output list:
  out = list(x = z,# z == x at optimality and this allows unroll WRT G, E, lb, ub
             z = z,
             u = u,
             lams = lams)
  if(any_A){
    out$nus = nus
  }
  if(unroll_grad){
    out$mat_inv = mat_inv
  }
  else{
    out$mat_data = mat_data
    out$mat_pivots = mat_pivots
  }
  if(!output_as_list){
    out = torch_cat(out,dim = 2)
  }

  return(out)


}




#' @export
torch_bounds_to_h<-function(lb,
                            ub,
                            any_lb,
                            any_ub)
{
  if(missing(any_lb)){
    any_lb = as.logical(torch_max(lb) > -Inf)
  }
  if(missing(any_ub)){
    any_ub = as.logical(torch_min(ub) < Inf)
  }
  if(any_lb & any_ub){
    h = torch_cat(list(-lb,ub),dim=2)
  }
  else if(any_lb){
    h = -lb
  }
  else if(any_ub){
    h = ub
  }
  return(h)
}

#' @export
torch_qp_grad_admm<-function(x,
                             dl_dz,
                             u,
                             lams,
                             nus,
                             mat_inv,
                             mat_data,
                             mat_pivots,
                             A,
                             lb,
                             ub,
                             E,
                             lambda_1,
                             rho,
                             Q)
{
  # --- NOTE: this does not yet support grads for E or lambda_1

  # --- prep:
  dim_x = dim(x)
  n_x = dim_x[2]
  n_batch = dim(dl_dz)[1]
  n_eq = get_ncon(A)
  any_eq = n_eq > 0
  any_l_1 = get_any(lambda_1, threshold = 0)
  idx_x = 1:n_x
  idx_eq = (n_x+1):(n_x+n_eq)

  xt = torch_transpose(x,2,3)
  x_u = x + u
  x_u_abs = torch_abs(x_u)

  # create tau
  lambda_1_E_rho = 0
  if(any_l_1){
    E_diag = torch_diagonal(E,dim1=2,dim2=3)
    E_diag = prep_torch_tensor(E_diag,target_dim=3)
    lambda_1_E = lambda_1 * E_diag
    lambda_1_E_rho = lambda_1_E/rho#tau
  }
  s_x_u = torch_soft_threshold(x_u,lambda_1_E_rho )

  # --- derivative of the projection operator:
  dpi_dx = torch_ones(dim_x)# should this be s(x_u) > ub
  dpi_dx[s_x_u > ub] = 0
  dpi_dx[s_x_u < lb] = 0

  # --- derivative of the soft-thresholding:
  ds_dx = torch_ones(dim_x)
  ds_dx[x_u_abs < lambda_1_E_rho] = 0

  # --- dl_dx: cahin rule
  dP_dx = dpi_dx*ds_dx
  dl_dx = dl_dz*dP_dx

  # --- rhs:
  if(any_eq){
    zeros = torch_zeros(c(n_batch,n_eq,1))
    rhs = torch_cat(list(-dl_dx,zeros),dim=2)
  }
  else{
    rhs = -dl_dx
  }
  # --- fast implementation
  Id_x = torch_eye(n_x)$unsqueeze(1)
  Q_I = Q + rho*Id_x
  dP_dv = torch_diag_embed(dP_dx$squeeze(3))#this needs to consider ds_dx as well
  tl = -rho*(2*dP_dv - torch_eye(n_x))
  zero = torch_zeros(n_batch,n_eq,n_x)
  if(any_eq){
    Id_eq = prep_batch_size(torch_eye(n_eq)$unsqueeze(1),n_batch)
    mat = torch_qp_eqcon_mat(Q = Q_I, A = A)
    D = torch_qp_eqcon_mat(tl,zero)
    R_mat = torch_qp_eqcon_mat(dP_dv,zero,bottom_right = Id_eq)
  }
  else{
    mat = Q_I
    D = tl
    R_mat = dP_dv
  }
  R = torch_diagonal(R_mat,dim1=2,dim2=3)
  lhs = R$unsqueeze(3)*mat + D#torch_matmul(R,mat)
  d_vec_2 = try(linalg_solve(lhs,rhs),silent=T)
  if(inherits(d_vec_2,'try-error')){
    d_vec_2 = linalg_solve(lhs + 10^-4*torch_eye(dim(lhs)[2])$unsqueeze(1),rhs)
  }

  # --- from here
  dv = d_vec_2[,idx_x,,drop=F]
  dvt = torch_transpose(dv,2,3)

  # --- dl_dp
  dl_dp = dv

  # --- dl_dQ
  dl_dQ = NULL
  if(check_requires_grad(Q)){
    #dl_dQ =  torch_matmul(dv,xt)#this matches unrolled
    dl_dQ = 0.5*(torch_matmul(dv,xt) + torch_matmul(x,dvt))
  }

  # --- dl_dA and dl_db
  dl_db = NULL
  dl_dA = NULL
  if(any_eq){
    dnu = d_vec_2[,idx_eq,,drop=F]
    dl_db = -dnu
    if(A$requires_grad){
      dnu_t = torch_transpose(dnu,2,3)
      dl_dA = torch_matmul(dnu,xt) + torch_matmul(nus,dvt)
    }
  }

  # -- simple equation from kkt ...
  if(lb$requires_grad | ub$requires_grad){

    kkt = -dl_dz - torch_matmul(Q,dv)
    if(any_eq){
      kkt = kkt - torch_matmul(torch_transpose(A,2,3),dnu)
    }
    div = rho*u
    div[div==0]=1

    dlam = kkt/div

  }
  # --- dl_dlb
  dl_dlb = NULL
  if(lb$requires_grad){
    dl_dlb = dlam*lams[,idx_x,]
  }
  # --- dl_dub
  dl_dub = NULL
  if(ub$requires_grad){
    dl_dub = -dlam*lams[,n_x+idx_x,]
  }

  # --- dl_dE and dl_dlam --->NOT YET OPERATIONAL
  dl_dE = NULL
  dl_dlam = NULL
  if(any_l_1){
    d1E = torch_ones(dim_x)
    d1E[x_u_abs < lambda_1_E_rho] = 0
    ds_dtau = d1E*torch_sign(x_u)
    dl_dE = (-lambda_1/rho)*dpi_dx*ds_dtau*dl_dz
    #matching dimension --- may need to apply this to all variables...
    dim_E = dim(E)
    if(dim_E[1] == 1){
      dl_dE = torch_sum(dl_dE,dim=1,keepdim=TRUE)
    }
    dl_dE = torch_diag_embed(dl_dE$squeeze(3))

    dl_dlam = (-E_diag/rho)*dpi_dx*ds_dtau*dl_dz
    dl_dlam = torch_sum(dl_dlam)$unsqueeze(1)
    dl_dlam = prep_torch_tensor(dl_dlam)

  }

  # --- out list of grads
  grads = list(Q = dl_dQ,
               p = dl_dp,
               A = dl_dA,
               b = dl_db,
               lb = dl_dlb,
               ub = dl_dub,
               E = dl_dE,
               lambda_1 = dl_dlam
  )

  return(grads)


}

if(F){
  #much slower
  if(F){
    d_vec = torch_lu_solve(rhs,mat_data,mat_pivots)

    # --- derivative of the fixed point iterate
    #v = x_u
    #vt = torch_transpose(v,2,3)
    dP_dv = torch_diag_embed(dP_dx$squeeze(3))#this needs to consider ds_dx as well
    tl = -rho*(2*dP_dv - torch_eye(n_x))
    zero = torch_zeros(n_batch,n_eq,n_x)
    rhs2 = torch_qp_eqcon_mat(tl,zero)

    Id = prep_batch_size(torch_eye(n_eq)$unsqueeze(1),n_batch)
    rhs3 = torch_qp_eqcon_mat(dP_dv,zero,bottom_right = Id)

    #I - df/dv = I -(-mat_inv%*%rhs2 +I - rhs3) = mat_inv%*%rhs2 + rhs3
    lhs2 = torch_lu_solve(rhs2,mat_data,mat_pivots) + rhs3
    with_no_grad({lhs_lu = torch_lu(torch_transpose(lhs2,2,3))})
    lhs_data = lhs_lu[[1]]
    lhs_pivots = lhs_lu[[2]]
    #[dv,dnu]
    dv_dnu_tmp = torch_lu_solve(rhs,lhs_data,lhs_pivots)
    d_vec_2 = torch_lu_solve(dv_dnu_tmp,mat_data,mat_pivots)
  }

  # --- old
  #with_no_grad({lhs_lu = torch_lu(lhs2)})
  #lhs_data = lhs_lu[[1]]
  #lhs_pivots = lhs_lu[[2]]

  #[dv,dnu]
  #d_vec_2 = torch_lu_solve(d_vec,lhs_data,lhs_pivots)
  #d_vec_2 = torch_matmul(rhs3,d_vec_2)

  #from here
}

#' @export
torch_qp_grad_admm_dep<-function(x,
                             dl_dz,
                             u,
                             nus,
                             mat_inv,
                             mat_data,
                             mat_pivots,
                             A,
                             lb,
                             ub,
                             E,
                             lambda_1,
                             rho,
                             Q_requires_grad = TRUE)
{
  # --- NOTE: this does not yet support grads for E or lambda_1

  # --- prep:
  dim_x = dim(x)
  n_x = dim_x[2]
  n_batch = dim(dl_dz)[1]
  n_eq = get_ncon(A)
  any_eq = n_eq > 0
  any_l_1 = get_any(lambda_1, threshold = 0)
  idx_x = 1:n_x
  idx_eq = (n_x+1):(n_x+n_eq)

  xt = torch_transpose(x,2,3)
  x_u = x + u
  x_u_abs = torch_abs(x_u)

  # create tau
  lambda_1_E_rho = 0
  if(any_l_1){
    E_diag = torch_diagonal(E,dim1=2,dim2=3)
    E_diag = prep_torch_tensor(E_diag,target_dim=3)
    lambda_1_E = lambda_1 * E_diag
    lambda_1_E_rho = lambda_1_E/rho#tau
  }
  s_x_u = torch_soft_threshold(x_u,lambda_1_E_rho )

  # --- derivative of the projection operator:
  dpi_dx = torch_ones(dim_x)# should this be s(x_u) > ub
  dpi_dx[s_x_u > ub] = 0
  dpi_dx[s_x_u < lb] = 0

  # --- derivative of the soft-thresholding:
  ds_dx = torch_ones(dim_x)
  ds_dx[x_u_abs < lambda_1_E_rho] = 0

  # --- dl_dx: cahin rule
  dP_dx = dpi_dx*ds_dx
  dl_dx = dl_dz*dP_dx

  # --- rhs:
  if(any_eq){
    zeros = torch_zeros(c(n_batch,n_eq,1))
    rhs = torch_cat(list(-dl_dx,zeros),dim=2)
  }
  else{
    rhs = -dl_dx
  }
  # --- d_vec
  #d_vec = torch_matmul(mat_inv,rhs)
  d_vec = torch_lu_solve(rhs,mat_data,mat_pivots)

  # --- derivative of the fixed point iterate
  #v = x_u
  #vt = torch_transpose(v,2,3)
  dP_dv = torch_diag_embed(dP_dx$squeeze(3))#this needs to consider ds_dx as well
  tl = -rho*(2*dP_dv - torch_eye(n_x))
  zero = torch_zeros(n_batch,n_eq,n_x)
  rhs2 = torch_qp_eqcon_mat(tl,zero)

  Id = prep_batch_size(torch_eye(n_eq)$unsqueeze(1),n_batch)
  rhs3 = torch_qp_eqcon_mat(dP_dv,zero,bottom_right = Id)

  #I - df/dv = I -(-mat_inv%*%rhs2 +I - rhs3) = mat_inv%*%rhs2 + rhs3
  lhs2 = torch_lu_solve(rhs2,mat_data,mat_pivots) + rhs3
  with_no_grad({lhs_lu = torch_lu(lhs2)})
  lhs_data = lhs_lu[[1]]
  lhs_pivots = lhs_lu[[2]]

  #[dv,dnu]
  d_vec_2 = torch_lu_solve(d_vec,lhs_data,lhs_pivots)
  d_vec_2 = torch_matmul(rhs3,d_vec_2)
  dv = d_vec_2[,idx_x,,drop=F]
  dvt = torch_transpose(dv,2,3)

  # --- dl_dp
  dl_dp = dv

  # --- dl_dQ
  dl_dQ = NULL
  if(Q_requires_grad){
    #dl_dQ =  torch_matmul(dv,xt)#this matches unrolled
    dl_dQ = 0.5*(torch_matmul(dv,xt) + torch_matmul(x,dvt))
  }

  # --- dl_dA and dl_db
  dl_db = NULL
  dl_dA = NULL
  if(any_eq){
    dnu = d_vec_2[,idx_eq,,drop=F]
    dl_db = -dnu
    if(A$requires_grad){
      dnu_t = torch_transpose(dnu,2,3)
      dl_dA = torch_matmul(dnu,xt) + torch_matmul(nus,dvt)
    }
  }

  # --- dl_dlb
  dl_dlb = NULL
  if(lb$requires_grad){
    dP_dlb = torch_ones(dim_x)
    dP_dlb[s_x_u >= lb] = 0#this needs to consider ds_dx as well s(x_u)
    dP_dlb = torch_diag_embed(dP_dlb$squeeze(3))
    #zero = torch_zeros(n_batch,n_eq,n_x)
    dP_mat = torch_qp_eqcon_mat(-2*rho*dP_dlb,zero)
    dP_dlb = torch_qp_eqcon_mat(dP_dlb,zero)
    df_dlb = -torch_matmul(mat_inv,dP_mat) - dP_dlb

    dv_dlb = torch_lu_solve(df_dlb,lhs_data,lhs_pivots)
    dv_dlb = dv_dlb[,idx_x,][,,idx_x]
    #dx_dlb = dv_dlb + dP_dlb
    dx_dlb = torch_matmul(dP_dv,dv_dlb)
    dl_dlb = torch_matmul(torch_transpose(dx_dlb,2,3),dl_dz)
  }


  # --- dl_dub
  dl_dub = NULL
  if(ub$requires_grad){
    dP_dub = torch_ones(dim_x)
    dP_dub[s_x_u <= ub] = 0#this needs to consider ds_dx as well s(x_u)
    dP_dub = torch_diag_embed(dP_dub$squeeze(3))
    #zero = torch_zeros(n_batch,n_eq,n_x)
    dP_mat = torch_qp_eqcon_mat(-2*rho*dP_dub,zero)
    dP_dub = torch_qp_eqcon_mat(dP_dub,zero)
    df_dub = -torch_matmul(mat_inv,dP_mat) - dP_dub

    dv_dub = torch_lu_solve(df_dub,lhs_data,lhs_pivots)
    dv_dub = dv_dub[,idx_x,][,,idx_x]
    dx_dub = torch_matmul(dP_dv,dv_dub)
    dl_dub = torch_matmul(torch_transpose(dx_dub,2,3),dl_dz)
  }

  # --- dl_dE and dl_dlam --->NOT YET OPERATIONAL
  dl_dE = NULL
  dl_dlam = NULL
  if(any_l_1){
    d1E = torch_ones(dim_x)
    d1E[x_u_abs < lambda_1_E_rho] = 0
    ds_dtau = d1E*torch_sign(x_u)
    dl_dE = (-lambda_1/rho)*dpi_dx*ds_dtau*dl_dz
    #matching dimension --- may need to apply this to all variables...
    dim_E = dim(E)
    if(dim_E[1] == 1){
      dl_dE = torch_sum(dl_dE,dim=1,keepdim=TRUE)
    }
    dl_dE = torch_diag_embed(dl_dE$squeeze(3))

    dl_dlam = (-E_diag/rho)*dpi_dx*ds_dtau*dl_dz
    dl_dlam = torch_sum(dl_dlam)$unsqueeze(1)
    dl_dlam = prep_torch_tensor(dl_dlam)

  }

  # --- out list of grads
  grads = list(Q = dl_dQ,
               p = dl_dp,
               A = dl_dA,
               b = dl_db,
               lb = dl_dlb,
               ub = dl_dub,
               E = dl_dE,
               lambda_1 = dl_dlam
  )

  return(grads)


}

#' @export
torch_qp_int_grads_admm<-function(x,
                                  lams,
                                  nus,
                                  E,
                                  lambda_1,
                                  d_x,
                                  d_lam,
                                  d_nu,
                                  n_eq,
                                  n_ineq,
                                  any_lb,
                                  any_ub,
                                  any_l_1,
                                  tau = 10^-6,
                                  Q_requires_grad = TRUE,
                                  A_requires_grad = TRUE)
{
  # --- compute regular gradients
  grads = torch_qp_int_grads(x = x,
                             lams = lams,
                             nus = nus,
                             d_x = d_x,
                             d_lam =  d_lam,
                             d_nu =  d_nu,
                             n_eq = n_eq,
                             n_ineq = n_ineq,
                             Q_requires_grad = Q_requires_grad,
                             A_requires_grad = A_requires_grad,
                             G_requires_grad = FALSE)

  # --- adjust dh to compute dlbs
  x_size = get_size(x)
  n_x = x_size[2]
  dlbs = dubs = NULL
  if(any_lb & any_ub){
    dlbs = -grads$h[,1:n_x,]
    dubs = grads$h[,(n_x+1):(2*n_x),]
  }
  else if(any_lb){
    dlbs = -grads$h[,1:n_x,]
  }
  else{
    dubs = grads$h[,1:n_x,]
  }

  # --- quadratic approximation for gradient of E and lambda_1
  if(any_l_1){
    dl_dQ = grads[['Q']]
    # --- prep:
    EE = torch_crossprod(E)
    Ex = torch_matmul(E,x)
    Ex_abs = torch_abs(Ex) + tau
    Ex_abs_inv = 1/Ex_abs
    Ex_abs_inv2 = Ex_abs_inv^2
    denom = torch_diag_embed(Ex_abs_inv$squeeze(3))
    denom2 = torch_diag_embed(Ex_abs_inv2$squeeze(3))
    mat = torch_matmul(denom,EE)
    # --- dlambda_1
    dlambda_1 = torch_matmul(dl_dQ,mat)
    dlambda_1 = torch_trace_batch(dlambda_1)

    # --- dE
    dl_dg = torch_matmul(dl_dQ,EE)
    d_dl_dg = torch_diag_batch(dl_dg)
    dl_df = torch_matmul(denom,dl_dQ)
    dEf = torch_matmul(E,dl_df) + torch_matmul(E,torch_transpose(dl_df,2,3))

    dg_dE = - torch_matmul(denom2,torch_sign(Ex))
    dg_dE = torch_matmul(dg_dE,torch_transpose(x,2,3))
    dEg = torch_matmul(d_dl_dg,dg_dE)

    dEs = dEg + dEf
  }

  grads = grads[c("Q","p","A","b","G")]
  grads$lb = dlbs
  grads$ub = dubs

  # -- -add grads for E and lambda_1
  if(any_l_1){
    grads$E = dEs
    grads$lambda_1 = dlambda_1
  }

  return(grads)

}

#' @export
torch_qp_grad_admm_dep2<-function(x,
                             dl_dz,
                             u,
                             nus,
                             mat_inv,
                             mat_data,
                             mat_pivots,
                             A,
                             lb,
                             ub,
                             E,
                             lambda_1,
                             rho)
{
  # --- prep:
  dim_x = dim(x)
  n_x = dim_x[2]
  n_batch = dim(dl_dz)[1]
  n_eq = get_ncon(A)
  any_eq = n_eq > 0
  any_l_1 = get_any(lambda_1, threshold = 0)
  idx_x = 1:n_x
  idx_eq = (n_x+1):(n_x+n_eq)

  xt = torch_transpose(x,2,3)
  x_u = x + u
  x_u_abs = torch_abs(x_u)

  # create tau
  lambda_1_E_rho = 0
  if(any_l_1){
  E_diag = torch_diagonal(E,dim1=2,dim2=3)
  E_diag = prep_torch_tensor(E_diag,target_dim=3)
  lambda_1_E = lambda_1 * E_diag
  lambda_1_E_rho = lambda_1_E/rho#tau
  }

  # --- derivative of the projection operator:
  dpi_dx = torch_ones(dim_x)
  dpi_dx[x_u > ub] = 0
  dpi_dx[x_u < lb] = 0

  # --- derivative of the soft-thresholding:
  ds_dx = torch_ones(dim_x)
  ds_dx[x_u_abs < lambda_1_E_rho] = 0

  # --- dl_dx: cahin rule
  dl_dx = dl_dz*dpi_dx*ds_dx

  # --- rhs:
  if(any_eq){
    zeros = torch_zeros(c(n_batch,n_eq,1))
    rhs = torch_cat(list(-dl_dx,zeros),dim=2)
  }
  else{
    rhs = -dl_dx
  }
  #d_vec = torch_matmul(mat_inv,rhs)
  d_vec = torch_lu_solve(rhs,mat_data,mat_pivots)
  dx = d_vec[,idx_x,,drop=F]
  dxt = torch_transpose(dx,2,3)
  # --- dl_dp
  dl_dp = dx

  # --- dl_dQ
  dl_dQ = 0.5*( torch_matmul(dx,xt) + torch_matmul(x,dxt) )

  # --- dl_dA and dl_db
  dl_db = NULL
  dl_dA = NULL
  if(any_eq){
    dv = d_vec[,idx_eq,,drop=F]
    dvt = torch_transpose(dv,2,3)
    dl_db = -dv
    dl_dA = torch_matmul(dv,xt) + torch_matmul(nus,dxt)
  }

  # --- dl_dlb
  dl_dlb = torch_ones(dim_x)
  dl_dlb[x_u >= lb] = 0
  dl_dlb = dl_dlb*dl_dz

  # --- dl_dub
  dl_dub = torch_ones(dim_x)
  dl_dub[x_u <= ub] = 0
  dl_dub = dl_dub*dl_dz

  # --- dl_dE and dl_dlam
  dl_dE = NULL
  dl_dlam = NULL
  if(any_l_1){
    d1E = torch_ones(dim_x)
    d1E[x_u_abs < lambda_1_E_rho] = 0
    ds_dtau = d1E*torch_sign(x_u)
    dl_dE = (-lambda_1/rho)*dpi_dx*ds_dtau*dl_dz
    #matching dimension --- may need to apply this to all variables...
    dim_E = dim(E)
    if(dim_E[1] == 1){
      dl_dE = torch_sum(dl_dE,dim=1,keepdim=TRUE)
    }
    dl_dE = torch_diag_embed(dl_dE$squeeze(3))

    dl_dlam = (-E_diag/rho)*dpi_dx*ds_dtau*dl_dz
    dl_dlam = torch_sum(dl_dlam)$unsqueeze(1)
    dl_dlam = prep_torch_tensor(dl_dlam)

  }

  # --- out list of grads
  grads = list(Q = dl_dQ,
               p = dl_dp,
               A = dl_dA,
               b = dl_db,
               lb = dl_dlb,
               ub = dl_dub,
               E = dl_dE,
               lambda_1 = dl_dlam
               )

  return(grads)


}
