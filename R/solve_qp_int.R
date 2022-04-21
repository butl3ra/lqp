#' @export
torch_solve_qp_int<-function(Q,
                             p,
                             A,
                             b,
                             G,
                             h,
                             control = nn_qp_control(solver = 'int',
                                                     max_iters = 10,
                                                     tol = 10^-4),
                             x = NULL,
                             nus = NULL,
                             lams = NULL,
                             slacks = NULL,
                             U_Q = NULL,
                             U_S = NULL,
                             R = NULL,
                             ...)
{

  # --- unpacking control:
  tol = control$tol
  tol_relative = control$tol_relative
  max_iters = control$max_iters
  tol_method = control$tol_method
  verbose = control$verbose
  output_as_list = control$output_as_list
  warm_start = control$warm_start
  int_reg = control$int_reg

  # --- check for warm start
  warm_start_vars = check_warm_start(x = x,
                                     lams = lams,
                                     slacks = slacks)
  warm_start_vars = warm_start_vars & warm_start

  # --- check warm start for matrices
  warm_start_mats = check_warm_start(U_Q = U_Q,
                                     U_S = U_S,
                                     R = R)
  warm_start_mats = warm_start_mats & warm_start

  # --- prep:
  x_size = get_size(p)
  n_batch = x_size[1]
  n_x = x_size[2]
  idx_x = 1:n_x

  any_A = get_any(A)
  A_size = get_size(A)
  n_eq = A_size[2]
  any_eq = n_eq > 0

  any_G = get_any(G)
  G_size = get_size(G)
  n_ineq = G_size[2]

  n_con = n_eq + n_ineq

  # --- initialize:
  GT = torch_transpose(G,2,3)
  AT = NULL
  if(any_eq){
    AT = torch_transpose(A,2,3)
  }

  #--- pre-factorization:
  if(!warm_start_mats){
    mat_factor = torch_qp_int_pre_factor_kkt(Q = Q,
                                             G = G,
                                             GT = GT,
                                             A = A,
                                             AT = AT,
                                             U_Q = NULL)
    U_Q = mat_factor$U_Q
    U_S = mat_factor$U_S
    R = mat_factor$R
  }


  # --- initialize:
  if(!warm_start_vars){
    sol_init = torch_qp_int_init(Q = Q,
                                 p = p,
                                 G = G,
                                 h = h,
                                 A = A,
                                 b = b,
                                 GT = GT,
                                 AT = AT,
                                 U_Q = U_Q,
                                 U_S = U_S,
                                 R = R,
                                 int_reg = int_reg)
    # --- unpack init:
    x = sol_init$x #main x
    s = sol_init$s #slacks
    z = sol_init$z #lams
    y = sol_init$y # nus
  }
  else{
    s = slacks
    z = lams
    y = nus
  }


  # --- main loop:
  prev_error = Inf
  idx = 1:max_iters
  one_step = torch_ones(c(n_batch,1))
  ry = torch_zeros(n_batch,1,1)
  rhs_corr = torch_zeros(n_batch,n_x+2*n_ineq + n_eq,1)
  zero_x = torch_zeros(n_batch,n_x,1)
  zero_v = torch_zeros(n_batch,n_ineq + n_eq,1)

  for(iter in idx){
    #print(i)
    # z is lambda, y is v, x is z,
    #right hand side values; positive of them

    # --- rhs:
    rx = torch_matmul(GT,z) + torch_matmul(Q,x) + p
    rs = z
    rz = torch_matmul(G,x) + s - h
    if(any_eq){
      rx = torch_matmul(AT,y) + rx
      ry = torch_matmul(A,x) - b
    }

    mu = torch_sum(s*z,dim=2) / n_ineq
    pri_resid = torch_norm_2(ry,dim=2) + torch_norm_2(rz,dim=2)
    dual_resid = torch_norm_2(rx,dim=2)
    resid = (pri_resid + dual_resid)/2 +  mu#n_ineq *
    d = z / s

    #---- stopping tolerances:
    error = torch_reduce(resid,method = tol_method)

    # --- verbose
    if(verbose){
      cat('iteration: ', iter, '\n')
      cat('duality gap = ', as.numeric(error),'\n')
    }
    stop_1 = as.logical(error < tol) & iter > 1# guarantee at least one iteration
    stop_2 = as.logical(abs(prev_error - error) < tol_relative )#not making significant improvements
    do_stop = stop_1 | stop_2
    if(do_stop){
      break
    }
    prev_error = error


    #---- factorization
    U_S = torch_qp_int_factor_kkt(U_S = U_S,
                                  R = R,
                                  d = d,
                                  n_eq = n_eq,
                                  n_ineq = n_ineq,
                                  eps = int_reg)

    #---- affine step
    aff_sol = torch_qp_int_solve_kkt(U_Q = U_Q,
                                     d = d,
                                     G = G,
                                     A = A,
                                     AT = AT,
                                     GT = GT,
                                     U_S = U_S,
                                     rx = rx,
                                     rs = rs,
                                     rz = rz,
                                     ry = ry,
                                     n_ineq = n_ineq,
                                     n_eq = n_eq)
    dx_aff = aff_sol$dx
    ds_aff = aff_sol$ds
    dz_aff = aff_sol$dz
    dy_aff = aff_sol$dy

    # compute centering directions
    z_step = torch_qp_int_get_step(z,dz_aff)
    s_step = torch_qp_int_get_step(s,ds_aff)
    alphas = torch_cat(list(z_step,s_step,one_step),2)
    alpha = torch_min(alphas,2,keepdim=T)[[1]]
    alpha = alpha$unsqueeze(3)
    alpha = alpha*0.999

    s_plus_step = s + alpha * ds_aff
    z_plus_step = z + alpha * dz_aff
    sig = ( torch_sum(s_plus_step*z_plus_step,2) / (torch_sum(s*z,2)))^3

    mu_sig = -mu * sig
    mu_sig = mu_sig$unsqueeze(3)

    non_zero = (mu_sig + ds_aff * dz_aff) / s
    cor_sol = torch_qp_int_solve_kkt(U_Q = U_Q,
                               d = d,
                               G = G,
                               A = A,
                               AT = AT,
                               GT = GT,
                               U_S = U_S,
                               rx = rx*0,
                               rs = non_zero,
                               rz = rz*0,
                               ry = ry*0,
                               n_ineq = n_ineq,
                               n_eq = n_eq)


    dx = dx_aff + cor_sol$dx
    ds = ds_aff + cor_sol$ds
    dz = dz_aff + cor_sol$dz
    if(any_eq){
      dy = dy_aff + cor_sol$dy
    }
    else{
      dy = NULL
    }



    z_step = torch_qp_int_get_step(z,dz)
    s_step = torch_qp_int_get_step(s,ds)
    alphas = torch_cat(list(z_step,s_step,one_step),2)
    alpha = torch_min(alphas,2,keepdim=T)[[1]]
    alpha = alpha$unsqueeze(3)
    alpha = alpha*0.999




    x = x + alpha * dx
    s = s + alpha * ds
    z = z + alpha * dz
    if(any_eq){
      y = y + alpha * dy #if neq > 0 else None
    }
  }

  # --- final factorizaation for backwards:
  d = z / s
  U_S = torch_qp_int_factor_kkt(U_S = U_S,
                                R = R,
                                d = d,
                                n_eq = n_eq,
                                n_ineq = n_ineq,
                                eps = int_reg)

  lams = torch_clamp(z, 10^-8) #--- this is also done in backward
  slacks = torch_clamp(s,10^-8) #--- this is also done in backward
  if(any_eq){
    nus = y
  }

  # --- concatenate if not output as list...
  if(!output_as_list){
    U_Q = torch_reshape_mat(mat = U_Q,forward = TRUE)
    U_S = torch_reshape_mat(mat = U_S,forward = TRUE)
    R = torch_reshape_mat(mat = R,forward = TRUE)
  }
  # --- make output list:
  out = list(x = x,
             lams = lams,
             slacks = slacks)
  if(any_eq){
      out$nus = nus
   }
  out$U_Q = U_Q
  out$U_S = U_S
  out$R = R
  if(!output_as_list){
    out = torch_cat(out,dim = 2)
  }



  return(out)


}

#' @export
torch_qp_int_get_step<-function(v,dv)
{

  a = -v/dv
  z = torch_threshold_(a,0,Inf)
  torch_min(a,2,keepdim=F)[[1]]
}

#' @export
torch_qp_int_solve_kkt<-function(U_Q,
                                 d,
                                 G,
                                 A,
                                 AT,
                                 GT,
                                 U_S,
                                 rx ,
                                 rs,
                                 rz,
                                 ry,
                                 n_ineq,
                                 n_eq)
{
  # --- init:
  n_con = n_eq + n_ineq
  any_eq = n_eq > 0
  invQ_rx = torch_cholesky_solve(rx,U_Q,upper = TRUE)

  # --- ineq and eq con:
  H = torch_matmul(G,invQ_rx) + rs/d - rz
  if(any_eq){
    H1 = torch_matmul(A,invQ_rx)- ry
    H = torch_cat(list(H1,H),dim = 2)
  }
  w = -torch_cholesky_solve(H,U_S,upper = TRUE)

  # --- g1:
  if(any_eq){
    idx = 1:n_eq
    n_idx = (1:n_con)[-idx]
    w_idx = w[,idx,,drop=F]
    w_n_idx =  w[,n_idx,,drop=F]
    g1 = -rx - torch_matmul(GT,w_n_idx)
    g1 = g1 - torch_matmul(AT,w_idx)
  }
  else{
    w_n_idx = w
    g1 = -rx - torch_matmul(GT,w_n_idx)
  }
  # --- g2
  g2 = -rs -w_n_idx

  dx = torch_cholesky_solve(g1,U_Q,upper = TRUE)
  ds = g2 / d
  dz = w_n_idx
  dy = NULL
  if(any_eq){
    dy = w_idx #if neq > 0 else None
  }

  return(list(dx = dx, ds = ds, dz = dz, dy = dy))
}

#' @export
torch_qp_int_init<-function(Q,
                            p,
                            A,
                            AT,
                            b,
                            G,
                            GT,
                            h,
                            U_Q,
                            U_S,
                            R,
                            int_reg = 10^-6)
{
  # --- prep:
  x_size = get_size(p)
  n_batch = x_size[1]
  n_x = x_size[2]
  n_ineq = get_ncon(G)
  n_eq = get_ncon(A)
  any_eq = n_eq > 0

  # --- rhs
  d = torch_ones(c(n_batch,n_ineq,1))
  rx = p
  rz = -h
  if(n_eq > 0){
    ry = -b
  }
  rs = torch_zeros(c(n_batch,n_ineq,1))

  # --- pre-factorization:
  U_S = torch_qp_int_factor_kkt(U_S = U_S,
                                R = R,
                                d = d,
                                n_eq = n_eq,
                                n_ineq = n_ineq,
                                eps = int_reg)

  # --- solve:
  kkt_sol = torch_qp_int_solve_kkt(U_Q = U_Q,
                                   d = d,
                                   G = G,
                                   A = A,
                                   AT = AT,
                                   GT = GT,
                                   U_S = U_S,
                                   rx = rx,
                                   rs = rs,
                                   rz = rz,
                                   ry = ry,
                                   n_ineq = n_ineq,
                                   n_eq = n_eq)
    x = kkt_sol$dx
    s = kkt_sol$ds
    z = kkt_sol$dz
    y = kkt_sol$dy

    # --- set lambdas and slacks positive:
    min_s = torch_min(s,2)[[1]]
    min_z = torch_min(z,2)[[1]]
    s = s + torch_threshold_(1-min_s,1,0)$unsqueeze(3)
    z = z + torch_threshold_(1-min_z,1,0)$unsqueeze(3)

    # --- out
    out = list(x = x,
               s = s,
               z = z,
               y = y)


    return(out)
}

#' @export
torch_qp_int_factor_kkt<-function(U_S,
                                  R,
                                  d,
                                  n_eq,
                                  n_ineq,
                                  eps = 10^-6)
{
  n_con = n_eq + n_ineq
  n_batch = get_size(U_S)[1]
  zeros = torch_zeros(n_batch,n_ineq,n_eq)

  #updates U_S for realization of d
  d_diag = torch_diag_embed(1/d$squeeze(3))

  reg = torch_eye(n_ineq)$unsqueeze(1)
  mat = linalg_cholesky(R + d_diag + eps*reg)
  mat = torch_transpose(mat,2,3)
  mat = torch_cat(list(zeros,mat),dim=3)

  U1 = U_S[,1,,drop=F]
  out = torch_cat(list(U_S[,1,,drop=F],mat),dim=2)

  return(out)

}



#' @export
torch_qp_int_pre_factor_kkt<-function(Q,
                                      G,
                                      GT,
                                      A,
                                      AT,
                                      U_Q = NULL)
{
  # --- logic:
  #S =  [ A Q^{-1} A^T        A Q^{-1} G^T           ]
  #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
  #S = rbind(cbind(A_invQ_AT,A_invQ_GT),cbind(G_invQ_AT,G%*%invQ_GT +diag(n_ineq)))
  #chol(S)[1,] == U_S[1,]

  # --- prep:
  n_batch = get_size(Q)[1]
  n_eq = get_ncon(A)
  n_ineq = get_ncon(G)
  n_con = n_eq + n_ineq
  any_ineq = n_ineq > 0
  any_eq = n_eq > 0

  # --- Cholesky factorization:
  if(is.null(U_Q)){
    U_Q = linalg_cholesky(Q)
    U_Q = torch_transpose(U_Q,2,3)
  }
  U_S = torch_zeros(c(n_batch,n_con,n_con))

  # --- ineq:
  invQ_GT = torch_cholesky_solve(GT,U_Q,upper = T)
  R = torch_matmul(G,invQ_GT)

  # --- if any equality constraints:
  if(any_eq){
    # --- eq
    invQ_AT = torch_cholesky_solve(AT,U_Q,upper = T)
    A_invQ_AT = torch_matmul(A,invQ_AT)
    U11  = linalg_cholesky(A_invQ_AT)
    U11 = torch_transpose(U11,2,3)


    # --- cross product terms
    G_invQ_AT = torch_matmul(G,invQ_AT)
    U12 = linalg_solve(U11,torch_transpose(G_invQ_AT,2,3))

    U1 = torch_cat(list(U11,U12),dim=3)
    zeros = torch_zeros(n_batch,n_ineq,n_con)
    U_S = torch_cat(list(U1,zeros),dim=2)

    R = R - torch_crossprod(U12,U12)
  }


  return(list(U_Q = U_Q, U_S = U_S, R = R))


}

#' @export
torch_qp_int_init_dep<-function(Q,
                                p,
                                G,
                                h,
                                A,
                                b,
                                GT,
                                AT,
                                n_batch,
                                n_eq,
                                n_ineq,
                                n_x)
{
  # --- initialization:
  any_eq = n_eq > 0
  Id = torch_eye_embed(n_batch,n_ineq)
  if(any_eq){
    zeros_1 = torch_zeros(n_batch,n_ineq,n_eq)
    zeros_2 = torch_zeros(n_batch,n_eq,n_ineq)
    zeros_3 = torch_zeros(n_batch,n_eq,n_eq)

    m1 = torch_cat(list(Q,GT,AT),3)
    m2 = torch_cat(list(G,-Id,zeros_1),3)
    m3 = torch_cat(list(A,zeros_2,zeros_3),3)

    lhs = torch_cat(list(m1,m2,m3),2)
    rhs = torch_cat(list(-p,h,b),2)

  }
  else{
    m1 = torch_cat(list(Q,GT),3)
    m2 = torch_cat(list(G,-Id),3)

    lhs = torch_cat(list(m1,m2),2)
    rhs = torch_cat(list(-p,h),2)
  }

  # --- solve linear system:
  sol = linalg_solve(lhs,rhs)

  # --- x:
  x = sol[,1:n_x,]

  # --- y:
  if(any_eq){
    idx = n_z + n_ineq + n_eq
    idx = (idx):(idx + n_eq - 1)
    y = sol[,idx,,drop=F] # nus
  }
  else{
    y = NULL
  }

  # --- set lambdas and slacks positive:
  z = torch_matmul(G,x)-h
  s = -z
  min_s = torch_min(s,2)[[1]]
  min_z = torch_min(z,2)[[1]]
  s = s + torch_threshold_(1-min_s,1,0)$unsqueeze(3)
  z = z + torch_threshold_(1-min_z,1,0)$unsqueeze(3)

  # --- out
  out = list(x = x,
             s = s,
             z = z,
             y = y)
}

#' @export
torch_qp_int_grads<-function(x,
                             lams,
                             nus,
                             d_x,
                             d_lam,
                             d_nu,
                             n_eq,
                             n_ineq,
                             Q_requires_grad = TRUE,
                             A_requires_grad = TRUE,
                             G_requires_grad = TRUE
                             )
{
  # --- prep:
  x_t = torch_transpose(x,2,3)
  d_x_t = torch_transpose(d_x,2,3)

  lams_t = torch_transpose(lams,2,3)
  d_lam_t = torch_transpose(d_lam,2,3)

  # --- gradients:
  dps = d_x
  dQs = NULL
  if(Q_requires_grad){
    dQs = 0.5 * (torch_matmul(x,d_x_t) + torch_matmul(d_x,x_t) )
  }

  # --- ineqaulity
  dGs = NULL
  dhs = NULL
  if(n_ineq > 0){
    if(G_requires_grad){
      D_lams = torch_diag_embed(lams$squeeze(3))
      dGs = torch_matmul(D_lams,torch_matmul(d_lam,x_t)) + torch_matmul(lams,d_x_t)
    }
    #dGs = torch_matmul(d_lam,x_t) + torch_matmul(lams,d_x_t)
    #dGs = torch_matmul(torch_diag_embed(lams$squeeze(3)) ,dGs)
    dhs = -lams*d_lam
  }
  # --- equality
  dAs = NULL
  dbs = NULL
  if(n_eq > 0 ){
    if(A_requires_grad){
      nus_t = torch_transpose(nus,2,3)
      d_nu_t = torch_transpose(d_nu,2,3)
      dAs = torch_matmul(d_nu,x_t) + torch_matmul(nus,d_x_t)
    }
    dbs = -d_nu
  }

  # --- out list of grads
  grads = list(Q = dQs,
               p = dps,
               G = dGs,
               h = dhs,
               A = dAs,
               b = dbs)
}
