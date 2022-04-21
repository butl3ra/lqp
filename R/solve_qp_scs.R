#' @export
torch_solve_qp_scs<-function(Q,
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
    Amat = torch_cat(list(A,G),2)
    bvec = torch_cat(list(b,h),2)
  }
  else{
    Amat = G
    bvec = h
  }


  # --- convert to arrays:
  Q_a = as_array(Q)
  p_a = as_array(p)
  Amat_a = as_array(Amat)
  bvec_a = as_array(bvec)

  x = array(0,c(n_batch,n_x,1))
  lams = array(0,c(n_batch,n_eq + n_ineq,1))
  slacks = array(0,c(n_batch,n_eq + n_ineq,1))

  # --- cone and scs control:
  cone = list(z = n_eq,l = n_ineq)
  scs_control = scs::scs_control(eps_rel = control$tol, eps_abs = control$tol,max_iters = control$max_iters)

  # --- main loop: sequential
  for(i in 1:n_batch){
    sol = scs::scs(obj = p_a[i,,],
                   P = Q_a[i,,],
                   A = Amat_a[i,,],
                   b = bvec_a[i,,],
                   cone = cone,
                   initial = NULL,
                   control = scs_control)
    x[i,,] = sol$x
    lams[i,,] = sol$y
    slacks[i,,] = sol$s

  }
  # --- convert to tensor:
  x = as_torch_tensor(x)
  lams = as_torch_tensor(lams)
  slacks = as_torch_tensor(slacks)

  # --- make output list:
  out = list(x = x,
             lams = lams,
             slacks = slacks)

  if(!output_as_list){
    out = torch_cat(out,dim = 2)
  }

  return(out)
}


#' @export
torch_qp_scs_grads<-function(dl_dx,
                             x,
                             lams,
                             slacks,
                             Q,
                             A,
                             b,
                             G,
                             h)
{

  # --- prep:
  n_x = get_n_x(x)

  A_size = get_size(A)
  n_eq = A_size[2]
  any_eq = n_eq > 0

  G_size = get_size(G)
  n_ineq = G_size[2]

  n_batch = G_size[1]
  n_con = n_eq + n_ineq
  zero_x = torch_zeros(c(n_batch,n_x,1))
  zero_con = torch_zeros(c(n_batch,n_con,1))

  # --- init: w
  u_star = torch_cat(list(x,lams),dim=2)
  v_star = torch_cat(list(zero_x,slacks),dim=2)
  w = u_star - v_star

  # --- if any eq:
  if(any_eq){
    Amat = torch_cat(list(A,G),2)
    bvec = torch_cat(list(b,h),2)
  }
  else{
    Amat = G
    bvec = h
  }
  Amat_size = get_size(Amat)

  # --- M matrix:
  bottom_right = torch_zeros(Amat_size[c(1,2,2)])
  AmatT = torch_transpose(Amat,2,3)
  lhs_u = torch_cat(list(Q,AmatT),3)
  lhs_l = torch_cat(list(-Amat, bottom_right ) ,3)
  M = torch_cat(list(lhs_u,lhs_l), 2)
  I = torch_eye(n_x + n_con)$unsqueeze(1)



  # --- Derivative of euclidean projection operator:
  idx = torch_tensor((n_x + n_eq+1):(n_x + n_eq+n_ineq),dtype = torch_int())
  w_y = w$index_select(dim=2,index = idx)
  D_w_y = 0.5*(torch_sign(w_y)+1)
  ones = torch_ones(c(n_batch,n_x + n_eq,1))
  D = torch_cat(list(ones,D_w_y),2)
  if(F){
  I_batch = torch_eye(n_x + n_eq)
  I_zero = torch_zeros(c(n_x+n_eq,n_ineq))
  I_batch = torch_cat(list(I_batch,I_zero),dim=2)
  I_batch = I_batch$unsqueeze(1)
  I_batch = prep_batch_size(I_batch,n_batch)
  idx = torch_tensor((n_x + n_eq+1):(n_x + n_eq+n_ineq),dtype = torch_int())
  w_y = w$index_select(dim=2,index = idx)
  D_w_y = 0.5*(torch_sign(w_y)+1)
  D_w_y = torch_diag_embed(D_w_y$squeeze(3))
  I_zero = torch_zeros(c(n_ineq,n_x+n_eq))$unsqueeze(1)
  I_zero = prep_batch_size(I_zero,n_batch)
  D_w_y = torch_cat(list(I_zero,D_w_y),dim=3)
  D = torch_cat(list(I_batch,D_w_y),2)
  }

  # --- Core system of Equations:
  rhs = torch_cat(list(-dl_dx,zero_con),dim=2)
  rhs = D*rhs#torch_matmul(D,rhs)
  mat = M*torch_transpose(D,2,3) - torch_diag_embed(D$squeeze(3)) + I#torch_matmul(M,D) - D + I  #+ 10^-8*I
  d = linalg_solve(torch_transpose(mat,2,3),rhs)

  # --- d:
  dx = d[,1:n_x,,drop=F]
  dy = d[,(n_x+1):(n_x + n_con),,drop=F]

  # --- gradients:
  dl_dp = dx
  dl_dQ = NULL
  dl_dA = NULL
  dl_db = NULL
  dl_dG = NULL
  dl_dh = NULL

  # --- if required:
  any_grad = check_requires_grad(Q) | check_requires_grad(A) | check_requires_grad(G) | check_requires_grad(b) | check_requires_grad(h)
  if(any_grad){
    # --- prep:
    xt = torch_transpose(x,2,3)
    dxt = torch_transpose(dx,2,3)

    # --- dl_dQ
    dl_dQ = 0.5*(torch_matmul(dx,xt) + torch_matmul(x,dxt) )

    # --- dl_dA:
    dl_dAmat = torch_matmul(lams,dxt) - torch_matmul(dy,xt)

    # --- remaining:
    if(any_eq){
      dl_dA = dl_dAmat[,1:n_eq,]
      dl_db = dy[,1:n_eq,]
      dl_dG = dl_dAmat[,(n_eq+1):(n_eq + n_con),]
      dl_dh = dy[,(n_eq+1):(n_eq + n_con),]
    }
    else{
      dl_dh = dy
      dl_dG = dl_dAmat
    }

  }

  # list of grad---
  grads = list(p = dl_dp,
               Q = dl_dQ,
               A = dl_dA,
               b = dl_db,
               G = dl_dG,
               h = dl_dh)



  return(grads)

}
