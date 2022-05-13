#' @export
torch_solve_qp_uncon_l1<-function(Q,
                                  p,
                                  E,
                                  lambda_1,
                                  control,
                                   ...)
{

  # --- solve the dual: a box-constrained QP,
  # --- and then recover the primal solution

  # --- prep ub and lb
  dim_Q = dim(Q)
  n_z = dim_Q[2]
  ones = torch_ones(c(1,n_z,1))
  lb = -ones*lambda_1
  ub = ones*lambda_1


  #--- prep objective of dual objective:
  Q_inv = torch_inverse(Q)
  xy = -p
  p_dual = torch_matmul(Q_inv,xy)
  p_dual = torch_neg(p_dual)

  # --- if E is more than just the identity:
  if(!is.null(E)){
    E_t = torch_transpose(E,2,3)
    Q_inv_dual = torch_matmul(torch_matmul(E_t,Q_inv),E)
    p_dual = torch_matmul(E_t,p_dual)#torch_matmul(p_dual,E)
  }
  else{
    Q_inv_dual = Q_inv
  }

  # --- solve the dual problem : a box constrained QP
  control$rho = 100/control$rho
  control$output_as_list=T

  sol = torch_solve_qp_admm(Q = Q_inv_dual,
                            p = p_dual,
                            A = NULL,
                            b = NULL,
                            G = NULL,
                            lb = lb,
                            ub = ub,
                            E = NULL,
                            lambda_1 = NULL,
                            control = control)
  v = sol$x

  # --- Retrieve the primal solution from the dual:
  if(!is.null(E)){
    z_hat = xy - torch_matmul(E_t,v)#torch_matmul(v,E)
  }
  else{
    z_hat = xy - v
  }
  z = torch_matmul(Q_inv,z_hat)
  #z = z$squeeze(3)

  return(z)


  # --- make output list:
  out = list(z = z,
             v = v)

  if(!output_as_list){
    out = torch_cat(out,dim = 2)
  }

  return(out)


}

#' @export
torch_solve_qp_con_l1<-function(Q,
                                p,
                                A = NULL,
                                b = NULL,
                                G = NULL,
                                h = NULL,
                                E,
                                lambda_1,
                                control,
                                ...)
{

  #---- prep
  n_z = ncol(Q)
  n_batch = nrow(Q)
  n_eq = get_ncon(A)
  n_ineq = get_ncon(G)
  any_eq = n_eq > 0
  any_ineq = n_ineq >0

  # --- decision variables:
  # [v,n,u]
  V = torch_inverse(Q)


  # --- build main matrix:
  E_t = torch_transpose(E,2,3)
  EVE = torch_matmul(torch_matmul(E,V),E_t)
  if(any_eq){
    A_t = torch_transpose(A,2,3)
    EVA = torch_matmul(torch_matmul(E,V),A_t)
    AVA = torch_matmul(torch_matmul(A,V),A_t)
  }
  if(any_ineq){
    G_t = torch_transpose(G,2,3)
    EVG = torch_matmul(torch_matmul(E,V),G_t)
    GVG = torch_matmul(torch_matmul(G,V),G_t)
  }
  if(any_eq & any_ineq){
    AVG = torch_matmul(torch_matmul(A,V),G_t)
  }
  # --- assemble:
  if(any_eq & !any_ineq){
    top = torch_cat(list(EVE,EVA),dim = 3) #cbind(EVE,EVA)
    mid = torch_cat(list(torch_transpose(EVA,2,3),AVA),dim = 3) #cbind(t(EVA),AVA)
    P = torch_cat(list(top,mid),dim = 2)#rbind(top,mid)
  }
  else if(any_ineq & !any_eq){
    top = torch_cat(list(EVE,EVG),dim = 3) #cbind(EVE,EVG)
    bot = torch_cat(list(torch_transpose(EVG,2,3),GVG),dim = 3) #cbind(t(EVG),GVG)
    P = torch_cat(list(top,bot),dim = 2)#rbind(top,bot)
  }
  else if(any_eq & any_ineq){
    top = torch_cat(list(EVE,EVA,EVG),dim = 3)#cbind(EVE,EVA,EVG)
    mid = torch_cat(list(torch_transpose(EVA,2,3),AVA,AVG),dim = 3) #cbind(t(EVA),AVA,AVG)
    bot = torch_cat(list(torch_transpose(EVG,2,3),torch_transpose(AVG,2,3),GVG),dim = 3)#cbind(t(EVG),t(AVG),GVG)
    P = torch_cat(list(top,mid,bot),dim = 2)#rbind(top,mid,bot)
  }
  else{
    P = EVE
  }
  # --- stability:
  P = P + 10^-12 * torch_eye(n_z + n_eq + n_ineq)$unsqueeze(1)#10^-12

  # --- linear components:
  Vp = torch_matmul(V,p)#V%*%p
  obj = torch_matmul(E,Vp)#E%*%Vp
  if(any_eq){
    obj_eq = torch_matmul(A,Vp) + b
    obj = torch_cat(list(obj,obj_eq),dim=2)#c(obj, A%*%Vp + b)
  }
  if(any_ineq){
    obj_ineq = torch_matmul(G,Vp) + h
    obj = torch_cat(list(obj,obj_ineq),dim=2)#c(obj,G%*%Vp +h )
  }


  # --- solve dual:
  n_all = n_z + n_eq + n_ineq
  lb = -lambda_1*torch_ones(c(n_batch,n_z,1))
  ub = lambda_1*torch_ones(c(n_batch,n_z,1))
  if(any_eq){
    big = torch_ones(c(n_batch,n_eq,1))*10^8
    lb = torch_cat(list(lb,-big),dim=2)
    ub = torch_cat(list(ub,big),dim=2)
  }
  if(any_ineq){
    big = torch_ones(c(n_batch,n_ineq,1))*10^8
    zero = torch_zeros(c(n_batch,n_ineq,1))
    lb = torch_cat(list(lb,zero),dim=2)
    ub = torch_cat(list(ub,big),dim=2)
  }

  control = nn_qp_control(solver = 'admm',
                          output_as_list = FALSE,
                          tol_primal = 10^-3,
                          tol_dual = 10^-3,
                          max_iters = 1000,
                          backprop = 'fixed_point',
                          verbose=FALSE,
                          do_D_crossprod = FALSE,
                          unroll_grad = TRUE)
  control$rho = 1
  n_obs = get_n_obs_proxy(P)#  --- scaling:
  sol = torch_solve_qp_admm(Q = P/n_obs,
                            p = obj/n_obs,
                            A = NULL,
                            b = NULL,
                            G = NULL,
                            lb = lb,
                            ub = ub,
                            E = NULL,
                            lambda_1 = NULL,
                            control = control)
  v = sol[,1:n_z,]
  rhs = - p - torch_matmul(E_t,v)# -p - E_t%*%v
  if(any_eq){
    nu = sol[, (n_z + 1):(n_z+n_eq), ]
    rhs = rhs - torch_matmul(A_t,nu)#rhs - A_t%*%nu
  }
  if(any_ineq){
    mu = sol[, (n_z + n_eq + 1):(n_all),]
    rhs = rhs - torch_matmul(G_t,mu)#rhs - t(G)%*%mu
  }

  # --- primal solution:
  z = torch_matmul(V,rhs) #V%*%rhs

  return(z)
  # --- scs:



  # --- solve dual:
  n_all = n_z + n_eq + n_ineq
  D = torch_eye(n_all)
  G_v = D[1:n_z,]
  G_d = torch_cat(list(-G_v,G_v))
  h_d = lambda_1*torch_ones(c(n_batch,2*n_z,1))
  if(any_ineq){
    G_u = -D[(n_z + n_eq+1):n_all ,]
    G_d = torch_cat(list(G_d,G_u))
    zero = torch_zeros(c(n_batch,n_ineq,1))
    h_d = torch_cat(list(h_d,zero),dim=2)
  }
  G_d = prep_batch_size(G_d$unsqueeze(1),n_batch)

  # --- solve the dual:
  sol_index_list = make_sol_index_list('scs',n_x = (n_z+n_eq+n_ineq),n_eq = 0,n_ineq = 2*n_z + n_ineq)
  control = nn_qp_control(solver = 'scs',
                output_as_list = FALSE,
                tol_primal = 10^-6,
                tol_dual = 10^-6,
                max_iters = 1000,
                backprop = 'fixed_point',
                verbose=FALSE,
                do_D_crossprod = FALSE)
  sol = nn_qp_scs(Q = P,
                  p = obj,
                  G = G_d,
                  h = h_d,
                  A = NULL,
                  b = NULL,
                  sol_index_list = sol_index_list,
                  control = control)

  # --- unpack:
  v = sol[,1:n_z,]
  rhs = - p - torch_matmul(E_t,v)# -p - E_t%*%v
  if(any_eq){
    nu = sol[, (n_z + 1):(n_z+n_eq), ]
    rhs = rhs - torch_matmul(A_t,nu)#rhs - A_t%*%nu
  }
  if(any_ineq){
    mu = sol[, (n_z + n_eq + 1):(n_all),]
    rhs = rhs - torch_matmul(G_t,mu)#rhs - t(G)%*%mu
  }

  # --- primal solution:
  z = torch_matmul(V,rhs) #V%*%rhs

  return(z)

}


#' @export
torch_solve_qp_con_l1<-function(Q,
                                p,
                                A = NULL,
                                b = NULL,
                                G = NULL,
                                h = NULL,
                                E,
                                lambda_1,
                                control,
                                ...)
{

  #---- prep
  n_z = ncol(Q)
  n_batch = nrow(Q)
  n_eq = get_ncon(A)
  n_ineq = get_ncon(G)
  any_eq = n_eq > 0
  any_ineq = n_ineq >0

  # --- decision variables:
  # [v,n,u]
  V = torch_inverse(Q)


  # --- build main matrix:
  E_t = torch_transpose(E,2,3)
  EVE = torch_matmul(torch_matmul(E,V),E_t)
  if(any_eq){
    A_t = torch_transpose(A,2,3)
    EVA = torch_matmul(torch_matmul(E,V),A_t)
    AVA = torch_matmul(torch_matmul(A,V),A_t)
  }
  if(any_ineq){
    G_t = torch_transpose(G,2,3)
    EVG = torch_matmul(torch_matmul(E,V),G_t)
    GVG = torch_matmul(torch_matmul(G,V),G_t)
  }
  if(any_eq & any_ineq){
    AVG = torch_matmul(torch_matmul(A,V),G_t)
  }
  # --- assemble:
  if(any_eq & !any_ineq){
    top = torch_cat(list(EVE,EVA),dim = 3) #cbind(EVE,EVA)
    mid = torch_cat(list(torch_transpose(EVA,2,3),AVA),dim = 3) #cbind(t(EVA),AVA)
    P = torch_cat(list(top,mid),dim = 2)#rbind(top,mid)
  }
  else if(any_ineq & !any_eq){
    top = torch_cat(list(EVE,EVG),dim = 3) #cbind(EVE,EVG)
    bot = torch_cat(list(torch_transpose(EVG,2,3),GVG),dim = 3) #cbind(t(EVG),GVG)
    P = torch_cat(list(top,bot),dim = 2)#rbind(top,bot)
  }
  else if(any_eq & any_ineq){
    top = torch_cat(list(EVE,EVA,EVG),dim = 3)#cbind(EVE,EVA,EVG)
    mid = torch_cat(list(torch_transpose(EVA,2,3),AVA,AVG),dim = 3) #cbind(t(EVA),AVA,AVG)
    bot = torch_cat(list(torch_transpose(EVG,2,3),torch_transpose(AVG,2,3),GVG),dim = 3)#cbind(t(EVG),t(AVG),GVG)
    P = torch_cat(list(top,mid,bot),dim = 2)#rbind(top,mid,bot)
  }
  else{
    P = EVE
  }
  # --- stability:
  P = P + 10^-12 * torch_eye(n_z + n_eq + n_ineq)$unsqueeze(1)#10^-12

  # --- linear components:
  Vp = torch_matmul(V,p)#V%*%p
  obj = torch_matmul(E,Vp)#E%*%Vp
  if(any_eq){
    obj_eq = torch_matmul(A,Vp) + b
    obj = torch_cat(list(obj,obj_eq),dim=2)#c(obj, A%*%Vp + b)
  }
  if(any_ineq){
    obj_ineq = torch_matmul(G,Vp) + h
    obj = torch_cat(list(obj,obj_ineq),dim=2)#c(obj,G%*%Vp +h )
  }


  # --- solve dual:
  n_all = n_z + n_eq + n_ineq
  lb = -lambda_1*torch_ones(c(n_batch,n_z,1))
  ub = lambda_1*torch_ones(c(n_batch,n_z,1))
  if(any_eq){
    big = torch_ones(c(n_batch,n_eq,1))*10^8
    lb = torch_cat(list(lb,-big),dim=2)
    ub = torch_cat(list(ub,big),dim=2)
  }
  if(any_ineq){
    big = torch_ones(c(n_batch,n_ineq,1))*10^8
    zero = torch_zeros(c(n_batch,n_ineq,1))
    lb = torch_cat(list(lb,zero),dim=2)
    ub = torch_cat(list(ub,big),dim=2)
  }

  control = nn_qp_control(solver = 'admm',
                          output_as_list = FALSE,
                          tol_primal = 10^-3,
                          tol_dual = 10^-3,
                          max_iters = 1000,
                          backprop = 'fixed_point',
                          verbose=FALSE,
                          do_D_crossprod = FALSE,
                          unroll_grad = FALSE)
  control$rho = 1
  n_obs = get_n_obs_proxy(P)#  --- scaling:
  model = nn_qp(Q = P/n_obs,
                p = obj/n_obs,
                A = NULL,
                b = NULL,
                G = NULL,
                lb = lb,
                ub = ub,
                E = NULL,
                lambda_1 = NULL,
                control = control)
  sol = model()
  v = sol[,1:n_z,]
  rhs = - p - torch_matmul(E_t,v)# -p - E_t%*%v
  if(any_eq){
    nu = sol[, (n_z + 1):(n_z+n_eq), ]
    rhs = rhs - torch_matmul(A_t,nu)#rhs - A_t%*%nu
  }
  if(any_ineq){
    mu = sol[, (n_z + n_eq + 1):(n_all),]
    rhs = rhs - torch_matmul(G_t,mu)#rhs - t(G)%*%mu
  }

  # --- primal solution:
  z = torch_matmul(V,rhs) #V%*%rhs

  return(z)

}
