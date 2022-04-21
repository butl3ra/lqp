#' @export
torch_solve_qp_osqp<-function(Q,
                              p,
                              A = NULL,
                              b = NULL,
                              G = NULL,
                              lb = NULL,
                              ub = NULL,
                              E = NULL,
                              lambda_1 = NULL,
                              control,
                              x = NULL,
                              z = NULL,
                              y = NULL,
                              mat_inv = NULL,
                              ...)
{
  # --- unpacking control:
  rho = control$rho
  rho_eq_scale = control$rho_eq_scale
  sigma = control$sigma
  alpha = control$alpha
  tol_primal = control$tol_primal
  tol_dual = control$tol_dual
  verbose = control$verbose
  max_iters = control$max_iters
  tol_method  = control$tol_method
  alpha2 = 1 - alpha
  output_as_list = control$output_as_list
  warm_start = control$warm_start
  lb_default = control$lb_default
  ub_default = control$ub_default

  # --- check for warm start
  warm_start_vars = check_warm_start(x = x,
                                     y = y,
                                     z = z)
  warm_start_vars = warm_start_vars & warm_start

  warm_start_mats = !is.null(mat_inv)
  warm_start_mats = warm_start_mats & warm_start

  # --- prep:
  lb = torch_clamp(lb,min = lb_default)
  ub = torch_clamp(ub, max = ub_default)
  x_size = get_size(p)
  n_x = x_size[2]
  n_batch = x_size[1]
  idx_x = 1:n_x

  any_A = get_any(A)
  A_size = get_size(A)
  n_eq = A_size[2]

  any_G = get_any(G)
  G_size = get_size(G)
  n_ineq = G_size[2]
  any_lb = as.logical(torch_max(lb[,1,]) > -Inf)
  any_ub = as.logical(torch_min(ub[,1,]) < Inf)
  any_ineq = any_lb | any_ub
  n_con = n_eq + n_ineq

  any_l_1 = get_any(lambda_1, threshold = 0)

  # --- create A bar matrices:
  if(any_A){
    A_bar_A = A
    lb_bar_A = b
    ub_bar_A = b
    rho_bar_A = rep(rho*rho_eq_scale,n_eq)
  }
  else{
    A_bar_A = lb_bar_A = ub_bar_A  = rho_bar_A = NULL
  }
  if(any_G){
    A_bar_G = G
    lb_bar_G = lb
    ub_bar_G = ub
    rho_bar_G = rep(rho,n_ineq)
  }
  else{
    A_bar_G = lb_bar_G = ub_bar_G  = rho_bar_G = NULL
  }
  if(any_l_1){
    A_bar_E = E
    E_size = get_size(E)
    n_E = E_size[2]
    lb_bar_E = torch_ones(E_size[1],n_E,1)*(-Inf)
    ub_bar_E = torch_ones(E_size[1],n_E,1)*(Inf)
    rho_bar_E = rep(rho,n_E)

    # --- soft-threshold values:
    one = torch_ones(c(E_size[1],E_size[2],1))
    if(n_con > 0){
      zero = torch_zeros(E_size[1],n_con,1)
      lambda_1_E = lambda_1 * torch_cat(list(zero,one),2)
    }
    else{
      lambda_1_E = lambda_1 * one
    }
    lambda_1_E_rho = lambda_1_E/rho

  }
  else{
    A_bar_E = lb_bar_E = ub_bar_E  = rho_bar_E = NULL
  }
  # --- consolidate tensors
  A_bar = torch_cat_list(list(A_bar_A,A_bar_G,A_bar_E),dim=2)
  lb_bar = torch_cat_list(list(lb_bar_A,lb_bar_G,lb_bar_E),dim=2)
  ub_bar = torch_cat_list(list(ub_bar_A,ub_bar_G,ub_bar_E),dim=2)
  rho_bar = c(rho_bar_A,rho_bar_G, rho_bar_E)

  # --- A_bar:
  A_bar_size = get_size(A_bar)
  A_bar_t = torch_transpose(A_bar,2,3)

  # --- rho bar is technically the action of matrix:
  rho_bar = prep_torch_tensor(rho_bar)
  rho_bar_inv = 1/rho_bar
  n_con = A_bar_size[2]
  z_size = c(x_size[1],n_con,x_size[3])
  idx_con = n_x + (1:n_con)

  # --- factorize and cache inverse if not supplied
  if(!warm_start_mats){
    Id = torch_eye(n_x)$unsqueeze(1)
    Q_I = Q + sigma*Id
    bottom_right = torch_diag_embed(-rho_bar_inv[,,1])
    bottom_right = torch_rep(bottom_right,c(A_bar_size[1],1,1))

    mat = torch_qp_eqcon_mat(Q = Q_I, A = A_bar,bottom_right = bottom_right)
    mat_inv = linalg_inv(mat)
  }

  # --- initialize x, z and u
  if(!warm_start_vars){
    x  =  torch_zeros(x_size)
    z  =  torch_zeros(z_size)
    y = torch_zeros(z_size)
  }

  # --- main loop
  iters = 1:max_iters
  for(iter in iters){

    # --- projection to sub-space:
    rhs_u = sigma * x - p
    rhs_l = z - rho_bar_inv*y
    rhs = torch_cat(list(rhs_u,rhs_l),2)

    xv = torch_matmul(mat_inv,rhs)
    x_tilde = xv[,idx_x,]
    v = xv[,idx_con,]
    z_tilde = z + rho_bar_inv*(v - y)

    # --- proximal projection:
    z_prev = z
    x = alpha*x_tilde + alpha2*x
    z = alpha*z_tilde + alpha2*z + rho_bar_inv*y

    if(any_l_1){
      z = torch_soft_threshold(z, lambda_1_E_rho)
    }
    if(any_ineq){
      z = torch_proj_box(z,
                         lb = lb_bar,
                         ub = ub_bar,
                         any_lb = any_lb,
                         any_ub = any_ub)
    }

    # --- dual update:
    y = y + rho_bar*(alpha*z_tilde + alpha2*z_prev - z)

    # --- update resdiuals
    r_primal = torch_matmul(A_bar,x) - z
    r_dual = torch_matmul(Q,x) + p + torch_matmul(A_bar_t,y)



    # ---  primal and dual errors:
    primal_error = torch_norm(r_primal,dim=2,keepdim=T)/n_x
    dual_error = torch_norm(r_dual,dim=2,keepdim=T)/n_x
    if(tol_method == 'max'){
      primal_error = torch_max(primal_error)
      dual_error = torch_max(dual_error)
    }
    else{
      primal_error = torch_mean(primal_error)
      dual_error = torch_mean(dual_error)
    }


    # --- verbose
    if(verbose){
      cat('iteration: ', iter, '\n')
      cat('|| primal_error||_2 = ', as.numeric(primal_error),'\n')
      cat('|| dual_error||_2 = ', as.numeric(dual_error),'\n')
    }

    do_stop = as.logical(primal_error < tol_primal) & as.logical(dual_error < tol_dual)
    if(do_stop){
      break
    }

  }

  # --- maybe output A_bar as well ...

  # --- extract the dual variables:
  nus = NULL
  if(any_A){
    idx = 1:n_eq
    nus = y[,idx,,drop=F]
  }
  lams = NULL
  if(any_G){
    idx = (n_eq+1):(n_eq + n_ineq)
    lams = y[,idx,,drop=F]
    lams_neg = torch_threshold_(-lams,0,0)
    lams_pos = torch_threshold_(lams,0,0)

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
  }

  # --- concatenate if not output as list...
  if(!output_as_list){
    mat_inv_reshape = torch_reshape_mat(mat_inv,forward = TRUE)
    out = list(x = x,
               z = z,
               y = y,
               lams = lams,
               nus = nus,
               mat_inv = mat_inv_reshape)
    out = torch_cat(out,dim = 2)
  }
  else{
    out = list(x = x,
               z = z,
               y = y,
               lams = lams,
               nus = nus,
               mat_inv = mat_inv)
  }

  return(out)



}
