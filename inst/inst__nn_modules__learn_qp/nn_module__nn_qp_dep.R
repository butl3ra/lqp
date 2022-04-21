nn_qp_dep <- torch::nn_module(
  classname = "nn_qp",
  # --- initialize:
  initialize = function(Q,
                        p,
                        A = NULL,
                        b = NULL,
                        G = NULL,
                        lb = -Inf,
                        ub = Inf,
                        D = NULL,
                        E = NULL,
                        lambda_1 = NULL,
                        lambda_2 = NULL,
                        control = nn_qp_control(),
                        ...) {
    #######################################################################
    #Solve a QP of the form:
    #   x_star =   argmin_x 0.5*x^TQx + p^Tx + lambda_2/2*||D^0.5 x|| + lambda_1*||Ex||
    #             subject to Ax =  b
    #                        lb <= Gx  <= ub
    # Q:  A (n_batch,n_x,n_x) or (n_x,n_x) SPD matrix
    # p:  A (n_batch,n_x,1) matrix or (n_x) matrix
    # A:  A (n_batch,n_eq, n_x) matrix or (n_eq,n_x)
    # b:  A (n_batch,n_eq,1) or (n_x) matrix
    # G:  A (n_batch,n_ineq, n_x) matrix
    # lb: A (n_batch,n_ineq,1) matrix
    # ub: A (n_batch,n_ineq,1) matrix
    # Returns: x_star:  A (n_x) vector
    #######################################################################

    # --- control:
    control = do.call(nn_qp_control,control)
    is_E_diag = control$is_E_diag
    is_G_diag = control$is_G_diag

    # --- prep minimum requirements:
    any_Q = get_any(Q)
    any_p = get_any(p)
    if(!any_Q & !any_p){
      stop('minimum requirement: must have at least Q or p')
    }
    else if(!any_Q){
      p = prep_torch_tensor(p,target_dim=3)
      n_x = ncol(p)
      Q = torch_zeros(1,n_x,n_x)
    }
    else if(!any_p){
      Q = prep_torch_tensor(Q)
      n_x = ncol(Q)
      p = torch_zeros(1,n_x,1)
    }
    else{
      p = prep_torch_tensor(p,target_dim=3)
      Q = prep_torch_tensor(Q)
    }
    null_size = c(0,0,0)
    p_size = get_size(p,null_size)
    Q_size = get_size(Q,null_size)
    n_x = p_size[2]


    # --- prep remaining variables:
    A = prep_torch_tensor(A)
    A_size = get_size(A,null_size)
    any_A = get_any(A)
    b = prep_torch_tensor(b)
    b_size = get_size(b,null_size)
    G = prep_torch_tensor(G)
    G_size = get_size(G,null_size)
    any_G = get_any(G)
    n_G = max(n_x,G_size[2])
    lb = prep_bound(x = lb,
                    n_x = n_G,
                    default = -Inf)
    lb_size = get_size(lb,null_size)
    ub = prep_bound(x = ub,
                    n_x = n_G,
                    default = Inf)
    ub_size = get_size(ub,null_size)

    # --- prep D and E:
    any_l_1 = get_any(lambda_1, threshold = 0)
    any_l_2 = get_any(lambda_2, threshold = 0)

    # --- NOTE:: we are NOT scaling by n_obs here...this differs from optimr implementation
    if(any_l_1){
      if(is.null(E)){
        E = torch_eye(n_x)$unsqueeze(1)
        is_E_diag = TRUE
        control$is_E_diag = is_E_diag
      }
      else{
        E = prep_torch_tensor(E)
      }
    }
    if(any_l_2){
      if(is.null(D)){
        D = torch_eye(n_x)$unsqueeze(1)
      }
      else{
        D = prep_torch_tensor(D)
      }
    }

    # --- n_batch
    dims = c(Q = 3,p = 3,A = 3,b = 3,G = 3,D = 3, E = 3,lb = 3, ub = 3)
    n_batch = get_n_batch(Q = Q, p = p, G = G, A = A, b = b,D = D, E = E,lb = lb,ub = ub, dims = dims)

    # --- format p and b
    if((is_G_diag & is_E_diag) | (!any_G & !any_l_1)){
      p = prep_batch_size(p,n_batch['b'])
      b = prep_batch_size(b,n_batch['p'])
    }

    # --- format Q and A
    if(is_G_diag & is_E_diag){
      max_batch = max(n_batch[c('Q','A')])
      Q = prep_batch_size(Q,max_batch)
      A = prep_batch_size(A,max_batch)
    }
    else{
      max_batch = max(n_batch[c('Q','A','G','E')])
      Q = prep_batch_size(Q,max_batch)
      A = prep_batch_size(A,max_batch)
      G = prep_batch_size(G,max_batch)
      E = prep_batch_size(Q,max_batch)
    }

    # --- format all non-params -- statics:
    is_param = check_nn_parameter(Q = Q,
                                  p = p,
                                  A = A,
                                  b = b,
                                  G = G,
                                  lb = lb,
                                  ub = ub,
                                  D = D,
                                  E = E,
                                  lambda_1 = lambda_1,
                                  lambda_2 = lambda_2)

    # --- cache mat inverse: TODO
    mat_inv = NULL


    # --- n_obs
    control$n_obs = get_n_obs_proxy(Q = Q, p = p)

    # --- cache all 'static' variables
    self$Q = Q
    self$p = p
    self$A = A
    self$b = b
    self$G = G
    self$lb = lb
    self$ub = ub
    self$D = D
    self$E = E
    self$lambda_1 = lambda_1
    self$lambda_2 = lambda_2
    self$control = control
    if(F){
      self$info = list(is_param = is_param,
                       format_spec = list(n_batch_dims = dims,
                                          n_batch = n_batch,
                                          any_A = any_A,
                                          any_G = any_G,
                                          is_G_diag  = is_G_diag ,
                                          is_E_diag = is_E_diag,
                                          any_l_1 = any_l_1,
                                          any_l_2 = any_l_2
                       ))
    }


  },
  # --- main forward:
  forward = function(Q = NULL,
                     p = NULL,
                     A = NULL,
                     b = NULL,
                     G = NULL,
                     lb = NULL,
                     ub = NULL,
                     D = NULL,
                     E = NULL,
                     lambda_1 = NULL,
                     lambda_2 = NULL,
                     ...) {
    # --- Q:
    if(is.null(Q)){
      Q = self$Q
    }
    # --- p:
    if(is.null(p)){
      p = self$p
    }
    # --- A:
    if(is.null(A)){
      A = self$A
    }
    # --- b:
    if(is.null(b)){
      b = self$b
    }
    # --- G:
    if(is.null(G)){
      G = self$G
    }
    # --- lb:
    if(is.null(lb)){
      lb = self$lb
    }
    # --- ub:
    if(is.null(ub)){
      ub = self$ub
    }
    # --- D:
    if(is.null(D)){
      D = self$D
    }
    # --- E:
    if(is.null(E)){
      E = self$E
    }
    # --- lambda_1:
    if(is.null(lambda_1)){
      lambda_1 = self$lambda_1
    }
    # --- lambda_2:
    if(is.null(lambda_2)){
      lambda_2 = self$lambda_2
    }

    # --- main solver:
    sol = torch_solve_qp(Q = Q,
                         p = p,
                         A = A,
                         b = b,
                         G = G,
                         lb = lb,
                         ub = ub,
                         E = E,
                         D = D,
                         lambda_1 = lambda_1,
                         lambda_2 = lambda_2,
                         control = self$control)

    # --- slice and caching solutions:


    return(x_z)
  }
)
