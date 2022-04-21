nn_qp <- torch::nn_module(
  classname = "nn_qp",
  # --- initialize:
  initialize = function(Q,
                        p,
                        A = NULL,
                        b = NULL,
                        G = NULL,
                        h = NULL,
                        lb = NULL,
                        ub = NULL,
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
    # Q:        A (n_batch,n_x,n_x) tensor
    # p:        A (n_batch,n_x,1) tensor
    # A:        A (n_batch,n_eq, n_x) tensor
    # b:        A (n_batch,n_eq,1) tensor
    # G:        A (n_batch,n_ineq, n_x) tensor
    # lb:       A (n_batch,n_ineq,1) tensor
    # ub:       A (n_batch,n_ineq,1) tensor
    # D:        A (n_batch,n_x,n_x) tensor
    # E:        A (n_batch,n_x,n_x) tensor
    # lambda_1: A (n_batch,1,1) tensor
    # lambda_2: A (n_batch,1,1) tensor
    # Returns: x_star:  A (n_batch,n_x,1) tensor
    #######################################################################

    # --- control:
    control = do.call(nn_qp_control,control)
    is_G_diag = control$is_G_diag
    is_E_diag = control$is_E_diag
    solver = control$solver

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

    # --- n_obs scaling: for l1 and l2
    control$n_obs = get_n_obs_proxy(Q = Q, p = p)

    # --- get general QP info
    info = get_qp_info(p = p,
                       A = A,
                       G = G,
                       lb = lb,
                       ub = ub,
                       E = E,
                       D = D,
                       lambda_1 = lambda_1,
                       lambda_2 = lambda_2)

    # --- prep for D: defaulting to identity
    if(info$any_l_2 & info$n_D == 0 ){
      D = torch_eye(info$n_x)$unsqueeze(1)
      control$do_D_crossprod = FALSE
      info$n_D = info$n_x
    }
    # --- prep for E: defaulting to identity
    if(info$any_l_1 & info$n_E == 0){
      E = torch_eye(info$n_x)$unsqueeze(1)
      control$is_E_diag = TRUE
      info$n_E = info$n_x
    }

    # --- get solver method
    solver_method = get_solver_method(any_eq = info$any_eq,
                                     any_ineq = info$any_ineq,
                                     any_l_1 = info$any_l_1,
                                     any_G = info$any_G,
                                     is_G_diag = is_G_diag,
                                     is_E_diag = is_E_diag,
                                     solver = solver)
    # --- uncon solver
    if(solver_method == 'uncon'){
      control$output_as_list = TRUE
    }
    # --- eqcon solver
    if(solver_method == 'eqcon'){
      control$output_as_list = FALSE
    }

    # --- sol_index_list:
    sol_index_list = make_sol_index_list(method = solver_method,
                                         n_x = info$n_x,
                                         n_eq = info$n_eq,
                                         n_ineq = info$n_ineq,
                                         n_E = info$n_E,
                                         output_as_tensor = TRUE,
                                         unroll_grad = control$unroll_grad)

    # --- check n-params vs statics:
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


    # --- note: all other variables prepped inside solve_qp at runtime

    # --- cache all variables in-case they are not supplied in forward execution
    self$Q = Q
    self$p = p
    self$A = A
    self$b = b
    self$G = G
    self$h = h
    self$lb = lb
    self$ub = ub
    self$D = D
    self$E = E
    self$lambda_1 = lambda_1
    self$lambda_2 = lambda_2
    self$control = control
    self$info = info
    self$solver_method = solver_method
    self$sol_index_list = sol_index_list
    self$is_param  = is_param
    self$iter = 0

  },
  # --- main forward:
  forward = function(Q = NULL,
                     p = NULL,
                     A = NULL,
                     b = NULL,
                     G = NULL,
                     h = NULL,
                     lb = NULL,
                     ub = NULL,
                     D = NULL,
                     E = NULL,
                     lambda_1 = NULL,
                     lambda_2 = NULL,
                     rerun = getOption('rerun'),
                     ...) {

    # --- iteration count upate:
    self$iter = self$iter + 1
    hybrid = self$control$hybrid
    hybrid_iter = self$control$hybrid_iter
    if(hybrid & self$iter >= hybrid_iter ){
      if(self$iter == hybrid_iter){
        cat('changing solver to: ',self$control$hybrid_solver, '\n')
        self$control$solver  =  self$control$hybrid_solver
        self$solver_method = self$control$hybrid_solver
        self$control$max_iters = self$control$hybrid_max_iters
        if(self$control$hybrid_solver == 'int'){
          self$control$unroll_grad = FALSE#default behaviour
        }


        self$sol_index_list = make_sol_index_list(method = self$solver_method,
                                                  n_x = self$info$n_x,
                                                  n_eq = self$info$n_eq,
                                                  n_ineq = self$info$n_ineq,
                                                  n_E = self$info$n_E,
                                                  output_as_tensor = TRUE,
                                                  unroll_grad = self$control$unroll_grad)

      }

    }

    # --- grab control:
    #control = self$control
    #if(self$iter == 1 & control$max_iters == 1){
    #  control$max_iters = 100
    #}

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
    # --- h:
    if(is.null(h)){
      h = self$h
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

    # --- cache list: warm starts
    cache_list = list()
    if(!rerun){
      cache_list$x = self$x
      cache_list$y = self$y
      cache_list$z = self$z
      cache_list$u = self$u
      cache_list$lams = self$lams
      cache_list$nus = self$nus
      cache_list$slacks = self$slacks
      cache_list$mat_inv = self$mat_inv
      cache_list$mat_data = self$mat_data
      cache_list$mat_pivots = self$mat_pivots
      cache_list$U_Q = self$U_Q
      cache_list$U_S = self$U_S
      cache_list$R = self$R
    }

    # --- main solver:
    sol = torch_solve_qp(Q = Q,
                         p = p,
                         A = A,
                         b = b,
                         G = G,
                         h = h,
                         lb = lb,
                         ub = ub,
                         E = E,
                         D = D,
                         lambda_1 = lambda_1,
                         lambda_2 = lambda_2,
                         control = self$control,
                         info = self$info,
                         solver_method = self$solver_method,
                         sol_index_list = self$sol_index_list,
                         x = cache_list$x,
                         y = cache_list$y,
                         z = cache_list$z,
                         u = cache_list$u,
                         lams = cache_list$lams,
                         nus = cache_list$nus,
                         slacks = cache_list$slacks,
                         mat_data = cache_list$mat_data,
                         mat_pivots = cache_list$mat_pivots,
                         mat_inv = cache_list$mat_inv,
                         U_Q = cache_list$U_Q,
                         U_S = cache_list$U_S,
                         R = cache_list$R
                         )

    # --- slice and caching solutions to self
    if(!self$control$output_as_list){
      # --- slice
      sol_list = torch_tensor_to_list(x = sol,
                                      index_list = self$sol_index_list,
                                      dim = 2)
      x = sol_list$x
      if(self$solver_method %in% c('quadprog','int','admm','osqp')){

        # --- caching primal-dual solution without grad
        self$x = detach_grad(sol_list$x)
        self$y = detach_grad(sol_list$y)
        self$z = detach_grad(sol_list$z)
        self$u = detach_grad(sol_list$u)
        self$lams = detach_grad(sol_list$lams)
        self$nus = detach_grad(sol_list$nus)
        self$slacks = detach_grad(sol_list$slacks)

        # --- cache matrix factorizations where appropriate:
        mat_list = cache_matrix(sol_list = sol_list,
                                solver_method = self$solver_method,
                                is_param = self$is_param,
                                info = self$info,
                                unroll_grad = self$control$unroll_grad)

        self$mat_data = mat_list$mat_data
        self$mat_pivots = mat_list$mat_pivots
        self$mat_inv = mat_list$mat_inv
        self$U_Q = mat_list$U_Q
        self$U_S = mat_list$U_S
        self$R = mat_list$R
      }
    }
    else{
      x = sol
    }

    return(x)

  }
)
