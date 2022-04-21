nn_lqp <- torch::nn_module(
  classname = "nn_lqp",
  # --- init:
  initialize = function(Q_model = nn_cov_constant(center = TRUE,normalize = FALSE),
                        p_model = nn_constant(value=NULL),
                        A_model = nn_constant(value=NULL),
                        b_model = nn_constant(value=NULL),
                        G_model = nn_constant(value=NULL),
                        h_model = nn_constant(value=NULL),
                        lb_model = nn_constant(value=NULL),
                        ub_model = nn_constant(value=NULL),
                        D_model = nn_constant(value=NULL),
                        E_model = nn_constant(value=NULL),
                        lambda_1_model = nn_constant(value=NULL),
                        lambda_2_model = nn_constant(value=NULL),
                        control = nn_osqp_control(),
                        x_index_list = list(Q = NULL,
                                            p = NULL,
                                            A = NULL,
                                            b = NULL,
                                            G = NULL,
                                            lb = NULL,
                                            ub = NULL,
                                            D = NULL,
                                            E = NULL,
                                            lambda_1 = NULL,
                                            lambda_2 = NULL),
                        ...)
  {
    # --- flag:
    cat("--- init qp learning model --- ",'\n')
    # --- models:

    # --- p model:
    self$p_model = p_model

    # --- Q model:
    self$Q_model = Q_model

    # --- A model:
    self$A_model = A_model

    # --- b model:
    self$b_model = b_model

    # --- G model:
    self$G_model = G_model

    # --- h model:
    self$h_model = h_model

    # --- lb model:
    self$lb_model = lb_model

    # --- ub model:
    self$ub_model = ub_model

    # --- D model:
    self$D_model = D_model

    # --- E model:
    self$E_model = E_model

    # --- lambda_1 model:
    self$lambda_1_model = lambda_1_model

    # --- lambda_2 model:
    self$lambda_2_model = lambda_2_model

    # --- control:
    self$control = control
    self$x_index_list = torch_make_index(x_index_list)
    #self$sol_index_list = torch_make_index(sol_index_list)

  },
  # --- forward:
  forward = function(x,
                     idx = NULL,
                     ...) {
    # --- invoke models:
    x_index_list = self$x_index_list

    if(!is.null(idx)){
      idx = torch_make_index(idx)
    }

    x_p = get_x(x,dim=2,index = x_index_list$p)
    p = self$p_model(x_p,...)
    p = get_sample(p, index = idx)

    x_Q = get_x(x,dim=2,index = x_index_list$Q)
    Q = self$Q_model(x_Q,...)
    Q = get_sample(Q, index = idx)

    x_A = get_x(x,dim=2,index = x_index_list$A)
    A = self$A_model(x_A,...)
    A = get_sample(A, index = idx)

    x_b = get_x(x,dim=2,index = x_index_list$b)
    b = self$b_model(x_b,...)
    b = get_sample(b, index = idx)

    x_G = get_x(x,dim=2,index = x_index_list$G)
    G = self$G_model(x_G,...)
    G = get_sample(G, index = idx)

    x_h = get_x(x,dim=2,index = x_index_list$G)
    h = self$h_model(x_h,...)
    h = get_sample(h, index = idx)

    x_lb = get_x(x,dim=2,index = x_index_list$lb)
    lb = self$lb_model(x_lb,...)
    lb = get_sample(lb, index = idx)

    x_ub = get_x(x,dim=2,index = x_index_list$ub)
    ub = self$ub_model(x_ub,...)
    ub = get_sample(ub, index = idx)

    x_D = get_x(x,dim=2,index = x_index_list$D)
    D = self$D_model(x_D,...)
    D = get_sample(D, index = idx)

    x_E = get_x(x,dim=2,index = x_index_list$E)
    E = self$E_model(x_E,...)
    E = get_sample(E, index = idx)

    x_lambda_1 = get_x(x,dim=2,index = x_index_list$lambda_1)
    lambda_1 = self$lambda_1_model(x_lambda_1,...)
    lambda_1 = get_sample(lambda_1, index = idx)

    x_lambda_2 = get_x(x,dim=2,index = x_index_list$lambda_2)
    lambda_2 = self$lambda_2_model(x_lambda_2,...)
    lambda_2 = get_sample(lambda_2, index = idx)

    # --- init qp solver if does not exist:
    if(is.null(self$qp_solver)){
      # --- init osqp model:
      self$qp_solver = nn_qp(Q = Q,
                             p = p,
                             A = A,
                             b = b,
                             G = G,
                             h = h,
                             lb = lb,
                             ub = ub,
                             D = D,
                             E = E,
                             lambda_1 = lambda_1,
                             lambda_2 = lambda_2,
                             control = self$control)

      if(FALSE){
      # --- update constants with prepped holders:
      update_constant(self$p_model,value = self$qp_solver$p)
      update_constant(self$Q_model,value = self$qp_solver$Q)
      update_constant(self$A_model,value = self$qp_solver$A)
      update_constant(self$b_model,value = self$qp_solver$b)
      update_constant(self$G_model,value = self$qp_solver$G)
      update_constant(self$lb_model,value = self$qp_solver$lb)
      update_constant(self$ub_model,value = self$qp_solver$ub)
      update_constant(self$D_model,value = self$qp_solver$D)
      update_constant(self$E_model,value = self$qp_solver$E)
      update_constant(self$lambda_1_model,value = self$qp_solver$lambda_1)
      update_constant(self$lambda_2_model,value = self$qp_solver$lambda_2)
      }

    }

    # --- invoke qp solver

    sol = self$qp_solver(Q = Q,
                         p = p,
                         A = A,
                         b = b,
                         G = G,
                         h = h,
                         lb = lb,
                         ub = ub,
                         D = D,
                         E = E,
                         lambda_1 = lambda_1,
                         lambda_2 = lambda_2,
                         control = self$control)


    return(sol)

  }
)
