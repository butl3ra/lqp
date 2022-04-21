nn_qp_admm<- torch::autograd_function(
  forward = function(ctx,
                     Q,
                     p,
                     A,
                     b,
                     G,
                     lb,
                     ub,
                     E,
                     lambda_1,
                     control = list(),
                     sol_index_list = NULL,
                     ...)
  {
    #######################################################################
    #Solve a batch of QPs.
    #The optimization problem for each instance in the batch
    #   x_star =   argmin_x 1/2 x^T Q x + p^T x + lambda_1 ||Ex||_1
    #             subject to Gx <= h
    #                        Ax  = b
    # where
    # Q:  A (n_batch, n_x, n_x) Tensor.
    # p:  A (n_batch, n_x, 1) Tensor.
    # G:  A (n_batch, n_ineq, n_x) Tensor.
    # h:  A (n_batch, n_ineq, 1) Tensor.
    # A:  A (n_batch, n_eq, n_x)  Tensor.
    # b : A (n_batch, n_eq, 1) Tensor.
    # lambda_1: A(n_batch, 1, 1) Tensor.
    # E: A (n_batch, n_x, n_x) PSD Tensor.
    # Returns: x_star: a (n_batch, n_z) Tensor.
    #######################################################################

    # --- start with no grad
    #with_no_grad({
    # --- invokes forward torch_solve_qp_admm:
    sol = torch_solve_qp_admm(Q = Q,
                              p = p,
                              A = A,
                              b = b,
                              G = G,
                              lb = lb,
                              ub = ub,
                              E = E,
                              lambda_1 = lambda_1,
                              control = control,
                              ...)
    # --- slice:
    output_as_list = control$output_as_list
    if(is.null(sol_index_list) & !output_as_list){
      # --- prep:
      n_x = get_n_x(p)
      n_eq = get_ncon(A)
      n_ineq = n_x#always true for admm
      sol_index_list = make_sol_index_list(method = 'admm',
                                           n_x = n_x,
                                           n_eq = n_eq,
                                           n_ineq = n_ineq)
    }
    # --- qp_prep_sol output
    lst = qp_prep_sol(sol = sol,
                      output_as_list = output_as_list,
                      sol_index_list = sol_index_list)
    sol = lst$sol
    x = lst$x

    # --- prepping:
    sol$mat_data = sol$mat_data$detach()

    sol$mat_pivots = change_tensor_dtype(sol$mat_pivots,dtype = torch_int())

    # --- caching for backward pass

    ctx$save_for_backward(Q = Q,
                          p = p,
                          A = A,
                          b = b,
                          G = G,
                          lb = lb,
                          ub = ub,
                          E = E,
                          lambda_1 = lambda_1,
                          x = sol$x,
                          u = sol$u,
                          lams = sol$lams,
                          nus = sol$nus,
                          mat_inv = sol$mat_inv,
                          mat_data = sol$mat_data$detach(),
                          mat_pivots = sol$mat_pivots$detach(),
                          output_as_list = output_as_list,
                          sol_index_list  = sol_index_list,
                          control = control)


    #})
    # --- end with no grad
    return(x)

  },

  backward = function(ctx,
                      dl_dx)
  {
    # --- start with no grad:
    #with_no_grad({
    # --- unpack saved variables
    saved_vars = ctx$saved_variables
    Q = saved_vars$Q
    p = saved_vars$p
    A = saved_vars$A
    b = saved_vars$b
    G = saved_vars$G
    lb = saved_vars$lb
    ub = saved_vars$ub
    E = saved_vars$E
    lambda_1 = saved_vars$lambda_1
    mat_inv = saved_vars$mat_inv
    #mat_inv = torch_reshape_mat(mat_inv,forward = FALSE)
    mat_data = saved_vars$mat_data
    mat_pivots = saved_vars$mat_pivots
    mat_data = torch_reshape_mat(mat_data,forward = FALSE)
    mat_pivots = mat_pivots$squeeze(3)



    x = saved_vars$x
    u = saved_vars$u
    lams = saved_vars$lams
    nus = saved_vars$nus
    slacks = saved_vars$slacks# likely null

    output_as_list  = saved_vars$output_as_list
    sol_index_list = saved_vars$sol_index_list
    control = saved_vars$control
    rho = control$rho
    backprop = control$backprop

    # --- prep dl_dx
    if(!output_as_list){
      dl_dx = get_x(dl_dx,index = sol_index_list$x)
    }

    # --- differentiate through kkt
    if(backprop == 'kkt'){
      # --- prep:
      any_A = get_any(A)
      any_G = get_any(G)
      any_lb = as.logical(torch_max(lb) > -Inf)
      any_ub = as.logical(torch_min(ub) < Inf)
      any_l_1 = get_any(lambda_1, threshold = 0)
      any_ineq = any_lb | any_ub
      if(any_ineq){
        h = torch_bounds_to_h(lb = lb,
                              ub = ub,
                              any_lb = any_lb,
                              any_ub = any_ub)

        slacks = h - torch_matmul(G,x)
        slacks = torch_threshold_(slacks,10^-8,10^-8)
      }
      # --- compute prep
      A_size = get_size(A)
      n_eq = A_size[2]
      any_eq = n_eq > 0

      G_size = get_size(G)
      n_ineq = G_size[2]
      n_batch = G_size[1]

      # --- make inversion matrix:
      sol_mats = torch_qp_make_sol_mat(Q = Q,
                                       G = G,
                                       A = A,
                                       lams = lams,
                                       nus = nus,
                                       slacks = slacks,
                                       E = E,
                                       lambda_1 = lambda_1,
                                       x = x)

      # --- Compute differentials:
      diff_list = torch_solve_qp_backwards(dl_dx = dl_dx,
                                           sol_mats = sol_mats,
                                           n_eq = n_eq,
                                           n_ineq = n_ineq)

      # --- compute gradients
      grads = torch_qp_int_grads_admm(x = x,
                                      lams = lams,
                                      nus = nus,
                                      E = E,
                                      lambda_1 = lambda_1,
                                      d_x = diff_list$d_x,
                                      d_lam =  diff_list$d_lam,
                                      d_nu =  diff_list$d_nu,
                                      n_eq = n_eq,
                                      n_ineq = n_ineq,
                                      any_lb = any_lb,
                                      any_ub = any_ub,
                                      any_l_1 = any_l_1,
                                      Q_requires_grad = check_requires_grad(Q),
                                      A_requires_grad = check_requires_grad(A))
    }
    else{
      # --- compute grads through fixed point:
      grads = torch_qp_grad_admm(x = x,
                                 dl_dz = dl_dx,
                                 u = u,
                                 nus = nus,
                                 lams = lams,
                                 mat_inv = mat_inv,
                                 mat_data = mat_data,
                                 mat_pivots = mat_pivots,
                                 A = A,
                                 lb = lb,
                                 ub = ub,
                                 E  = E,
                                 lambda_1 = lambda_1,
                                 rho = rho,
                                 Q = Q)
    }




    #})
    # --- end with no grad
    return(grads)




    return(grads)



  }

)
