nn_qp_int<- torch::autograd_function(
  forward = function(ctx,
                     Q,
                     p,
                     A,
                     b,
                     G,
                     h,
                     control = nn_qp_control(solver = 'int',
                                             max_iters = 10,
                                             tol = 10^-4),
                     sol_index_list = NULL,
                     ...)
  {
    #######################################################################
    #Solve a batch of QPs.
    #The optimization problem for each instance in the batch
    #   x_star =   argmin_z 1/2 z^T Q z + p^T z
    #             subject to Gz <= h
    #             Az  = b
    # where
    # Q:  A (n_batch, n_z, n_z) or (n_z, n_z) Tensor.
    # p:  A (n_batch, n_z) or (n_z) Tensor.
    # G:  A (n_batch, n_ineq, n_z) or (n_ineq, n_z) Tensor.
    # h:  A (n_batch, n_ineq) or (n_ineq) Tensor.
    # A:  A (n_batch, n_eq, n_z) or (n_eq, n_z) Tensor.
    # b : A (n_batch, n_eq) or (n_eq) Tensor.
    # Returns: x_star: a (n_batch, n_z) Tensor.
    #######################################################################

    # --- start with no grad
    #with_no_grad({

    # --- invokes forward torch_solve_qp_int:
    sol = torch_solve_qp_int(Q = Q,
                             p = p,
                             A = A,
                             b = b,
                             G = G,
                             h = h,
                             control = control,
                             ...)


    # --- slice spec:
    output_as_list = control$output_as_list
    if(is.null(sol_index_list) & !output_as_list){
      # --- prep:
      n_x = get_size(p)[2]
      n_eq = get_ncon(A)
      n_ineq = get_ncon(G)
      sol_index_list = make_sol_index_list(method = 'int',
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
    # --- caching for backward pass
    ctx$save_for_backward(Q = Q,
                          p = p,
                          G = G,
                          h = h,
                          A = A,
                          b = b,
                          x =sol$x,
                          lams = sol$lams,
                          slacks = sol$slacks,
                          nus = sol$nus,
                          U_Q = sol$U_Q,
                          U_S = sol$U_S,
                          output_as_list = output_as_list,
                          sol_index_list  = sol_index_list )
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
    G = saved_vars$G
    h = saved_vars$h
    A = saved_vars$A
    b = saved_vars$b
    U_Q = saved_vars$U_Q
    U_S = saved_vars$U_S

    x = saved_vars$x
    lams = saved_vars$lams
    slacks = saved_vars$slacks
    nus = saved_vars$nus

    # --- numerical stability
    lams = torch_clamp(lams, 10^-8)
    slacks = torch_clamp(slacks,10^-8)

    output_as_list  = saved_vars$output_as_list
    sol_index_list = saved_vars$sol_index_list

    # --- compute prep
    n_x = get_n_x(x)

    A_size = get_size(A)
    n_eq = A_size[2]
    any_eq = n_eq > 0

    G_size = get_size(G)
    n_ineq = G_size[2]
    n_batch = G_size[1]

    n_con = n_ineq + n_eq

    # --- prep if output is concat tensors
    if(!output_as_list){
      dl_dx = get_x(dl_dx,index = sol_index_list$x)
      U_Q = torch_reshape_mat(mat = U_Q,forward = FALSE)
      U_S = torch_reshape_mat(mat = U_S,forward = FALSE)
    }

    rx = dl_dx#$unsqueeze(3)
    rs = rz = torch_zeros(c(n_batch,n_ineq,1))

    d = lams / slacks
    GT = torch_transpose(G,2,3)
    AT = NULL
    if(any_eq){
      AT = torch_transpose(A,2,3)
      ry = torch_zeros(c(n_batch,n_eq,1))
    }

    # --- Compute differentials:
    diff_list = torch_qp_int_solve_kkt(U_Q = U_Q,
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
    # --- As per Amos page 5, diff_list$dz = d_lam_tilde = D(lambdas)d_lambda
    # --- Correcting for that here which is then re-applied in torch_qp_int_grads:
    diff_list$dz = diff_list$dz/lams
    # --- compute gradients
    grads = torch_qp_int_grads(x = x,
                               lams = lams,
                               nus = nus,
                               d_x = diff_list$dx,
                               d_lam =  diff_list$dz,
                               d_nu =  diff_list$dy,
                               n_eq = n_eq,
                               n_ineq = n_ineq,
                               Q_requires_grad = check_requires_grad(Q),
                               A_requires_grad = check_requires_grad(A),
                               G_requires_grad = check_requires_grad(G))
    #})
    # --- end with no grad
    return(grads)



  }

)
