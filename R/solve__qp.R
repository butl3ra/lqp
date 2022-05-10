#' @export
torch_solve_qp<-function(Q,
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
                         info = NULL,
                         solver_method = NULL,
                         sol_index_list = NULL,
                              ...)

{
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
  do_D_crossprod = control$do_D_crossprod
  solver = control$solver
  n_obs = control$n_obs
  n_obs = ifelse(is.null(n_obs),1,n_obs)
  unroll_grad = control$unroll_grad

  # --- prep torch tensor:
  Q =  prep_torch_tensor(Q)
  p =  prep_torch_tensor(p,target_dim = 3)
  A =  prep_torch_tensor(A)
  b =  prep_torch_tensor(b)
  G =  prep_torch_tensor(G)
  lb =  prep_torch_tensor(lb)
  ub =  prep_torch_tensor(ub)
  D =  prep_torch_tensor(D)
  E =  prep_torch_tensor(E)
  lambda_1 =  prep_torch_tensor(lambda_1)
  lambda_2 =  prep_torch_tensor(lambda_2)

  # --- prep variables:
  if(is.null(info)){
      info = get_qp_info(p = p,
                         A = A,
                         G = G,
                         lb = lb,
                         ub = ub,
                         D = D,
                         E = E,
                         lambda_1 = lambda_1,
                         lambda_2 = lambda_2)
  }
  # - n_x
  n_x = info$n_x

  # - equalities
  n_eq = info$n_eq
  any_eq = info$any_eq

  # - inequalities
  n_G = info$n_G
  any_G = info$any_G

  # - lb and ub
  n_lb = info$n_lb
  n_ub = info$n_ub
  any_lb = info$any_lb
  any_ub = info$any_ub
  n_ineq = info$n_ineq
  any_ineq = info$any_ineq

  # - l1
  n_E = info$n_E
  any_l_1 = info$any_l_1

  # - l2
  n_D = info$n_D
  any_l_2 = info$any_l_2

  # --- bound prep:
  lb = prep_bound(lb,n_x = n_x,default = -Inf)
  ub = prep_bound(ub,n_x = n_x,default = Inf)

  # --- n_obs scaling supported by nn_qp interface
  if(any_l_1){
    lambda_1 = n_obs*lambda_1
    if(n_E == 0){
      E = torch_eye(n_x)$unsqueeze(1)
      control$is_E_diag = TRUE
    }
  }
  if(any_l_2){
    lambda_2 = n_obs*lambda_2
    if(n_D == 0){
      D = torch_eye(n_x)$unsqueeze(1)
      control$do_D_crossprod = FALSE
    }
  }

  # --- create G if empty and lb and/or ub exist:
  if(!any_G & (any_lb | any_ub) ){
    G = torch_make_G_bound(n_x = n_x,
                           any_lb = any_lb,
                           any_ub = any_ub)
  }

  # --- get solver method: this won't change either
  if(is.null(solver_method)){
    solver_method = get_solver_method(any_eq = any_eq,
                                      any_ineq = any_ineq,
                                      any_l_1 = any_l_1,
                                      any_G = any_G,
                                      is_G_diag = is_G_diag,
                                      is_E_diag = is_E_diag,
                                      solver = solver)
  }

  # --- make sol_index_list: these values don't change
  if(is.null(sol_index_list)){
    sol_index_list = make_sol_index_list(method = solver_method,
                                         n_x = n_x,
                                         n_eq = n_eq,
                                         n_ineq = n_ineq,
                                         n_E = n_E,
                                         output_as_tensor = TRUE)
  }
  # --- formatting:
  dims = 3
  n_batch = get_n_batch(Q = Q,
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
                        dims = dims)
  max_batch = max(n_batch)#[c('Q','A','G','E')]

  # ---format appropriately based on solver dispatch:
  # --- nothing required for uncon
  if(solver_method == 'eqcon'){
    p = prep_batch_size(p,n_batch['b'])
    b = prep_batch_size(b,n_batch['p'])
    A = prep_batch_size(A,max_batch)
    Q = prep_batch_size(Q,max_batch)
  }
  else if(solver_method == 'int' | solver_method == 'quadprog'){
    Q = prep_batch_size(Q,max_batch)
    p = prep_batch_size(p,max_batch)
    A = prep_batch_size(A,max_batch)
    b = prep_batch_size(b,max_batch)
    G = prep_batch_size(G,max_batch)
    h = prep_batch_size(h,max_batch)
  }
  else if(solver_method == 'scs'){
    Q = prep_batch_size(Q,max_batch)
    p = prep_batch_size(p,max_batch)
    A = prep_batch_size(A,max_batch)
    b = prep_batch_size(b,max_batch)
    G = prep_batch_size(G,max_batch)
    h = prep_batch_size(h,max_batch)
  }
  else if(solver_method == 'admm'){
    Q = prep_batch_size(Q,max_batch)
    p = prep_batch_size(p,max_batch)
    A = prep_batch_size(A,max_batch)
    b = prep_batch_size(b,max_batch)
    G = prep_batch_size(G,max_batch)
    lb = prep_batch_size(lb,max_batch)
    ub = prep_batch_size(ub,max_batch)
  }
  else{
    Q = prep_batch_size(Q,max_batch)
    p = prep_batch_size(p,max_batch)
    A = prep_batch_size(A,max_batch)
    b = prep_batch_size(b,max_batch)
    G = prep_batch_size(G,max_batch)
    h = prep_batch_size(h,max_batch)
    lb = prep_batch_size(lb,max_batch)
    ub = prep_batch_size(ub,max_batch)
    E = prep_batch_size(E,max_batch)
    lambda_1 = prep_batch_size(lambda_1,max_batch)
  }

  # --- l2 regularization:
  if(any_l_2){
    if(do_D_crossprod){
      D = torch_crossprod(D)
    }
    Q = Q + lambda_2*D
  }

  # --- dispatching to appropriate solvers:
  #---- unconstrained problem
  if(solver_method == 'uncon'){
    sol =  torch_solve_qp_uncon(Q = Q,
                                p = p)
  }
  else if(solver_method == 'eqcon'){
    #----  equality constrained problem
    sol = torch_solve_qp_eqcon(Q = Q,
                               p = p,
                               A = A,
                               b = b)
  }
  else if(solver_method == 'uncon_l1'){
    #---- unconstrained with l1 regularization problem
    sol = torch_solve_qp_uncon_l1(Q = Q,
                                  p = p,
                                  E = E,
                                  lambda_1 = lambda_1,
                                  control = control)
  }
  else if(solver_method == 'con_l1'){
    #---- unconstrained with l1 regularization problem
    sol = torch_solve_qp_con_l1(Q = Q,
                                p = p,
                                A = A,
                                b = b,
                                G = G,
                                h = h,
                                lb = lb,
                                ub = ub,
                                E = E,
                                lambda_1 = lambda_1,
                                control = control)
  }
  else{
    if(solver_method == 'int' ){
      # --- interior point solver:
      if(unroll_grad){
        sol = torch_solve_qp_int( Q = Q,
                                  p = p,
                                  A = A,
                                  b = b,
                                  G = G,
                                  h = h,
                                  control = control,
                                  sol_index_list = sol_index_list,
                                  ...)
      }
      else{
        sol = nn_qp_int(Q = Q,
                        p = p,
                        A = A,
                        b = b,
                        G = G,
                        h = h,
                        control = control,
                        sol_index_list = sol_index_list,
                        ...)
      }
    }
    else if(solver_method == 'quadprog' ){
      # --- quadprog solver: warm starting not possible
      sol = nn_qp_quadprog(Q = Q,
                           p = p,
                           A = A,
                           b = b,
                           G = G,
                           h = h,
                           control = control,
                           sol_index_list = sol_index_list,
                           ...)
    }
    else if(solver_method == 'scs' ){
      # --- scs solver:
      sol = nn_qp_scs(Q = Q,
                      p = p,
                      A = A,
                      b = b,
                      G = G,
                      h = h,
                      control = control,
                      sol_index_list = sol_index_list,
                           ...)
    }
    else if(solver_method == 'admm' ){
      # --- admm box constraints and diagonal E
      if(unroll_grad){
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
                                  sol_index_list = sol_index_list,
                                  ...)
      }
      else{
        sol = nn_qp_admm(Q = Q,
                         p = p,
                         A = A,
                         b = b,
                         G = G,
                         lb = lb,
                         ub = ub,
                         E = E,
                         lambda_1 = lambda_1,
                         control = control,
                         sol_index_list = sol_index_list,
                         ...)
      }

    }
    else{
      # ---- general inequality constrained problem:
      if(unroll_grad){
        sol = torch_solve_qp_osqp(Q = Q,
                                  p = p,
                                  A = A,
                                  b = b,
                                  G = G,
                                  lb = lb,
                                  ub = ub,
                                  E = E,
                                  lambda_1 = lambda_1,
                                  control = control,
                                  sol_index_list = sol_index_list,
                                  ...)
      }
      else{
        sol = nn_qp_osqp(Q = Q,
                         p = p,
                         A = A,
                         b = b,
                         G = G,
                         lb = lb,
                         ub = ub,
                         E = E,
                         lambda_1 = lambda_1,
                         control = control,
                         sol_index_list = sol_index_list,
                         ...)
      }

    }
  }

  return(sol)

}

#' @export
torch_make_G_bound<-function(n_x,
                             any_lb,
                             any_ub,
                             requires_grad = FALSE)
{
  Id = torch_eye(n_x,requires_grad = requires_grad)$unsqueeze(1)
  if(any_lb & any_ub){
    G = torch_cat(list(-Id,Id),dim=2)
  }
  else if(any_lb){
    G = -Id
  }
  else if(any_ub){
    G = Id
  }
  return(G)
}


