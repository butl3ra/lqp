#' @export
make_sol_index_list<-function(method,
                              n_x,
                              n_eq,
                              n_ineq,
                              n_E = NULL,
                              output_as_tensor = TRUE,
                              unroll_grad = FALSE,
                              ...)
{
  method = tolower(method)
  n_con = n_eq + n_ineq
  if(method == 'uncon' | method == 'uncon_l1' | method == 'con_l1'){
    sol_index_list = list(x = 1:n_x)

  }
  else if(method == 'eqcon'){
    sol_index_list = list(x = 1:n_x,
                          nu = (n_x+1):(n_x + n_eq))

  }
  else if(method == 'int' | method == 'quadprog' ){
    idx = 1:n_x
    idx_lams = 1:n_ineq
    n_x2 = n_x^2
    n_ineq2 = n_ineq^2
    n_con2 = n_con^2
    idx_x2 = 1:n_x2
    idx_con2 = 1:n_con2
    idx_ineq2 = 1:n_ineq2

    sol_index_list = list()
    sol_index_list$x = idx
    sol_index_list$lams = n_x + idx_lams
    sol_index_list$slacks = n_x + n_ineq + idx_lams
    if(n_eq > 0){
      idx_nus = 1:n_eq
      sol_index_list$nus = n_x + 2*n_ineq + idx_nus
    }
    if(method == 'int'){
      sol_index_list$U_Q = n_x + 2*n_ineq + n_eq + idx_x2
      sol_index_list$U_S = n_x + 2*n_ineq + n_eq + n_x2 + idx_con2
      sol_index_list$R = n_x + 2*n_ineq + n_eq + n_x2 + n_con2 + idx_ineq2
    }

  }
  else if(method == 'admm'){
    n_con = n_x + n_eq
    n_con2 = n_con^2
    idx = 1:n_x
    idx_lams = 1:(2*n_x)
    idx_con1 = 1:n_con
    idx_con2 = 1:n_con2

    sol_index_list = list()
    sol_index_list$x = idx
    sol_index_list$z = n_x+idx
    sol_index_list$u = 2*n_x+idx
    sol_index_list$lams = 3*n_x + idx_lams
    if(n_eq > 0){
      idx_nus = 1:n_eq
      sol_index_list$nus = 5*n_x + idx_nus
    }
    if(unroll_grad){
      sol_index_list$mat_inv = (5*n_x + n_eq) + idx_con2
    }
    else{
      sol_index_list$mat_data = (5*n_x + n_eq) + idx_con2
      sol_index_list$mat_pivots = (5*n_x + n_eq + n_con2) + idx_con1
    }

  }
  else if(method == 'osqp'){
    n_con = n_eq + n_ineq + n_E
    idx = 1:n_x
    idx_con = 1:n_con
    idx_ineq = 1:(2*n_ineq)
    idx_mat = 1: (n_x + n_con)^2

    sol_index_list = list()
    sol_index_list$x = idx
    sol_index_list$z = n_x + idx_con
    sol_index_list$y = (n_x + n_con)+idx_con
    if(n_ineq > 0){
      sol_index_list$lams = (n_x + 2*n_con) + idx_ineq
    }
    if(n_eq > 0){
      idx_nus = 1:n_eq
      sol_index_list$nus = (n_x + 2*n_con + 2*n_ineq) + idx_nus
    }
    sol_index_list$mat_inv = (n_x + 2*n_con + 2*n_ineq + n_eq) + idx_mat

  }
  else if (method == 'scs'){
    idx = 1:n_x
    idx_lams = 1:(n_eq + n_ineq)

    sol_index_list = list()
    sol_index_list$x = idx
    sol_index_list$lams = n_x + idx_lams
    sol_index_list$slacks = n_x + n_eq + n_ineq + idx_lams

  }
  # --- return tensors of dtype int
  if(output_as_tensor){
    sol_index_list = torch_make_index(sol_index_list)
  }

  return(sol_index_list)

}

