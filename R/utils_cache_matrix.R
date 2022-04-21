#' @export
cache_matrix<-function(sol_list,
                       solver_method,
                       is_param,
                       info,
                       unroll_grad)
{
  out = list()
  if(solver_method == 'admm'){
    if(!is_param['Q'] & !is_param['A'] & !is_param['D'] & !is_param['lambda_2'] & unroll_grad){
      mat_inv = sol_list$mat_inv
      mat_inv = torch_reshape_mat(mat_inv,forward = FALSE)
      #detach_grad(mat_inv)
      mat_inv = mat_inv$detach()
      out$mat_inv = mat_inv
    }
    if(!is_param['Q'] & !is_param['A'] & !is_param['D'] & !is_param['lambda_2'] & !unroll_grad){
      mat_data = sol_list$mat_data
      mat_data = torch_reshape_mat(mat_data,forward = FALSE)
      mat_data = mat_data$detach()

      mat_pivots = sol_list$mat_pivots
      mat_pivots = mat_pivots$squeeze(3)
      mat_pivots = mat_pivots$detach()

      out$mat_data = mat_data
      out$mat_pivots = mat_pivots
    }
  }
  else if(solver_method == 'osqp'){
    if(!is_param['Q'] & !is_param['A'] & !is_param['G'] & !is_param['G']){
      mat_inv = sol_list$mat_inv
      mat_inv = torch_reshape_mat(mat_inv,forward = FALSE)
      #mat_inv = detach_grad(mat_inv)
      mat_inv = mat_inv$detach()
      out$mat_inv = mat_inv
    }
  }
  else if(solver_method == 'int'){
    if(!is_param['Q'] & !is_param['D'] & !is_param['lambda_2'] ){
      U_Q = sol_list$U_Q
      U_Q = torch_reshape_mat(U_Q,forward = FALSE)
     #U_Q = detach_grad(U_Q)
      U_Q = U_Q$detach()
      out$U_Q = U_Q
    }
    if(!is_param['Q'] & !is_param['D'] & !is_param['lambda_2'] & !is_param['A'] & !is_param['G']){
      U_S = sol_list$U_S
      U_S = torch_reshape_mat(U_S,forward = FALSE)
      #U_S = detach_grad(U_S)
      U_S = U_S$detach()
      out$U_S = U_S
    }
    if(!is_param['Q'] & !is_param['D'] & !is_param['lambda_2'] & !is_param['A'] & !is_param['G']){
      R = sol_list$R
      R = torch_reshape_mat(R,forward = FALSE)
      #R = detach_grad(R)
      R = R$detach()
      out$R = R
    }
  }
  return(out)
}
