#' @export
get_solver_method<-function(any_eq,
                            any_ineq,
                            any_l_1,
                            any_G,
                            is_G_diag,
                            is_E_diag,
                            solver)
{
  if(!any_eq & !any_ineq & !any_l_1){
    method = 'uncon'
  }
  else if(!any_ineq & !any_l_1){
    method = 'eqcon'
  }
  else if(!any_eq & !any_ineq & any_l_1 & solver == 'uncon_l1'){
    method = 'uncon_l1'
  }
  else if(any_l_1 & solver == 'con_l1'){
    method = 'con_l1'
  }
  else if(!any_l_1 & solver == 'int'){
    method = 'int'
  }
  else if(!any_l_1 & solver == 'quadprog'){
    method = 'quadprog'
  }
  else if( (!any_G | is_G_diag) & (!any_l_1 | is_E_diag ) & solver == 'admm'){
    method = 'admm'
  }
  else if(!any_l_1 & solver == 'scs'){
    method = 'scs'
  }
  else{
    method = 'osqp'
  }
  return(method)

}
