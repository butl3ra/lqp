#' @export
torch_solve_qp_uncon<-function(Q,
                                    p,
                                    Q_inv = NULL)
{
  if(is.null(Q_inv)){
    sol = linalg_solve(Q,-p)
  }
  else{
    sol = torch_matmul(Q_inv,-p)
  }
  return(sol)
}
