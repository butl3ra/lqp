#' @export
nnf_qp_loss<-function(x,
                      p,
                      Q,
                      reduction = "mean",
                      ...)
{
  loss = torch_qp_value(x = x, p = p, Q = Q)
  if(reduction == 'mean'){
    loss = loss$mean()
  }
  else{
    loss = loss$sum()
  }
  return(loss)
}
