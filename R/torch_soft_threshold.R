#' @export
torch_soft_threshold<-function(x,tau)
{
  x_sign = torch_sign(x)
  x_abs = torch_abs(x)
  y = x_sign*torch_relu(x_abs - tau)
  return(y)
}
