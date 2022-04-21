#' @export
torch_sum_1<-function(x,
                      dim = 1,
                      eps = 10^-12)
{
  x_sum = torch_sum(x,dim = dim,keepdim=T)
  y = x/(x_sum + eps)
  return(y)
}
