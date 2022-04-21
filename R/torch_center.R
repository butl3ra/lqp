#' @export
torch_center<-function(x,
                       dim = 1
)
{
  mu = torch_mean(x,dim,keepdim = T)
  return(x - mu)
}
