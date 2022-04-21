#' @export
torch_normalize<-function(x,
                          dim = 1)
{
  std_dev = torch_sd(x,dim = dim,keepdim=T)
  #std_dev[std_dev==0] = 1
  std_dev = torch_threshold_(std_dev,10^-12,1)
  y = x/std_dev
  return(y)
}
