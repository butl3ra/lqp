#' @export
torch_norm_2<-function(x,
                       dim,
                       keepdim = FALSE,
                       dtype = NULL)
{

  sqrt( torch_sum(x^2,dim = dim,keepdim = keepdim,dtype = dtype))

}
