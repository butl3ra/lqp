#' @export
torch_tensor_to_list<-function(x,
                               index_list,
                               dim = 2)
{
  out = lapply(index_list,get_x, x = x, dim = dim)
  return(out)
}
