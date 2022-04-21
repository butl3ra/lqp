#' @export
prep_bound<-function(x,
                     n_x,
                     default = NULL,
                     null_size = c(0,0,0))
{
  if(is.null(x)){
    x = default
  }
  x = prep_torch_tensor(x)
  x_size = get_size(x,null_size)
  if(x_size[2] == 1){
    dim_rep = x_size
    dim_rep[2] = n_x
    x = torch_repeat(x = x,dim = dim_rep)
  }
  return(x)
}
