#' @export
detach_grad<-function(x)
{
  # --- this is a bit of a hack
  x_class = class(x)[1]
  if(x_class == 'torch_tensor'){
    #x = as_torch_tensor(as_array(x))
    x = x$detach()
  }
  return(x)
}
