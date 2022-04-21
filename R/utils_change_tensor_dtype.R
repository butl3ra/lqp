#' @export
change_tensor_dtype<-function(x,
                              dtype = dtype)
{
  # --- this is a bit of a hack
  x_class = class(x)[1]
  if(x_class == 'torch_tensor'){
    x = as_torch_tensor(as_array(x),dtype = dtype)
  }
  return(x)
}
