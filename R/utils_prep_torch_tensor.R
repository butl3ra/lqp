#' @export
prep_torch_tensor<-function(x,
                            target_dim = 1)
{
  x_class = class(x)[1]
  # --- if null then ignore all steps:
  if(x_class != "NULL"){
    # --- convert to tensor:
    if(x_class != 'torch_tensor'){
      x = as_torch_tensor(x)
    }
    # --- if vector then convert to matrix:
    dim_x = dim(x)
    len_dim = length(dim_x)
    if(len_dim == 1){
      x = x$unsqueeze(2)
    }
    # --- if no batch dimension then add batch:
    dim_x = dim(x)
    len_dim = length(dim_x)
    if(len_dim == 2){
      x = x$unsqueeze(target_dim)
    }

  }
  return(x)
}
