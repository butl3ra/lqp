#' @export
torch_make_index<-function(x,
                           dtype = torch_int32())
{
  if(is.list(x)){
    x = lapply(x,torch_make_index,dtype = dtype)
  }
  else{
    if(!is.null(x)){
      if(!is_torch_tensor(x)){
        x = as_torch_tensor(x,dtype = dtype)
      }
      else{
        if(x$dtype != 'torch_Int'){
          x = as_torch_tensor(as_array(x),dtype = dtype)
        }
      }
    }
  }
  return(x)
}
