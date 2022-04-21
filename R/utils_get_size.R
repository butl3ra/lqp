#' @export
get_size<-function(x,
                   null_size = c(0,0))
{
  len_x = length(x)
  if(len_x ==0){
    sz = null_size
  }
  else if(is.vector(x)){
    sz = c(1,len_x)
  }
  else if(is_torch_tensor(x)){
    sz = dim(x)
  }
  if(is.array(x)){
    sz = dim(x)
  }
  return(sz)

}




