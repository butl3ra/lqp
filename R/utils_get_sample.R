#' @export
get_sample<-function(x,
                     index = NULL,
                     dim = 1)
{
  dim_x = get_size(x)
  do_sample =  !is.null(x) & !is.null(index)
  do_sample = do_sample & (dim_x[dim] > 1)
  if(do_sample){
    return(x$index_select(dim = dim,index = index))
  }
  else{
    return(x)
  }
}
