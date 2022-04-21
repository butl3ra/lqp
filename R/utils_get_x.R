#' @export
get_x<-function(x,
                index = NULL,
                dim = 2
                )
{
  if(!is.null(x) & !is.null(index) ){
    return(x$index_select(dim = dim,index = index))
  }
  else{
    return(x)
  }
}


