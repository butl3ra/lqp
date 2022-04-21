#' @export
torch_crossprod<-function(x,
                          y = NULL)
{
  xt = torch_transpose_batch(x)
  if( is.null(y) ){
    mat = torch_matmul(xt,x)
  }
  else{
    mat = torch_matmul(xt,y)
  }
  return( mat )
}
