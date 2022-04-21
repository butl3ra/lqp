#' @export
get_n_x<-function(x)
{
  x_size = dim(x)
  n_x = x_size[2]
  return(n_x)
}
