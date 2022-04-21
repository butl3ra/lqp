#' @export
make_matrix<-function(x)
{
  x_size = get_size(x)
  if(length(x_size) == 1){
    x = x$unsqueeze(2)
  }
  return(x)

}
