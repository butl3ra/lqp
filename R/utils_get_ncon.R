#' @export
get_ncon<-function(x)
{
  d_x = get_size(x)
  if(length(d_x)==3){
    n_con = d_x[2]
  }
  else{
    n_con = d_x[1]
  }
  return(n_con)
}
