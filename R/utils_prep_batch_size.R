#' @export
prep_batch_size<-function(x,
                          batch_size)
{
  x_size = get_size(x)
  if(x_size[1] > 0){
    if(x_size[1] < batch_size){
      u = rep(1,length(x_size))
      u[1] = batch_size
      #x = torch_rep(x,u)
      x = torch_repeat(x = x,dim = u)
    }
  }
  return(x)
}
