#' @export
torch_reduce<-function(x,
                       method = 'mean'
                       )
{
  if(method == 'max'){
    x_r= torch_max(x)
  }
  else if(method == 'median'){
    x_r= torch_median(x)

  }
  else{
    x_r= torch_mean(x)
  }
  return(x_r)

}
