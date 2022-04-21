#' @export
torch_scale<-function(x,
                      dim = 1,
                      normalize = TRUE,
                      center = TRUE,
                      scale_factor = 1)
{
  if(center){
    x = torch_center(x,dim = dim)
  }
  if(normalize){
    x = torch_normalize(x,dim = dim)
  }
  x = x*scale_factor
  return(x)
}

#' @export
torch_moving_scale<-function(x,
                             weight,
                             normalize = TRUE,
                             center = TRUE)
{
  if(center){
    mu = torch_wma(x,weight)
    x = x - mu
  }
  if(center){
    std_dev = torch_wsd(x,weight)
    x = x / std_dev
  }
  return(x)

}



