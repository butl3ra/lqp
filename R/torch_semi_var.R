#' @export
torch_semi_var<-function(x,
                         min = NULL,
                         max = 0,
                         center = FALSE)
{

  z = torch_clamp(x,min = min, max = max)
  if(is.null(min)){
    min = -Inf
  }
  if(is.null(max)){
    max = Inf
  }
  thresh_count = z > min & z < max
  n_obs = thresh_count$sum(1)

  #First moment
  z_var = (z^2)$sum(1)/(n_obs - 1)
  if(center){
    mu = z$sum(1)/(n_obs-1)
    z_var = (z_var - mu^2)
  }
  return(z_var)
}
