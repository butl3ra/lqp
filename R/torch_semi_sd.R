#' @export
torch_semi_sd<-function(x,
                        min = NULL,
                        max = 0,
                        center = F)
{
  v = torch_semi_var(x,min = min, max = max, center = center)
  return(sqrt(v))

}
