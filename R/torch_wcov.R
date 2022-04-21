#' @export
torch_wcov<-function(x,
                        weight,
                        center = TRUE,
                        normalize = FALSE
                        )
{
  # --- E[x^2]
  xx = torch_quad_form(x)
  covar = torch_wma2(xx,weight)

  # --- E[x]^2:
  if(center){
    mu = torch_wma(x,weight)
    mu2 = torch_quad_form(mu)
    covar = covar - mu2
  }
  # --- converting to correlation
  if(normalize){
    covar = torch_cov2cor(covar)
  }

  return(covar)
}
