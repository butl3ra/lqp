#' @export
torch_garch_ccc<-function(x,
                          alpha,
                          beta,
                          omega,
                          center = TRUE,
                          buffer = 0,
                          n_ahead = 1)
{
  # --- compute sigmas:
  sigma = torch_garch(x,
                      alpha = alpha,
                      beta = beta,
                      omega = omega,
                      buffer = buffer,
                      n_ahead = n_ahead)

  # --- correlation
  x_sigma = x/sigma
  cor_mat = torch_cor(x = x_sigma,center = center)

  # --- compute covariance
  covar = torch_cor2cov(cor_mat = cor_mat,sigma = sigma)

  return(covar)
}
