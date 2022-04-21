#' @export
torch_cor2cov<-function(cor_mat,
                        sigma)
{
  # --- make cor mat 3 dimensional
  dim_cor = dim(cor_mat)
  len_dim_cor = length(dim_cor)
  dim_sigma = dim(sigma)
  len_dim_sigma = length(dim_sigma)
  if(len_dim_cor < 3 & len_dim_sigma > 1){
    cor_mat = cor_mat$unsqueeze(1)
  }
  if(len_dim_sigma == 3){
    sigma = sigma$squeeze(3)
  }

  # --- make sigma an embedded diagonal matrix
  sigma = torch_diag_embed(sigma)

  covar = torch_matmul(torch_matmul(sigma,cor_mat),sigma)
  return(covar)

}


