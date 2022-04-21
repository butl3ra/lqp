#' @export
cor2cov<-function(cor_mat,
                  sigma)
{
  sigma = as.vector(sigma)
  v = diag(sigma)
  cov_mat = v%*%cor_mat%*%v
  colnames(cov_mat) = colnames(cor_mat)
  rownames(cov_mat) = rownames(cov_mat)
  return(cov_mat)
}
