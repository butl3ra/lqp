#' @export
torch_cor<-function(x,
                    center = F)
{
  cov_mat = torch_cov(x,center = center)
  mat = torch_cov2cor(cov_mat)
  return(mat)
}


