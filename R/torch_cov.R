#' @export
torch_cov<-function(x,
                    center = F)
{
  x_size = get_size(x)
  is_bm = is_batch_mat(x)
  if(is_bm){
    d = 2
    n_obs = x_size[d]
  }
  else{
    d = 1
    n_obs = x_size[d]
  }

  mat = torch_crossprod(x)
  mat = mat / n_obs

  if(center){
    mu = torch_mean(x,d,keepdim=T)
    mu_mat = torch_crossprod(mu)
    mat = mat - mu_mat
  }
  return(mat)
}
