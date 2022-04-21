#' @export
get_n_obs_proxy<-function(Q,
                          p,
                          tol = 10^-12)
{
  n_obs = torch_mean(torch_diagonal(Q,dim1=2,dim2=3))
  n_obs = as_array(n_obs)
  if(n_obs < tol){
    n_obs = torch_mean(p^2)
    n_obs = as_array(n_obs)
  }
  return(n_obs)

}
