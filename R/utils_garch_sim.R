#' @export
garch_sim<-function(alpha = matrix(1.221116e-01,1,2),
                    beta = matrix(8.623080e-01,1,2),
                    omega = matrix(sqrt(2.536084e-06),1,2),
                    mu = matrix(0,1,2),
                    cor_mat = diag(2),
                    n_samples = 100,
                    n_trans = 100)
{
  # --- simulate a CCC GARCH(p,q) process:

  # --- prep inputs
  alpha = as.matrix(alpha)
  beta = as.matrix(beta)
  omega = as.matrix(omega)

  # --- ncol
  n_x = ncol(beta)
  is_multi = n_x > 1
  p = nrow(beta)
  q = nrow(alpha)
  d = max(p,q)
  total_n = n_samples + n_trans

  # --- holder for x:
  x = matrix(0,total_n,n_x)
  sigma_t = x

  # --- generate sigma2
  w = omega^2
  sigma2 = colSums(alpha) + colSums(beta)
  sigma2 = w/(1-sigma2)
  sigma_t[1:d,]=sigma2
  # --- generate errors and init x:
  if(is_multi){
    mu_eps = rep(0,n_x)
    eps = mvrnorm(total_n,mu_eps,cor_mat)

    cov_mat = cor2cov(cor_mat,sqrt(sigma2))
    x[1:d,] = mvrnorm(d,mu_eps,cov_mat)
  }
  else{
    eps = rnorm(total_n)
    eps = matrix(eps)

    x[1:d,] = rnorm(d,sd = sqrt(sigma2))
  }

  # --- main loop
  idx = (d+1):total_n
  idx_q = 1:q
  idx_p = 1:p
  for(i in idx){
    idx_q_i = i - idx_q
    idx_p_i = i - idx_p
    sigma_t[i,] = w + colSums(alpha*x[idx_q_i,,drop=F]^2) + colSums(beta*sigma_t[idx_p_i,,drop=F])
    x[i,] = eps[i,]*sqrt(sigma_t[i,])
  }

  # --- cleanup and output
  x = x[(n_trans+1):total_n,,drop=F]
  x = sweep(x, 2, mu, '+',check.margin=F)
  return(x)

}


