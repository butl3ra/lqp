#' @export
generate_coef_list<-function(n_vars = 10,
                             pct_true = 0.5,
                             method = c('rnorm','constant','runif'),
                             polys = c(1),
                             ...)
{
  out = lapply(polys,
               generate_coef,
               n_vars = n_vars,
               pct_true = pct_true,
               method = method,
               ...)
  return(out)
}

#' @export
generate_coef<-function(n_vars = 10,
                        pct_true = 0.5,
                        method = c('rnorm','constant','runif'),
                        p = 1,
                        ...)
{
  b = rep(0,n_vars)
  idx_s = 1:n_vars
  if(pct_true < 1){
    size = round(pct_true*n_vars)
    idx_s = sample(idx_s,size = size)
  }
  n = length(idx_s)
  fn = match.fun(method[1])
  b[idx_s] = fn(n = n, ...)
  return(b)
}

#' @export
constant<-function(n,
                   value = 1)
{
  rep(value,n)
}

#' @export
mvrnorm<-function(n_obs,
                  mu,
                  V)
{
  p <- length(mu)
  d_V = dim(V)
  if(d_V[1]!=p | d_V[2]!=p){
    stop('length of mu does not match dim of V')
  }

  D <- chol(V)
  z = matrix(rnorm(n_obs * p), ncol = p)
  out = z %*% D + rep(mu, rep(n_obs, p))
  colnames(out) = colnames(V)
  return(out)
}

#' @export
generate_cor<-function(n_vars,
                       rho = 0,
                       v = 1:n_vars)
{
  if(rho==0){
    mat = diag(n_vars)
  }
  else{
    mat = dist(v,method = 'manhattan',upper=T,diag=T)
    mat = as.matrix(mat)
    mat = rho^mat
  }
  colnames(mat) = rownames(mat) = 1:n_vars
  return(mat)
}


#' @export
generate_sd<-function(n_vars,
                        min = 0.10,
                        max = 0.25,
                        scale_factor = sqrt(252),
                        ...)
{
  std_devs = runif(n_vars,min = min,max = max)
  return(std_devs/scale_factor)
}

#' @export
generate_cov<-function(std_devs,
                       rho_mat
)
{
  std_devs = diag(std_devs)
  covar = std_devs%*%rho_mat%*%std_devs
  colnames(covar) = rownames(covar) = colnames(rho_mat)
  return(covar)
}

#' @export
generate_x<-function(n_x,
                     n_obs,
                     mu = rep(0,n_x),
                     V = diag(n_x)
)
{
  mvrnorm(n_obs = n_obs,mu = mu,V = V)
}

#' @export
generate_y<-function(x,
                     b)
{
  y = x%*%b
  return(y)
}

#' @export
generate_P<-function(n_vars,
                     n_x)
{
  m = n_vars*n_x
  P = matrix(0,n_vars,m)
  for(i in 1:n_vars){
    idx = (i*n_x - n_x+1):(i*n_x)
    P[i,idx] =1
  }
  return(P)
}

#' @export
generate_noise<-function(y,
                         snr)
{
  v = var(y)
  sigma = sqrt(v/snr)
  noise = rnorm(length(y),0,sigma)
  return(noise)
}

#' @export
generate_artificial_data<-function(n_y,
                                   n_obs = 7500,
                                   n_x = 1,
                                   snr = 0.05,
                                   rho = 0.70,
                                   min_sd = 0.10,
                                   max_sd = 0.25,
                                   polys = c(1,3),
                                   alpha_nonlinear = 0,
                                   sparse = TRUE,
                                   scale_factor = 1)
{

  n_total_x = n_y * n_x
  #generate daily std_dev:
  std_devs = generate_sd(n_vars = n_y,
                       min = min_sd,
                       max = max_sd,
                       scale_factor = scale_factor)


  #generate P
  if(sparse){
    P = generate_P(n_y,n_x)
    theta = generate_coef_list(n_total_x,
                             pct_true = 1,
                             method = 'runif',
                             polys = polys,
                             min = -3,
                             max = 3)
    P_theta = lapply(theta,"*",t(P))
  }
  else{
    theta = generate_coef_list(n_total_x,
                               pct_true = 1,
                               method = 'runif',
                               polys = polys,
                               min = -3,
                               max = 3)
    theta = lapply(theta,matrix,nrow = n_x,ncol = n_y)
    P_theta = theta
    P = NULL
    n_total_x = n_x
  }

  v = rep(1:n_x,n_y)-1 + 1:n_total_x
  rho_mat_x = generate_cor(n_vars = n_total_x,
                           rho = 0,
                           v = v)

  x = generate_x(n_x = n_total_x,
                 n_obs = n_obs,
                 V = rho_mat_x )

  x_list = lapply(polys,function(i) x^i)
  f = Map(generate_y,x = x_list, b = P_theta)
  P_theta = lapply(P_theta,t)
  f = Reduce("+",f)
  #f = x*sin(f)
  f = (1-alpha_nonlinear)*f + (alpha_nonlinear) * f * sin(f)

  #generate random errors:
  rho_mat = generate_cor(n_vars = n_y,
                         rho = rho)
  var_f = matrixStats::colVars(f,na.rm=T)#diag(cov(f))
  std_dev_errors =  sqrt(var_f/snr)
  V_errors = generate_cov(std_dev_errors,rho_mat)
  errors = generate_x(n_x = n_y,
                      n_obs = n_obs,
                      V = V_errors)

  #Tests:
  if(F){
    test = f+errors
    cor(test)
    diag(cov(f))/diag(cov(errors))
    lm(test[,1]~x[,1:n_x]+0)
  }

  #generate y:
  y = f + errors
  std_dev_y = matrixStats::colSds(y,na.rm=T)
  scale_factor_y = std_devs/std_dev_y
  ys = sweep(y, 2, scale_factor_y, '*',check.margin=F)

  scale_factor_x = rep(scale_factor_y,rep(n_x,n_y))
  xs = sweep(x, 2, scale_factor_x, '*',check.margin=F)
  #theta = lapply(theta,'*',scale_factor_x)

  out = list(x = xs,
             y = ys,
             P = P,
             theta = theta,
             P_theta = P_theta)
  return(out)
}

#' @export
relative_risk<-function(b_hat,
                        b_true,
                        rho_mat)
{
  d = (b_hat - b_true)
  num = d%*%rho_mat%*%d
  denom = b_true%*%rho_mat%*%b_true
  return(as.numeric(num/denom))
}

#' @export
pct_variance_explained<-function(y_hat,
                                 y_true)
{
  1-mean((y_true - y_hat)^2)/mean(y_true^2)
}

#' @export
relative_test_error<-function(y_hat,
                              y_true,
                              noise)
{
  num = mean((y_true - y_hat)^2)
  denom = mean(noise^2)
  return(num/denom)
}
