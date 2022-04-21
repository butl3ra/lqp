#' @export
torch_garch_dcc<-function(x,
                          sigma,
                          cor_mat_bar,
                          a,
                          b,
                          buffer = 0,
                          n_ahead = 1)
{
  # --- Dynamic correlation:
  b = b*0.99# stability
  a_b = a + b

  # --- standardized values:
  zt = x/sigma
  zz = torch_quad_form(zt,zt)
  d_zz = dim(zz)
  nc_x = d_zz[2]
  nc_x2 = nc_x^2
  n_obs = d_zz[1]

  # --- compute weight
  if(as.logical(a_b>1)){
    w = 0.01
  }
  else{
    w = (1-a_b)/(1-b)
  }
  with_no_grad({
    num_weights = ceiling(log(0.01,base=as.numeric(b)))+1
  })
  weight = torch_repeat(b,c(num_weights,nc_x2))
  weight = torch_cumprod(weight,1)/b
  weight = torch_flip(weight,1)

  # --- add buffer to weight:
  if(as.logical(buffer > 0)){
    weight = torch_cat(list(weight,torch_zeros(buffer,nc_x)))
  }

  # --- compute moving average of correlations
  zz_ma =  torch_wma2(zz,weight=weight)

  # --- cor is average between static and dynamic components
  cor_mat = w*cor_mat_bar  + a*zz_ma

  # --- n_ahead greater than 1 -> weighted average between var and long-term omeg
  if(n_ahead > 1){
    with_no_grad({
      a_b = torch_clamp(a_b,min = 0,max=0.99)
    })
    a_b_n = a_b^n_ahead
    co2 = (1/n_ahead)*(1-a_b_n)/(1-a_b)
    co1 = (1-co2)#/(1-a_b)

    cor_mat = co1*cor_mat_bar + co2*cor_mat
  }

  # --- make a proper correlation matrix:
  cor_mat = torch_cov2cor(cor_mat)

  # --- convert to covariance:
  cov_mat = torch_cor2cov(cor_mat,sigma)

  return(cov_mat)
}
