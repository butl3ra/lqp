#' @export
torch_garch<-function(x,
                      alpha,
                      beta,
                      omega,# this is sqrt --> different then usual omega definition
                      buffer = 0,
                      n_ahead = 1)
{

  # --- prep:
  nc_x = ncol(x)
  n_alpha = nrow(alpha)
  n_beta = nrow(beta)
  min_x = max( n_alpha,n_beta)


  # --- create implied garch weight
  omega_2 = omega^2

  w = (1/(1-beta))*omega_2
  with_no_grad({
    num_weights = ceiling(log(0.01,base=as.numeric(beta$max())))+1
  })
  weight = torch_repeat(beta,c(num_weights,1))
  weight = torch_cumprod(weight,1)/beta
  weight = torch_flip(weight,1)

  # --- add buffer to weight:
  if(as.logical(buffer > 0)){
    weight = torch_cat(list(weight,torch_zeros(buffer,nc_x)))
  }

  # --- moving average of squared observations
  x_2 = x^2
  a = torch_wma(input = x_2,
                weight = weight,
                bias = NULL)

  # --- variance:
  var = w + alpha*a

  # --- n_ahead greater than 1 -> weighted average between var and long-term omega
  if(n_ahead > 1){
    a_b = alpha + beta
    with_no_grad({
      a_b = torch_clamp(a_b,min = 0,max=0.99)
    })
    a_b_n = a_b^n_ahead
    co2 = (1/n_ahead)*(1-a_b_n)/(1-a_b)
    co1 = (1-co2)/(1-a_b)

    var = co1*omega_2 + co2*var
  }

  # --- sigma:
  sigma = sqrt(var)

  return(sigma)

}


