#' @export
window_sma<-function(l,
                     n_col =1)
{
  w = torch_ones(c(l,n_col))
  w = w / l
  return(w)
}

#' @export
window_ema<-function(l,
                     n_col = 1,
                     max_rows = 250)
{
  if(l == 1){
    w = window_sma(l = l, n_col = n_col)
  }
  else{
    alpha = 2/(l+1)
    beta = 1 - alpha
    d_max = round( min(max_rows, 10/(1 - beta)) )
    idx = (d_max-1):0
    w = (1-beta) * beta^idx
    w = w / sum(w)
    w = as_torch_tensor(w)
    w = w$unsqueeze(2)
    if(n_col > 1){
      w = w$'repeat'(c(1,n_col))
    }
  }

  return(w)

}

