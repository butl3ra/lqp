#' @export
nnf_msr_loss<-function(z,
                       y,
                      ...)
{
  loss = torch_sharpe_ratio(x = z, y = y)
  return(-loss)
}

#' @export
torch_sharpe_ratio<-function(x,
                             y)
{
  pret =  x*y
  pret = pret$sum(2)
  sr = pret$mean()/torch_sqrt(pret$var())
  return(sr)
}

#' @export
torch_pret<-function(x,y)
{
  pret =  x*y
  pret = pret$sum(2)
  return(pret)
}



