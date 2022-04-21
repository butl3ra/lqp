#' @export
nnf_var_loss<-function(z,
                       y,
                       ...)
{

  pret = torch_pret(x = z, y = y)
  loss = pret$var()
  return(loss)
}
