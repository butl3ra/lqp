#' @export
torch_rep<-function(x,
                    dim)
{
  tmp = torch_ones(dim)
  return(tmp*x)
}
