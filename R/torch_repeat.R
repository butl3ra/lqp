#' @export
torch_repeat<-function(x,
                       dim)
{
  return(x$'repeat'(dim))

}
