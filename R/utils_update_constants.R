#' @export
update_constant<-function(object,
                          value)
{
  if("constant"%in%names(object)){
    object$constant = value
  }
  return(object)
}
