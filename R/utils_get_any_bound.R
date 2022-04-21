#' @export
get_any_lb<-function(lb)
{
  any_lb = get_any(lb)
  if(any_lb){
    any_lb = as.logical(torch_max(lb) > -Inf)
  }
  return(any_lb)
}

#' @export
get_any_ub<-function(ub)
{
  any_ub = get_any(ub)
  if(any_ub){
    any_ub = as.logical(torch_min(ub) < Inf)
  }
  return(any_ub)

}
