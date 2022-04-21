#' @export
check_warm_start<-function(...)
{
  lst = list(...)
  is_null = sapply(lst,is.null)
  warm_start = all(!is_null)
  return(warm_start)
}
