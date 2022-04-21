#' @export
check_nn_parameter<-function(...)
{
  params = list(...)

  #is_param = Map( is_nn_parameter,x = params)
  is_param = lapply(params,check_requires_grad)
  is_param = unlist(is_param)
  return(is_param)
}

#' @export
check_requires_grad<-function(x)
{
  rg = x$requires_grad
  if(is.null(rg)){
    rg = FALSE
  }
  return(rg)
}
