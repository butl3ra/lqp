#' @export
get_any<-function(x,
                  threshold = NULL)
{
  n_x = get_size(x)[1]
  any_x = n_x > 0
  if(any_x & !is.null(threshold)){
    any_x = any(as.logical(x > threshold))
  }
  return(any_x)
}
