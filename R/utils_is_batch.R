#' @export
is_batch_mat<-function(x)
{
  dim_x = dim(x)
  len_dim_x = length(dim_x)
  is_bm = len_dim_x == 3
  return(is_bm)
}
