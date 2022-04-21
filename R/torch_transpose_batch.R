#' @export
torch_transpose_batch<-function(x)
{
  is_bm = is_batch_mat(x)
  if(is_bm){
    dim0 = 2
    dim1 = 3
  }
  else{
    dim0 = 1
    dim1 = 2
  }
  xt = torch_transpose(x,dim0 = dim0, dim1 = dim1)
  return(xt)

}
