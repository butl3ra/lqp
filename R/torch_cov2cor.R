#' @export
torch_cov2cor<-function(x)
{
  is_bm = is_batch_mat(x)
  if(is_bm){
    y = torch_diagonal(x,dim1 = 2,dim2 = 3)
    y = sqrt(y)
    y = torch_diag_embed(1/y)
  }
  else{
    y = torch_diag(x)
    y = sqrt(y)
    y = torch_diag(1/y)
  }
  z = torch_matmul(torch_matmul(y,x),y)
  return(z)

}

