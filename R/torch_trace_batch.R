#' @export
torch_trace_batch<-function(x)
{
  is_bm = is_batch_mat(x)
  if(is_bm){
    d = torch_diagonal(x,dim1=2,dim2=3)
    tr = torch_sum(d,dim=2)
  }
  else{
    tr = torch_trace(x)
  }
  return(tr)
}

