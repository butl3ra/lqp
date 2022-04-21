#' @export
torch_diag_batch<-function(x)
{
  is_bm = is_batch_mat(x)
  # --- this is probably unnecessarily slow
  if(is_bm){
    dim_x = dim(x)
    Id = torch_eye_embed(dim_x[1],dim_x[2])
    d = x*Id
  }
  else{
    d = torch_diag(x)
    d = torch_diag(x)
  }
  return(d)
}
