#' @export
torch_eye_embed<-function(n_batch,n_dim)
{
  Id = torch_ones(n_batch,n_dim)
  torch_diag_embed(Id)
}
