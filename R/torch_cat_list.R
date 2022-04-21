#' @export
torch_cat_list<-function(tensors,
                         dim = 1L)
{
  len = sapply(tensors,length)
  tensors = tensors[len > 0]
  torch_cat(tensors,dim = dim)

}
