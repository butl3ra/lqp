#' @export
torch_sd<-function(self,
                   dim=1,
                   unbiased = TRUE,
                   keepdim = FALSE){
  torch_sqrt(torch_var(self = self,
                 dim = dim,
                 unbiased = unbiased,
                 keepdim = keepdim) )
}
