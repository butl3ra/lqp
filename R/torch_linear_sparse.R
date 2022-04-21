#' @export
torch_linear_sparse<-function(x,
                              P = torch_eye(ncol(x)),
                              weight,
                              bias = NULL
                              )
{
  x_u3 = x$unsqueeze(3)
  x_theta = x_u3*weight
  y_hats = torch_matmul(P,x_theta)
  y_hats = y_hats$squeeze(3)
  if(!is.null(bias)){
    y_hats = y_hats + bias
  }
  return(y_hats)
}





