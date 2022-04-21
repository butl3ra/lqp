#' @export
torch_runif<-function(...,
                      min = 0,
                      max = 1,
                      names = NULL, dtype = NULL, layout = torch_strided(),
                      device = NULL, requires_grad = FALSE)
{
  x =  torch_rand(...,
                  names = names,
                  dtype = dtype,
                  layout = layout,
                  device = device,
                  requires_grad = requires_grad)

  x*(max - min) + min

}
