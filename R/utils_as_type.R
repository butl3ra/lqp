#' @export
as_torch_tensor<-function(x,
                          dtype = NULL,#torch_float64(),#NULL
                          device = NULL,
                          requires_grad = FALSE,
                          pin_memory = FALSE
)
{
  torch_tensor(data = x,
               dtype = dtype,
               device = device,
               requires_grad = requires_grad,
               pin_memory = pin_memory)
}

#' @export
as_array<-function(x)
{
  torch::as_array(x)
}

#' @export
as_array.NULL<-function(x,
                        ...)
{
  NULL
}

#' @export
is_null<-function(x)
{
  length(x)<1
}

#' @export
is_torch_tensor<-function(x)
{
  class(x)[1] == "torch_tensor"
}

#' @export
as_matrix<-function(x,
                    ...)
{
  UseMethod("as_matrix")
}

#' @export
as_matrix.numeric<-function(x,
                            ...)
{
  matrix(x)
}

#' @export
as_matrix.torch_tensor<-function(x,
                                 ...)
{
  torch::as_array(x)
}

#' @export
as_matrix.matrix<-function(x,
                           ...)
{
  x
}

#' @export
as_matrix.array<-function(x,
                          ...)
{
  x
}

#' @export
as_matrix.NULL<-function(x,
                         ...)
{
  x
}
