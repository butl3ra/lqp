#' @export
torch_quad_form<-function(x,
                          y = x,
                          dim_x = 3,
                          dim_y = 2)
{
    torch_matmul(x$unsqueeze(dim_x),y$unsqueeze(dim_y))
}

#' @export
torch_quad_form_mat<-function(x,
                              mat)
{
  xt = torch_transpose_batch(x)
  torch_matmul(torch_matmul(xt,mat),x)
}
