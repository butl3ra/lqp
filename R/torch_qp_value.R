#' @export
torch_qp_value<-function(x,
                         p,
                         Q)
{
  xt = torch_transpose_batch(x)
  0.5 * torch_quad_form_mat(x = x,mat = Q) + torch_matmul(xt,p)
}
