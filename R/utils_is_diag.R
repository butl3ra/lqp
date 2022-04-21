#' @export
is_diag<-function(x,
                  dim1 = 2,
                  dim2 = 3,
                  tol = 10^-10
)
{
  x_size = get_size(x)
  test = x_size[dim1] == x_size[dim2]
  if(test){
    x_diag = torch_diagonal(x,dim1=dim1,dim2=dim2)
    x_diag = torch_diag_embed(x_diag)
    x_diff = x - x_diag
    test = mat_sum < tol
  }
  return(test)

}
