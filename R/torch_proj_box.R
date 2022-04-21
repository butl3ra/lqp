#' @export
torch_proj_box<-function(x,
                         lb,
                         ub,
                         any_lb = TRUE,
                         any_ub = TRUE)
{
  if(any_lb){
    lb_diff = lb - x
    lb_diff_relu = torch_relu(lb_diff)
    x = x + lb_diff_relu
  }
  if(any_ub){
    ub_diff = x - ub
    ub_diff_relu = torch_relu(ub_diff)
    x = x - ub_diff_relu
  }
  return(x)
}
