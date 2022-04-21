#' @export
qp_prep_sol<-function(sol,
                      output_as_list,
                      sol_index_list)
{
  if(!output_as_list){
    # --- return the full output --- this is used for warm-starting
    x = sol
    # --- slice into list
    sol = torch_tensor_to_list(x = sol,
                               index_list = sol_index_list,
                               dim = 2)

  }# --- otherwise return just x
  else{
    sol_index_list  = NULL
    x = sol$x#$squeeze(3)
  }
  out = list(sol = sol,
             x = x)
  return(out)
}
