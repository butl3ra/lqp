#' @export
torch_wvar<-function(input,
                     weight,
                     center = FALSE,
                     bias = NULL)
{
  # --- prep
  input = make_matrix(input)
  weight = make_matrix(weight)
  input_size = get_size(input)
  weight_size = get_size(weight)
  nr_input = input_size[1]
  nc_input = input_size[2]
  nr_weight = weight_size[1]
  nc_weight = weight_size[2]

  # --- repeat weight if necessary
  if(nc_input != nc_weight){
    weight = torch_repeat(weight,c(1,nc_input))
  }

  # --- E[x^2]
  out = torch_wma(input = input^2,
                  weight = weight,
                  bias = bias)
  # --- E[x]^2
  if(center){
    mu = torch_wma(input = input,
                   weight = weight,
                   bias = bias)
    out = out - mu^2
  }

  return(out)

}

#' @export
torch_wsd<-function(input,
                    weight,
                    center = FALSE,
                    bias = NULL)
{
  out = torch_wvar(input = input,
                   weight = weight,
                   center = center,
                   bias = bias)
  out = sqrt(out)
  return(out)

}
# --- an unbiased estimator would be x*sqrt(length(weight))/sqrt(length(weight)-1)
