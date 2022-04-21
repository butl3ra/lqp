#' @export
torch_wma<-function(input,
                    weight,
                    bias=NULL)
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
  # --- reshaping
  input = input$t()$unsqueeze(1)
  weight = weight$t()$unsqueeze(2)


  # --- conv1d:
  padding = nr_weight-1
  groups = nc_input
  out = nnf_conv1d(input = input,
                   weight = weight,
                   bias = bias,
                   groups =groups,
                   padding = padding )
  out = out$squeeze(1)
  out = out[,1:nr_input,drop=F]
  out = out$t()
  return(out)

}

#' @export
torch_wma2<-function(input,
                     weight,
                     bias=NULL)
{
  input_size = get_size(input)
  n_obs = input_size[1]
  n_col = input_size[2]*input_size[3]

  #here
  out = torch_wma(input = input$view(c(n_obs,n_col)),
                    weight = weight,
                    bias = bias)

  out = out$view(c(n_obs,input_size[2],input_size[3]))
  return(out)

}

