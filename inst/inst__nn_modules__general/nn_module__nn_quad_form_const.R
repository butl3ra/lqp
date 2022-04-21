nn_quad_form_const <- torch::nn_module(
  classname = "nn_quad_form_const",
  # --- init
  initialize = function(in_features,
                        out_features,
                        x_mat,
                        bias = FALSE,
                        is_bias_diagonal = T,
                        bias_min = -3,
                        bias_max = -1,
                        ...) {
    # --- nn_parameters
    self$x_mat = x_mat
    # --- weight:
    self$weight <- nn_parameter(torch_rand(in_features, out_features))

    # --- bias:
    self$has_bias = bias
    self$is_bias_diagonal = is_bias_diagonal
    if(bias){
      if(is_bias_diagonal){
        self$bias <- nn_parameter(torch_runif(out_features,min = bias_min, max= bias_max)  )
      }
      else{
        self$bias <- nn_parameter(torch_randn(out_features,out_features))
      }
    }

  },
  # --- forward:
  forward = function(...) {

    # --- x is a matrix or batch matrix: #B^T Omega B
    out =  torch_quad_form_mat(self$weight,self$x_mat)

    # --- if has bias
    if(self$has_bias){

      bias = self$bias
      # --- make bias PD
      if(self$is_bias_diagonal){
        bias = torch_sigmoid(bias)
        bias = torch_diag(bias)
      }
      else{
        bias = torch_crossprod(bias)
      }
      out = out + bias
    }

    return(out)

  }
)
