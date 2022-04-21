nn_linear_single <- torch::nn_module(
  classname = "nn_linear_single",
  # --- init:
  initialize = function(in_features,
                        bias = F,
                        ...) {
    self$in_features = in_features
    self$has_bias = bias
    self$weight <- nn_parameter(torch_randn(1, in_features))
    if(bias){
      self$bias <- nn_parameter(torch_randn(in_features))
    }

  },
  # --- forward:
  forward = function(x,...) {
    # ---  y = weight*x + b
    out = self$weight$mul(x)
    if(self$has_bias){
      out = out + self$bias
    }
    return(out)
  }
)
