nn_linear_sparse <- torch::nn_module(
  classname = "nn_linear_sparse",
  # --- init
  initialize = function(in_features,
                        out_features = NULL,
                        P,
                        bias = FALSE,
                        ...) {

    # --- in and out features
    self$in_features = in_features
    if(!is_torch_tensor(P)){
      P = as_torch_tensor(P)
    }
    self$P = P
    self$weight <- nn_parameter(torch_randn( in_features,1 ) )
    if(bias){
      self$out_features = out_features
      self$bias <- nn_parameter(torch_randn( out_features ) )
    }

  },
  # t--- forward
  forward = function(x,...) {
    y_hats = torch_linear_sparse(x = x,
                                 P = self$P,
                                 weight = self$weight,
                                 bias = self$bias)
    return(y_hats)

  }
)
