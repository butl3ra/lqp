nn_negative <- torch::nn_module(
  classname = "nn_negative",
  # --- init
  initialize = function(...){
  # -- empty

  },
  # --- forward
  forward = function(x,...) {
    return( -x )
  }
)
