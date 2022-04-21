nn_exp <- torch::nn_module(
  classname = "nn_exp",
  # --- init
  initialize = function(...) {


  },
  # --- forward:
  forward = function(x,...) {

    # --- perform exponential
    return( torch_exp(x) )

  }
)
