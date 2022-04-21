nn_constant <- torch::nn_module(
  classname = "nn_constant",
  # --- init
  initialize = function(value,
                        ...){
    #holds a single constant
    self$constant = value

  },
  # --- forward
  forward = function(...) {
    self$constant
  }
)
