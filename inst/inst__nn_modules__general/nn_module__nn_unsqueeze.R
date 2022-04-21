nn_unsqueeze <- torch::nn_module(
  classname = "nn_unsqueeze",
  # --- init
  initialize = function(dim = 2,
                        ...) {
    # --- spec
    self$dim = dim

  },
  # --- forward:
  forward = function(x,...) {

    # --- perform unsqueeze
    x = x$unsqueeze(self$dim)
    return(x)

  }
)
