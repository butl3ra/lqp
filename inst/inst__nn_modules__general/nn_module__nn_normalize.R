nn_normalize <- torch::nn_module(
  classname = "nn_normalize",
  # --- init
  initialize = function(dim = 2,
                        p = 2,
                        normalize = TRUE,
                        scale_factor = 1,
                        eps = 10^-12,
                        ...) {
    # --- spec
    self$dim = dim
    self$p = p
    self$scale_factor = scale_factor
    self$eps = eps

  },
  # --- forward:
  forward = function(x,...) {

    # --- perform normalization
    x_normal = nnf_normalize(x,
                             p = self$p,
                            dim = self$dim,
                            eps = self$eps)
   return(x_normal*self$scale_factor)

  }
)
