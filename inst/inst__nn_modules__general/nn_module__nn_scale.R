nn_scale <- torch::nn_module(
  classname = "nn_scale",
  # --- init
  initialize = function(dim = 2,
                        normalize = TRUE,
                        center = FALSE,
                        scale_factor = 1,
                        ...) {
    # --- spec
    self$dim = dim
    self$normalize = normalize
    self$center = center
    self$scale_factor = scale_factor

  },
  # --- forward:
  forward = function(x,...) {

    # --- perform normalization
    out =  torch_scale(x,
                       dim = self$dim,
                       normalize = self$normalize,
                       center = self$center,
                       scale_factor = self$scale_factor)

    return(out)

  }
)
