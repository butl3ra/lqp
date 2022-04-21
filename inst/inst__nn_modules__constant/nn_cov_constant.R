nn_cov_constant <- torch::nn_module(
  classname = "nn_cov_constant",
  # --- init
  initialize = function(center = FALSE,
                        normalize = FALSE,
                        ...){
    # --- cov specs
    self$center = center
    self$normalize = normalize

  },
  # --- forward
  forward = function(x,...) {
    has_cov = !is.null(self$constant)
    # --- re-compute covar
    if(!has_cov){
      if(self$normalize){
        self$constant = torch_cor(x,center = self$center)
      }
      else{
        self$constant = torch_cov(x,center = self$center)
      }
    }
    return(self$constant)
  }
)
