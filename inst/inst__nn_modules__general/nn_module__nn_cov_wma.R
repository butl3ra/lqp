nn_cov_wma <- torch::nn_module(
  classname = "nn_cov_wma",
  # --- init
  initialize = function(in_features,
                        weight = NULL,
                        center = TRUE,
                        normalize = FALSE,
                        weight_as_nn_parameter = FALSE,
                        ...){
    # --- centering and normalization spec
    self$in_features = in_features
    self$center = center
    self$normalize = normalize

    # ---  weight:
    if(weight_as_nn_parameter){
      if(is.null(weight)){
        weight = torch_runif(in_features,min=-3,max = 0)
      }
      weight = nn_parameter(weight)
    }
    self$weight = weight

  },
  # --- forward
  forward = function(x,rerun = getOption('rerun'),...) {
    has_cov = !is.null(self$covar)
    is_param = is_nn_parameter(self$weight)
    do_re_compute = !has_cov | is_param | rerun

    # --- ensure positivity and sum to 1
    weight = self$weight
    if(is_param){
      weight = torch_exp(weight)
      weight = torch_sum_1(weight)
    }

    # --- re-compute covar
    if(do_re_compute){
      covar = torch_wcov(x,
                         weight = weight,
                         center = self$center,
                         normalize = self$normalize)
      self$covar = covar
    }
    else{
      covar = self$covar
    }

    return(covar)
  }
)
