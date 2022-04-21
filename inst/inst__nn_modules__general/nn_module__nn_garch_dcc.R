nn_garch_dcc <- torch::nn_module(
  classname = "nn_garch_dcc",
  # --- init:
  initialize = function(in_features,
                        alpha_order = 1,
                        beta_order = 1,
                        buffer = 0,
                        n_ahead = 1,
                        center = TRUE,
                        static_cor_mat = TRUE,
                        ...)
  {

    # --- correlation parameters:
    a = torch_runif(1,min=-4,max=-1)
    a = nn_parameter(a)
    b = torch_runif(1,min=0,max=4)
    b = nn_parameter(b)

    # --- garch model:
    self$garch_model = nn_garch(in_features = in_features,
                                alpha_order = alpha_order,
                                beta_order = beta_order,
                                buffer = buffer,
                                n_ahead = n_ahead)

    # --- attach values:
    self$a = a
    self$b = b
    self$n_ahead = n_ahead
    self$buffer = buffer
    self$center = center
    self$static_cor_mat = static_cor_mat

  },
  # --- forward:
  forward = function(x,...) {

    # --- check for correlation
    has_cor = !is.null(self$cor_mat_bar)
    static_cor_mat = self$static_cor_mat
    do_re_compute = !has_cor | rerun | !static_cor_mat

    # --- compute garch sigmas:
    sigma = self$garch_model(x)

    # --- re-compute correlation
    if(do_re_compute){
      if(static_cor_mat){
        x_sigma = x
      }
      else{
        # --correction for bias in garch correlation
        x_sigma = x/sigma
      }
      cor_mat = torch_cor(x = x_sigma,center = self$center)
      self$cor_mat_bar = cor_mat
    }

    # --- constraint hack:
    a = torch_sigmoid(self$a)
    b = torch_sigmoid(self$b)

    #compute dynamic covariance:
    covar = torch_garch_dcc(x = x,
                            sigma = sigma,
                            cor_mat_bar = self$cor_mat_bar,
                            a = a,
                            b = b,
                            buffer = self$buffer,
                            n_ahead = self$n_ahead)

    return(covar)

  }
)
