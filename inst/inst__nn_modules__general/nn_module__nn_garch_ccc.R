nn_garch_ccc <- torch::nn_module(
  classname = "nn_garch_ccc",
  # --- init:
  initialize = function(in_features,
                        alpha_order = 1,
                        beta_order = 1,
                        buffer = 0,
                        n_ahead = 1,
                        center = TRUE,
                        static_cor_mat = TRUE,
                        cor_mat = NULL,
                        idx = NULL,
                        ...)
  {
    # --- garch model:
    self$garch_model = nn_garch(in_features = in_features,
                                alpha_order = alpha_order,
                                beta_order = beta_order,
                                buffer = buffer,
                                n_ahead = n_ahead)

    # --- correlation specs:
    self$cor_mat = cor_mat
    self$center = center
    self$static_cor_mat = static_cor_mat
    self$idx = idx

  },
  # forward:
  forward = function(x,rerun = getOption('rerun'),...) {

    # --- check for correlation
    has_cor = !is.null(self$cor_mat)
    static_cor_mat = self$static_cor_mat
    do_re_compute = !has_cor | rerun | !static_cor_mat

    # --- compute sigmas:
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
      self$cor_mat = cor_mat
    }

    # --- compute covariance
    covar = torch_cor2cov(cor_mat = self$cor_mat,sigma = sigma)

    # --- indexing:
    if(!is.null(self$idx)){
      covar = covar[self$idx,,]
    }

    return(covar)


  }
)
