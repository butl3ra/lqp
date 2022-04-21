nn_garch <- torch::nn_module(
  classname = "nn_garch",
  # --- init
  initialize = function(in_features,
                        alpha_order = 1,
                        beta_order = 1,
                        buffer = 0,
                        n_ahead = 1,
                        ...) {
    # --- nn_parameters
    alpha = torch_runif(alpha_order,in_features,min=-4,max=-1)
    alpha = nn_parameter(alpha)
    beta = torch_runif(beta_order,in_features,min=0,max=4)
    beta = nn_parameter(beta)
    omega = torch_runif(in_features,min = -8,max = -6)
    omega = nn_parameter(omega)

    # --- self:
    self$in_features = in_features
    self$alpha = alpha
    self$beta = beta
    self$omega = omega
    self$buffer = torch_relu(buffer)
    self$n_ahead = n_ahead

  },
  # --- forward:
  forward = function(x,
                     ...) {

    # --- constraint hacking
    # --- alpha, beta and omega must be less than 1

    alpha = torch_sigmoid(self$alpha)
    beta = torch_sigmoid(self$beta)
    omega = torch_sigmoid(self$omega)

    # --- compute garch estimates
    sigma = torch_garch(x = x,
                        alpha = alpha,
                        beta = beta,
                        omega = omega,
                        buffer = self$buffer,
                        n_ahead = self$n_ahead)

    return(sigma)

  }
)
