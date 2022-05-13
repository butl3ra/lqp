# --- load data:
data = load_ff_data(freq = 'weeks')
set.seed(1)
idx = sort(sample.int(n_y_all,n_y))
x = data$x
y = data$y[,idx]
n_y = ncol(y)
n_obs = nrow(y)
n_x = ncol(x)


# --- in-sample and out-of-sample:
idx_is = 1:1043
idx_oos = 1044:n_obs
y_is = as.matrix(y[idx_is,])
y_oos = as.matrix(y[idx_oos,])
x_is = as.matrix(x[idx_is,])
x_oos = as.matrix(x[idx_oos,])
y_is_t = as_torch_tensor(y_is)
y_oos_t = as_torch_tensor(y_oos)
x_is_t = as_torch_tensor(x_is)
x_oos_t = as_torch_tensor(x_oos)


# --- beta OLS:
m_ols_ff = lm(y_is~x_is+0)
bias_ols = coef(m_ols_ff)
beta_ols = coef(m_ols_ff)
beta_ols_t = as_torch_tensor(beta_ols)
resid = m_ols_ff$residuals
resid = diag(diag(cov(resid)))
rownames(beta_ols) = colnames(data$x)

# --- Q_x
Q_x = torch_cov(x_is_t)
Q_x = Q_x$unsqueeze(1)
Q_x = Q_x/sqrt(torch_diagonal(Q_x)$mean())

# --- Q_X
x_lag = mlag(data$x,1)
x_lag[1,] = x_lag[2,]
Q_x_roll = roll_cov(x_lag, weights = rep(1/52,52),min_obs = 10,eps = 0)
Q_x_roll_t = as_torch_tensor(Q_x_roll)
Q_all = torch_matmul(torch_matmul(beta_ols_t$t()$unsqueeze(1),Q_x_roll_t),beta_ols_t$unsqueeze(1))
Q_all = Q_all + resid
Q_is_t = Q_all[idx_is,,]
Q_oos_t = Q_all[idx_oos,,]

# --- scaling:
Q_is_t = Q_is_t*50
Q_oos_t = Q_oos_t*50

# --- setup solver control:
control = nn_qp_control(solver = 'con_l1',
                        tol_primal = 10^-4,
                        tol_dual = 10^-4,
                        max_iters = 1000,
                        rho = 0.20,
                        backprop = 'fixed_point')

# --- training specs:
learning_rate = 0.25
n_epochs = 100
start_value = -4

# --- p_model:
p_model = nn_constant(value = torch_zeros(c(1,n_y,1)))
# --- Q_model:
Q_model = nn_constant(value = Q_is_t)
# --- A_model and b_model:
A_model = nn_constant(value = torch_ones(c(1,1,n_y)))
b_model = nn_constant(value = torch_ones(c(1)))
# --- lb_model and ub_model:
lb_model = nn_constant(value = torch_zeros(c(1,n_y,1)))
ub_model = nn_constant(value = torch_ones(c(1,n_y,1),requires_grad = requires_grad))
# --- G_model and h_model:
G_model = nn_constant(value = -torch_eye(n_y)$unsqueeze(1))
h_model = nn_constant(value = lb_model())
# --- E_model and lambda_1:
E = nn_parameter(as_torch_tensor(diag(runif(n_y)),requires_grad = TRUE))
E_model = nn_sequential(nn_constant(value = E),
                        nn_unsqueeze(dim = 1),
                        nn_relu())
lambda_1 = nn_parameter(as_torch_tensor(start_value,requires_grad = TRUE))
lambda_1_model = nn_sequential(nn_constant(value = lambda_1), nn_exp())
# --- D_model and lambda_2:
D_model = nn_quad_form_const(in_features =  n_x, out_features=n_y, x_mat = Q_x)
lambda_2 = nn_parameter(as_torch_tensor(start_value,requires_grad = TRUE))
lambda_2_model = nn_sequential(nn_constant(value = lambda_2), nn_exp())

# --- nominal program:
control_nom = control
control_nom$solver='admm'
model = nn_qp(Q = Q_model(),
              p = p_model(),
              A = A_model(),
              b = b_model(),
              lb = lb_model(),
              ub = ub_model(),
              control = control_nom)
z_nom_is = model()
z_nom_is = torch_sum_1(z_nom_is,dim = 2)

model = nn_qp(Q = Q_oos_t,
              p = p_model(),
              A = A_model(),
              b = b_model(),
              lb = lb_model(),
              ub = ub_model(),
              control = control_nom)
z_nom_oos = model()
z_nom_oos = torch_sum_1(z_nom_oos,dim = 2)


# --- norm-penalized program:
model = nn_lqp(Q_model = Q_model,
               p_model = p_model,
               A_model = A_model,
               b_model = b_model,
               G_model = G_model,
               h_model = h_model,
               lb_model = lb_model,
               ub_model = ub_model,
               D_model = D_model,
               lambda_2_model = lambda_2_model,
               E_model = E_model,
               lambda_1_model = lambda_1_model,
               control = control)
optimizer <- optim_adam(model$parameters, lr = learning_rate)
#---------------- training loop -------------------
loss_hist = NULL
for (t in 1:n_epochs ) {
  # --- forward pass
  z = model(x = x_is_t)
  z = torch_sum_1(z,dim = 2)
  # --- compute loss
  loss <-  nnf_var_loss(z = z, y_is_t$unsqueeze(3))
  cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  loss_hist = c(loss_hist,loss$item())
  # --- backpropagation:
  optimizer$zero_grad()
  loss$backward()
  # ---  update params:
  optimizer$step()
}

# --- in_sample:
z_is = z
# --- out-of-sample
model = nn_qp(Q = Q_oos_t,
              p = p_model(),
              A = A_model(),
              b = b_model(),
              lb = lb_model(),
              ub = ub_model(),
              G = G_model(),
              h =  h_model(),
              E = E_model(),
              lambda_1 = lambda_1_model(),
              D = D_model(),
              lambda_2 = lambda_2_model(),
              control = control)
z_oos = model()
z_oos = torch_sum_1(z_oos,dim=2)



