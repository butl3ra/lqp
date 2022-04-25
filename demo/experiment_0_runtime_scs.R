# --- run_specs:
n_runs = 10
n_batch = 128
rho = 0.50
bound_range = c(1,2)
n_y_list = c(10,50,100,250,500,1000)

# --- spec list setup
spec_list_0 = list(
  admm_unrolled_cold_1 = list(solver = 'admm',
                              tol = 10^-1,
                              output_as_list = FALSE,
                              warm_start = FALSE,
                              max_iters = 1000,
                              unroll_grad = TRUE,
                              hybrid = FALSE),
  admm_unroll_cold_3 = list(solver = 'admm',
                            tol = 10^-3,
                            output_as_list = FALSE,
                            warm_start = FALSE,
                            max_iters = 1000,
                            unroll_grad = TRUE,
                            hybrid = FALSE),
  admm_unroll_cold_5 = list(solver = 'admm',
                            tol = 10^-5,
                            output_as_list = FALSE,
                            warm_start = FALSE,
                            max_iters = 1000,
                            unroll_grad = TRUE,
                            hybrid = FALSE),
  admm_fp_cold_1 = list(solver = 'admm',
                        tol = 10^-1,
                        output_as_list = FALSE,
                        warm_start = FALSE,
                        max_iters = 1000,
                        unroll_grad = FALSE,
                        backprop = 'fixed_point'),
  admm_fp_cold_3 = list(solver = 'admm',
                        tol = 10^-3,
                        output_as_list = FALSE,
                        warm_start = FALSE,
                        max_iters = 1000,
                        unroll_grad = FALSE,
                        backprop = 'fixed_point'),
  admm_fp_cold_5 = list(solver = 'admm',
                        tol = 10^-5,
                        output_as_list = FALSE,
                        warm_start = FALSE,
                        max_iters = 1000,
                        unroll_grad = FALSE,
                        backprop = 'fixed_point'
  ),
  admm_kkt_cold_1 = list(solver = 'admm',
                         tol = 10^-1,
                         output_as_list = FALSE,
                         warm_start = FALSE,
                         max_iters = 1000,
                         unroll_grad = FALSE,
                         backprop = 'kkt'),
  admm_kkt_cold_3 = list(solver = 'admm',
                         tol = 10^-3,
                         output_as_list = FALSE,
                         warm_start = FALSE,
                         max_iters = 1000,
                         unroll_grad = FALSE,
                         backprop = 'kkt'),
  admm_kkt_cold_5 = list(solver = 'admm',
                         tol = 10^-5,
                         output_as_list = FALSE,
                         warm_start = FALSE,
                         max_iters = 1000,
                         unroll_grad = FALSE,
                         backprop = 'kkt'),
  optnet_cold_1 = list(solver = 'int',
                       tol = 10^-1,
                       output_as_list = FALSE,
                       warm_start = FALSE,
                       max_iters = 12,
                       unroll_grad = FALSE,
                       backprop = 'kkt'),
  optnet_cold_3 = list(solver = 'int',
                       tol = 10^-3,
                       output_as_list = FALSE,
                       warm_start = FALSE,
                       max_iters = 12,
                       unroll_grad = FALSE,
                       backprop = 'kkt'),
  optnet_cold_5 = list(solver = 'int',
                       tol = 10^-5,
                       output_as_list = FALSE,
                       warm_start = FALSE,
                       max_iters = 12,
                       unroll_grad = FALSE,
                       backprop = 'kkt'),
  scs_fp_cold_1 = list(solver = 'scs',
                       tol = 10^-1,
                       output_as_list = FALSE,
                       warm_start = FALSE,
                       max_iters = 1000,
                       unroll_grad = FALSE,
                       backprop = 'fixed_point'),
  scs_fp_cold_3 = list(solver = 'scs',
                       tol = 10^-3,
                       output_as_list = FALSE,
                       warm_start = FALSE,
                       max_iters = 1000,
                       unroll_grad = FALSE,
                       backprop = 'fixed_point'),
  scs_fp_cold_5 = list(solver = 'scs',
                       tol = 10^-5,
                       output_as_list = FALSE,
                       warm_start = FALSE,
                       max_iters = 1000,
                       unroll_grad = FALSE,
                       backprop = 'fixed_point'
  )
)

n_y = n_y_list[1]
spec_idx = (1:15)
spec_list =spec_list_0[spec_idx]

# --- main loop:
out = list()
run_idx = 1:n_runs
for(run in run_idx){
  seed = run
  set.seed(seed)
  torch_manual_seed(seed)
  cat('run: ',run, '\n')
  n_z = n_y
  n_ineq = n_samples =  2*n_z
  n_eq = 1

  # --- Q:
  L = torch_randn(n_batch,n_samples,n_z)
  Q = torch_crossprod(L) + 10^-3*torch_eye(n_z,n_z)$unsqueeze(1)
  Q = Q/n_samples

  # --- p:
  p = torch_randn(n_batch,n_z,1,requires_grad=TRUE)

  # --- A:
  A = torch_ones(c(n_batch,1,n_y))

  # --- b:
  b = torch_ones(c(n_batch,1,1))

  # --- lb:
  lb = -torch_runif(c(1,n_y,1),min=bound_range[1],max=bound_range[2])
  lb = prep_batch_size(lb,n_batch)

  # --- ub:
  ub = torch_runif(c(1,n_y,1),min=bound_range[1],max=bound_range[2])
  ub = prep_batch_size(ub,n_batch)

  # --- G_model:
  G0 = torch_eye(n_y)$unsqueeze(1)
  G = torch_cat(list(-G0,G0),dim=2)
  G = prep_batch_size(G,n_batch)

  # --- h_model:
  h = torch_cat(list(-lb,ub),dim=2)

  # --- dl_dz
  dl_dz = torch_ones(c(n_batch,n_z,1))

  # --- inner loop:

  out_list = list()
  idx = 1:length(spec_list)
  for(i in idx){
    cat('inner loop: ',i,'\n')
    solver = spec_list[[i]]$solver
    control = nn_qp_control(solver = solver,
                            output_as_list = spec_list[[i]]$output_as_list,
                            tol_primal = spec_list[[i]]$tol,
                            tol_dual = spec_list[[i]]$tol,
                            tol = spec_list[[i]]$tol,
                            warm_start = spec_list[[i]]$warm_start,
                            unroll_grad = spec_list[[i]]$unroll_grad,
                            max_iters = spec_list[[i]]$max_iters,
                            rho = rho,
                            backprop = spec_list[[i]]$backprop
    )

    # --- make sure we get grads for everything:
    Q = as_torch_tensor(as_array(Q),requires_grad = T)
    p = as_torch_tensor(as_array(p),requires_grad = T)
    A = as_torch_tensor(as_array(A),requires_grad = T)
    b = as_torch_tensor(as_array(b),requires_grad = T)
    {if(grepl("admm",solver)){
      G = as_torch_tensor(as_array(G),requires_grad = F)
      h = as_torch_tensor(as_array(h),requires_grad = F)
    }
    else{
      G = as_torch_tensor(as_array(G),requires_grad = T)
      h = as_torch_tensor(as_array(h),requires_grad = T)
    }
    }
    lb = as_torch_tensor(as_array(lb),requires_grad = T)
    ub = as_torch_tensor(as_array(ub),requires_grad = T)


    cat('model: ',names(spec_list)[i],'\n')
    model = nn_qp(Q = Q,
                  p = p,
                  A = A,
                  b = b,
                  G = G,
                  h = h,
                  lb = lb,
                  ub = ub,
                  control = control)
    # --- forward pass
    tic()
    z = model()
    forward_time = toc(echo=F)

    # --- backward pass
    tic()
    z$backward(dl_dz)
    backward_time = toc(echo=F)

    # --- total time
    total_time = forward_time + backward_time
    cat('total_time: ', total_time,'\n')

    # ---storage
    out_list[[i]]= list(runtime = total_time,
                        forward_time = forward_time,
                        backward_time = backward_time)

    gc()

  }

  names(out_list) = names(spec_list)

  out[[run]] = out_list


}

# --- write output
test = purrr::transpose(out)

# --- write runtime
runtime = lapply(test,get_item_list,item='runtime')
runtime = lapply(runtime,do.call,what=rbind)
runtime = do.call(cbind,runtime)
colnames(runtime) = names(test)
barplot(colMeans(runtime))



# --- write forward time
forward_time = lapply(test,get_item_list,item='forward_time')
forward_time = lapply(forward_time,do.call,what=rbind)
forward_time = do.call(cbind,forward_time)
colnames(forward_time) = names(test)
barplot(colMeans(forward_time))



# --- chart backward time
backward_time = lapply(test,get_item_list,item='backward_time')
backward_time = lapply(backward_time,do.call,what=rbind)
backward_time = do.call(cbind,backward_time)
colnames(backward_time) = names(test)
barplot(colMeans(backward_time))






