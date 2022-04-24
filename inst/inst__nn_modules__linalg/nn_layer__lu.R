lu_layer <- torch::autograd_function(
  forward = function(ctx,
                     A,
                     b,
                     A_lu = NULL,
                     ...)
  {
    #######################################################################
    # Differentiable layer for LU factorization and solve:
    #######################################################################
    if(is.null(A_lu)){
      with_no_grad({A_lu = torch_lu(A)})
    }
    A_data = A_lu[[1]]
    A_pivots = A_lu[[2]]
    x = torch_lu_solve(b,A_data,A_pivots)

    # --- caching for backward pass

    ctx$save_for_backward(A_lu = A_lu, x = x)

    return(x)

  },

  backward = function(ctx,
                      dl_dx)
  {
    # --- note: if this is to be general purpose function then
    #           eventually need to generalize for potential A not symmetric

    # --- unpack saved variables
    saved_vars = ctx$saved_variables
    A_lu = saved_vars$A_lu
    x = saved_vars$x
    A_data = A_lu[[1]]
    A_pivots = A_lu[[2]]

    dim_x = dim(x)
    if(length(dim_x) < 2){
      xt = x$t()
    }
    else{
      xt = torch_transpose(x,2,3)
    }
    # --- solve: note A transpose needed if A not symmetric
    dx = torch_lu_solve(-dl_dx,A_data,A_pivots)
    dl_db = -dx
    dl_dA = torch_matmul(dx,xt)

    grads = list(A = dl_dA,
                 b = dl_db)

    return(grads)



  }

)
