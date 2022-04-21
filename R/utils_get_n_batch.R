#' @export
get_n_batch_core<-function(x,
                               dim_batch)
{
  if(is_null(x)){
    n_batch = 0
  }
  else{
    dim_x = dim(x)
    len_dim_x = length(dim_x)
    dim_diff = dim_batch - len_dim_x
    if(dim_diff == 1){
      n_batch = 1
    }
    else if(dim_diff == 0){
      n_batch = dim_x[1]
    }
    else if(dim_x == 0){
      n_batch = 0
    }
    else{
      msg = sprintf("dimension of x: %s is inconsistent with required dimension: %s",dim_x,dim_batch)
      stop(msg)
    }
  }

  return(n_batch)
}

#' @export
get_n_batch<-function(...,
                      dims = 3
)
{
  # --- unpack:
  params = list(...)

  if(length(dims) == 1){
    dims = rep(dims,length(params))
    names(dims) = names(params)
  }
  dims = dims[names(params)]

  n_batch = Map(get_n_batch_core,x = params, dim_batch = dims)
  n_batch = unlist(n_batch)

  return(n_batch)
}
