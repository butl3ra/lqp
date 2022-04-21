#' @export
torch_reshape_mat<-function(mat,
                            forward = TRUE)
{
  dm = dim(mat)
  if(forward){
    mat = mat$reshape(c(dm[1],dm[2]*dm[3],1))
    #attr(mat,'original_shape') = dm
  }
  else{
    dm_23 = sqrt(dm[2])
    #dm = attributes(mat)$original_shape
    mat = mat$reshape(c(dm[1],dm_23,dm_23))
  }
  return(mat)
}



#' @export
torch_reshape_mat_admm<-function(mat,
                                 forward = TRUE)
{
  dm = dim(mat)
  if(forward){
    mat = mat$reshape(c(dm[1],dm[2]*dm[3],1))
  }
  else{
    mat = mat$reshape(c(dm[1],dm[2],dm[3]))
  }
  return(mat)
}

#' @export
torch_reshape_U_Q<-function(U_Q,
                            n_batch,
                            n_x,
                            forward = TRUE)
{
  if(forward){
    U_Q = U_Q$reshape(c(n_batch,n_x*n_x,1))
  }
  else{
    U_Q = U_Q$reshape(c(n_batch,n_x,n_x))
  }
  return(U_Q)

}

#' @export
torch_reshape_U_S<-function(U_S,
                            n_batch,
                            n_con,
                            forward = TRUE)
{
  if(forward){
    U_S = U_S$reshape(c(n_batch,n_con*n_con,1))
  }
  else{
    U_S = U_S$reshape(c(n_batch,n_con,n_con))
  }
  return(U_S)
}

#' @export
torch_reshape_U_S<-function(U_S,
                            n_batch,
                            n_ineq,
                            forward = TRUE)
{
  if(forward){
    U_S = U_S$reshape(c(n_batch,n_ineq*n_ineq,1))
  }
  else{
    U_S = U_S$reshape(c(n_batch,n_con,n_con))
  }
  return(U_S)
}


