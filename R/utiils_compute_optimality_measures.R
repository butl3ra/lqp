#' @export
compute_slacks<-function(z,
                         G,
                         h)
{
  slacks = h - torch_matmul(G,z)
  return(slacks)
}

#' @export
compute_duality_measure<-function(lams,
                                  slacks,
                                  method = 'median')
{
  mu = torch_crossprod(slacks,lams)
  mu = torch_abs(torch_reduce(mu,method))
  return(mu)
}

#' @export
compute_residual_primal<-function(z,
                                  slacks,
                                  A = NULL,
                                  b = NULL,
                                  G = NULL,
                                  h = NULL,
                                  method = 'median'
                                 )
{
  r_p_eq = 0
  if(!is.null(A)){
    r_p_eq = torch_matmul(A,z) - b
    r_p_eq = torch_norm(r_p_eq,dim=2)
  }
  r_p_ineq = 0
  if(!is.null(G)){
    r_p_ineq = torch_matmul(G,z) + slacks - h
    r_p_ineq = torch_norm(r_p_ineq,dim=2)
  }
  r_p = r_p_eq + r_p_ineq
  r_p = torch_reduce(r_p,method)
  return(r_p)
}


#' @export
compute_residual_primal_eq_core<-function(z,
                                          A = NULL,
                                          b = NULL)
{
  r_p_eq = 0
  if(!is.null(A)){
    r_p_eq = torch_matmul(A,z) - b
  }
  return(r_p_eq)
}

#' @export
compute_residual_primal_ineq_core<-function(z,
                                            slacks = NULL,
                                            G = NULL,
                                            h = NULL)
{
  r_p_ineq = 0
  if(!is.null(G)){
    if(is.null(slacks)){
      slacks = compute_slacks(z = z,
                              G = G,
                              h = h)
      slacks = torch_threshold_(slacks,10^-8,0)
    }
    r_p_ineq = torch_matmul(G,z) + slacks - h
  }
  return(r_p_ineq)
}



#' @export
compute_residual_dual<-function(z,
                                Q,
                                p,
                                lams,
                                nus,
                                A = NULL,
                                b = NULL,
                                G = NULL,
                                h = NULL,
                                method = 'median')
{
  r_d = torch_matmul(Q,z) + p
  if(!is.null(A)){
    r_d = r_d + torch_matmul(torch_transpose_batch(A),nus)
  }
  if(!is.null(G)){
    r_d = r_d + torch_matmul(torch_transpose_batch(G),lams)
  }
  r_d = torch_norm(r_d,dim=2)
  r_d = torch_reduce(r_d,method)
  return(r_d)
}

#' @export
compute_residual_dual_core<-function(z,
                                     Q,
                                     p,
                                     lams = NULL,
                                     nus = NULL,
                                     A = NULL,
                                     G = NULL)
{
  r_d = torch_matmul(Q,z) + p
  if(!is.null(A)){
    r_d = r_d + torch_matmul(torch_transpose_batch(A),nus)
  }
  if(!is.null(G)){
    r_d = r_d + torch_matmul(torch_transpose_batch(G),lams)
  }
  return(r_d)
}

