#' @export
compute_dual_cert<-function(model,
                            method = 'median')
{
  m = model
  if(is.null(model$qp_solver)){
    m = model
  }
  else{
    m = model$qp_solver
  }
  nus = m$nus
  lams = m$lams
  slacks = m$slacks
  x = m$x
  n_x = get_n_x(x)
  A = m$A
  Q = m$Q
  p = m$p
  G = m$G
  h = m$h
  lb = m$lb
  ub = m$ub
  if(is.null(G)){
    G = torch_make_G_bound(n_x = n_x,
                           any_lb = T,
                           any_ub = T)
    h = torch_cat(list(-lb,ub),dim=2)
  }
  if(is.null(slacks)){
    slacks = compute_slacks(z = x,G = G,h = h)
  }
  lams = torch_threshold_(lams,10^-8,0)
  slacks = torch_threshold_(slacks,10^-8,0)
  dual_measure = compute_duality_measure(lams,slacks,method=method)
  resid_primal = compute_residual_primal(z = x,
                                         slacks = slacks,
                                         A = A,
                                         b = b,
                                         G = G,
                                         h = h,
                                         method = method)
  resid_dual = compute_residual_dual(z = x,
                                     Q = Q,
                                     p = p,
                                     lams = lams,
                                     nus = nus,
                                     A = A,
                                     b = b,
                                     G = G,
                                     h = h,
                                     method = method)
  out = list(dual_measure = dual_measure$item(),
             resid_primal =   resid_primal$item(),
             resid_dual = resid_dual$item()
             )
  return(out)
}


if(F){
nus = model$qp_solver$nus
lams = model$qp_solver$lams
slacks = model$qp_solver$slacks
if(is.null(slacks)){
  slacks = compute_slacks(z = z_oos_0,G = G,h = h)
}
lams = torch_threshold_(lams,10^-8,0)
slacks = torch_threshold_(slacks,10^-8,0)


dual_measure = compute_duality_measure(lams,slacks)
resid_primal = compute_residual_primal(z = z_oos_0,
                                       slacks = slacks,
                                       A = A_oos,
                                       b = b,
                                       G = G,
                                       h = h)
resid_dual = compute_residual_dual(z = z_oos_0,
                                   Q = Q,
                                   p = p,
                                   lams = lams,
                                   nus = nus,
                                   A = A_oos,
                                   b = b,
                                   G = G,
                                   h = h)
}
