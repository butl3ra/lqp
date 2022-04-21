#' @export
nnf_mvo_loss<-function(input,
                       target,
                       risk_aversion = 1,
                       ...)
{
  loss = -(input*target)$sum(2)
  loss$mean() + 0.5*risk_aversion*loss$var()
}


#' @export
nnf_mvo_covmat_loss<-function(input,
                              target,
                              model=NULL,
                              risk_aversion=1,
                              trade_lag = 2,
                              V = NULL,
                              ...)
{
  loss = -(input*target)$sum(2)
  if(is.null(model)){
    loss = loss$mean() + 0.5*risk_aversion*loss$var()
  }
  else{
    if(is.null(V)){
      Q = model$modules$Q_module(x = target)
    }
    else{
      Q = V
    }
    if(length(dim(Q)) > 2){
      if(trade_lag != 0){
        Q = torch_mlag(Q,-trade_lag)
      }
      p_var = torch_matmul(torch_matmul(input$unsqueeze(2),Q),input$unsqueeze(3))
      p_var = p_var$squeeze(2)
      p_var = p_var$squeeze(2)
    }
    else{
      p_var = (input * (input$mm(Q)) )$sum(2)
    }
    loss = loss + 0.5 * risk_aversion * p_var
    loss = loss$mean()

  }
  return(loss)
}
