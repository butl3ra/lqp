#' @export
tic<-function(...)
{
  assign("elapsedTime", proc.time()[3], envir = .lqpEnv)
  invisible()
}

#' @export
toc<-function (echo = TRUE,
               ...)
{
  prevTime <- get("elapsedTime", envir = .lqpEnv)
  diffTimeSecs <- proc.time()[3] - prevTime
  if (echo) {
    cat(sprintf("elapsed time is %f seconds", diffTimeSecs),"\n")
    return(invisible(diffTimeSecs))
  }
  else {
    return(diffTimeSecs)
  }
}
