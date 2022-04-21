# --- tic/toc structure like pracma
.lqpEnv <- new.env()
assign("elapsedTime", 0, envir = .lqpEnv)


.onLoad <-function(libname,
                   pkgname)
{
  library('torch')
  lqp_inst_root = system.file("",package = 'lqp')
  folders = list.files(lqp_inst_root,full.names=T)
  idx = grepl("inst__",folders)
  folders = folders[idx]
  files = list.files(folders,full.names=T)
  for(tmp in files){
    source(tmp)
  }
  rm('tmp')

  # --- set options for modules:
  options(rerun = FALSE)

  #default_dtype = torch_float64()
  #default_dtype = torch_float()
  #torch_set_default_dtype(default_dtype)

  environment(.lqpEnv) <- asNamespace("lqp")

}
