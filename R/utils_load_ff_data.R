
load_experiment_data<-function(main_file = 'main.json',
         universe_file = 'proto_markets.json',
         main_dir = paste0(dir_experiment_config(),'experiments/a_mvo_constant_covar/a_uncon_uni/'),
         universe_dir = paste0(dir_experiment_config(),'config/universe_spec/'),
         load_master = TRUE)
{

  #*****************************************************************
  #boilerplate
  #*****************************************************************
  if(load_master){
    ResolveTaskManager::source.master.file()
  }

  #*****************************************************************
  #load universe and main config
  #*****************************************************************
  config_file = paste0(main_dir,main_file)
  uni_file = paste0(universe_dir,universe_file)

  uni_config = read_universe_config(uni_file)
  miner_list = miner::miner_preprocess(config_file=config_file,
                                       batch_list_item = uni_config)

  y_mat = miner_list$y_mat
  x_mat = miner_list$x_mat

  xy =  miner::xy_na_handle(x_mat = x_mat,y_mat = y_mat)
  x_mat = xy$x_mat
  y_mat = xy$y_mat


  lst = get_y_mat_prices_returns(y_mat,token = 'response')
  out = list()
  out$x = as.xts(x_mat)
  out$y = as.xts(lst$y)
  out$prices = as.xts(lst$prices)
  out$ret = as.xts(lst$ret)

  config = miner_list$config
  #out = do_feature_post_transform(config,out)

  return(out)

}


get_y_mat_prices_returns<-function(y_mat,
         token = 'response')
{
  col_names = colnames(y_mat)
  idx = grepl(token,col_names)
  n_assets = length(idx[idx])
  price_idx = 1:n_assets
  ret_idx = price_idx+n_assets
  prices = y_mat[,price_idx,drop=F]
  ret = y_mat[,ret_idx,drop=F]
  y = y_mat[,idx,drop=F]

  new_names = sub.before("_",colnames(y))
  colnames(prices) = new_names
  colnames(ret) = new_names
  colnames(y) = new_names

  out = list(y = y,
             prices = prices,
             ret = ret)
  return(out)

}

read_universe_config<-function(uni_file)
{
  lst = jsonlite::read_json(uni_file,simplifyVector = T)
  lst = purrr::transpose(lst)
  lst = lapply(lst,unlist)
  return(lst)
}

dir_experiment_config<-function()
{
  paste0("~/workspace/","experiment_config/")
}

#' @export
load_ff_data<-function( dir = "/Users/WOPR/Dropbox (Personal)/workspace_phd/data/",
                        x_file = 'ff_5_prices.csv',#ff_5_prices.csv
                        y_file = 'stock_prices.csv',
                        dates = '1990::2020-10',
                        freq = 'weeks')
{

  library('xts')

  x = read.csv(paste0(dir,x_file),header=T)
  y = read.csv(paste0(dir,y_file),header=T)

  x = as.xts(x[,-1],order.by = as.Date(x[,1]))
  y = as.xts(y[,-1],order.by = as.Date(y[,1]))

  # -- cleanup:
  x[] = ifna_prev_mat(x)
  y[] = ifna_prev_mat(y)

  date_int = intersect(as.character(index(x)),as.character(index(y)))
  date_int = as.Date(date_int)
  x = x[date_int,]
  y = y[date_int,]

  x = x[dates,]
  y = y[dates,]

  idx = endpoints(x,freq)
  x = x[idx,]
  y = y[idx,]

  # --- make y returns:
  y = roc(y)
  y = y[-1,]

  # --- make x returns:
  if(x_file == "ff_5_prices.csv"){
    x = roc(x)
  }
  x = x[-1,]

  data = list(x = x,
              y = y)
  return(data)
}

#' @export
roc<-function(x,
         nlag = 1
)
{
  y=x/mlag(x,nlag)-1
}

#' @export
mlag<-function(
  m,
  nlag = 1
)
{
  if( is.null(dim(m)) ) {
    n = len(m)
    if(nlag > 0) {
      m[(nlag+1):n] = m[1:(n-nlag)]
      m[1:nlag] = NA
    } else if(nlag < 0) {
      m[1:(n+nlag)] = m[(1-nlag):n]
      m[(n+nlag+1):n] = NA
    }
  } else {
    n = nrow(m)
    if(nlag > 0) {
      m[(nlag+1):n,] = m[1:(n-nlag),]
      m[1:nlag,] = NA
    } else if(nlag < 0) {
      m[1:(n+nlag),] = m[(1-nlag):n,]
      m[(n+nlag+1):n,] = NA
    }
  }
  return(m);
}

#' @export
ifna_prev<-function(x)
{
  x1 = !is.na(x)
  x1[1]=T
  return( x[cummax( (1:length(x)) * x1 )]	)
}

#' @export
ifna_prev_mat<-function(x)
{
  x[] = apply(x,2,ifna_prev)
  return(x)
}

#' @export
load_ff_data_dep<-function( dir = "/Users/WOPR/Dropbox (Personal)/workspace_phd/data/",
                        x_file = 'ff_factors_5.csv',
                        y_file = 'stock_prices.csv',
                        frequency = 'weeks')
{

  idx = 1:5
  x = miner::read_xts_csv(paste0(dir,x_file),header=T)[,idx]
  y = miner::read_xts_csv(paste0(dir,y_file),header=T)
  y = y[zoo::index(x),]
  x = x[zoo::index(y),]
  ix = xts::endpoints(y,frequency)
  ix = ix[ix>0]
  x = x[ix,]
  y = y[ix,]
  y = na.omit(y/sit::mlag(y,1)-1)
  x = x[-1,]

  data = list()
  data$x = x
  data$y = y
  return(data)
}


#' @export
window.ema<-function(l,
         threshold = 0.975)
{
  n = (seq(max(10/(1-l),250)-1,0,-1))
  w = (1-l)*l^(n)
  w = w/sum(w)
  cs = cumsum(rev(w))
  lookback = min(which(cs > threshold))
  out = tail(w,lookback)
  out = out/sum(out)
  return(out)
}

#' @export
roll_cov<-function(x,
         weights = window.ema(0.94),
         eps = 10^-4,
         min_obs = 2,
         do_fill = TRUE,
         ...)
{
  n_obs = nrow(x)
  n_x = ncol(x)
  width = length(weights)
  if(is.null(min_obs)){
    min_obs = min(n_x + 2, width)
  }
  cov_mat = roll::roll_cov(data = x, width, weights = weights,min_obs = min_obs,...)
  cov_mat = aperm(cov_mat,c(3,1,2))

  # --- Fill:
  if(do_fill){
    idx = 1:(min_obs-1)
    for(i in idx){
      cov_mat[i,,] = cov_mat[min_obs,,]
    }
  }

  # --- do regularization
  if(eps > 0){
    I = diag(n_x)
    for(i in 1:n_obs){
      cov_mat[i,,] = cov_mat[i,,] + eps*I
    }
  }

  return(cov_mat)

}
