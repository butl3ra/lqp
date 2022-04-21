#' @export
torch_mlag<-function(x,
                     shifts,
                     dims = 1)
{
  ld = length(dim(x))
  x = torch_roll(x,shifts = shifts,dims = dims)
  if(shifts < 0){
    idx0 = (nrow(x)+shifts)
    idx=idx0:nrow(x)
    if(ld == 2){
      for(i in idx){
        x[i,] = x[idx0,]
      }
    }
    if(ld == 3){
      for(i in idx){
        x[i,,] = x[idx0,,]
      }
    }
  }
  if(shifts > 0){
    idx0 = shifts
    idx=1:idx0
    if(ld == 2){
      for(i in idx){
        x[i,] = x[idx0,]
      }

    }
    if(ld == 3){
      for(i in idx){
        x[i,,] = x[idx0,,]
      }
    }
  }
  return(x)

}
