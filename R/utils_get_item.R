#' @export
get_item<-function(object,
         item,
         default=NULL)
{
  for(i in item){
    object = object[[i]]
  }
  if(is.null(object)){
    object = default
  }
  return(object)
}

#' @export
get_item_list<-function(l,
                         item,
                         default=NULL)
{
  out = lapply(l,function(x) get_item(x,item,default))
  return(out)
}
