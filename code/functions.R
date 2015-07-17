require(xgboost)
require(methods)
mxgboost <- function(train, test) {
  
  y = train[,ncol(train)]
  y = gsub('Class_','',y)
  y = as.integer(y)-1 #xgboost take features in [0,numOfClass)
  
  x = rbind(train[,-ncol(train)],test)
  x = as.matrix(x)
  x = matrix(as.numeric(x),nrow(x),ncol(x))
  trind = 1:length(y)
  teind = (nrow(train)+1):nrow(x)
  
  
  # Set parameters
  param <- list("objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = 9,
                "nthread" = 6,
                "max_depth" = 14,
                "gamma" = 0.8,
                "min_child_weight" = 10,
                "subsample" = 1,
                "colsample_bytree" = 0.6,
                "eta" = 0.0219) #0.4
  
  # Train the model
  nround = 1750 
  
  bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
  
  # Make prediction
  pred = predict(bst,x[teind,])
  pred = matrix(pred,9,length(pred)/9)
  pred = t(pred)
  
  # Output submission
  pred = format(pred, digits=2,scientific=F) # shrink the size of submission
  names(pred) = c(paste0('Class_',1:9))
  
  pred
}

# Logloss Function
LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -sum(actual*log(predicted))/nrow(actual)
}