#CHANGE THE PATH
setwd("...Otto-Group/code")

source("functions.R")
library(plyr)
library(dplyr)
library(caret)

#Load train data
train.set = read.csv(file='../input/train.csv',head=TRUE,sep=",",stringsAsFactors=F)
#randomize order
set.seed(1337)
train.set <- train.set[sample(nrow(train.set)),]

#new features
X <- select(train.set, feat_1:feat_93)
X$rowSum <- rowSums(X)
X$maxVal <- apply(X[,1:93], 1, max)
X$zeros <- apply(X[,1:93],1,function(x) sum(x==0))
  X$maxValDIVZeros <-X$maxVal/X$zeros

y <- train.set[,c("id","target")]

#data.frame that will hold the results
Y_xB <- data.frame(y$id, NA, NA, NA, NA, NA, NA, NA, NA, NA)
colnames(Y_xB) <- c("id", levels(as.factor((train.set$target))))

# Create folds
set.seed(117)
folds <- createFolds(y$target, k=10, list=T) #10 fold cross validation

foldIndex <- 0;

# Training
for(i in 1:length(folds)){
  
  print(paste0("i: ", i))
  
  #train and test data for the current fold
  fold.X.train <- X[-folds[[i]],]
  fold.X.test <- X[folds[[i]],]
  
  fold.y.train <- y[-folds[[i]],]
  fold.y.test <- y[folds[[i]],]
  
  #target rows for this fold
  trgtRows <- c((foldIndex+1):(foldIndex+length(folds[[i]])))
  
  ##TRAINING AND PREDICTION
  #set up the ids for the results
  Y_xB[trgtRows, "id"] <- fold.y.test[,"id"]
  Y_xB[trgtRows, -1] <- mxgboost(data.frame(fold.X.train, fold.y.train$target), data.frame(fold.X.test))
  
	#next fold
  foldIndex <- foldIndex + length(folds[[i]])  
}

#order the results
Y_xB <- arrange(Y_xB, id)

bd.val <- X
bd.val$target <- y[,"target"]

# splits the factor into dummy variables (one hot encoding)
dummy.formula <- dummyVars(~ target, data=bd.val, levelsOnly=TRUE)
actualClasses <- predict(dummy.formula,bd.val)

y_f <- Y_xB

#Cross validation score
local.score <- LogLoss((actualClasses), (as.matrix(sapply(y_f[,-1], as.numeric))))

#Score per class
index_C1 <- y[,"target"] == "Class_1"
index_C2 <- y[,"target"] == "Class_2"
index_C3 <- y[,"target"] == "Class_3"
index_C4 <- y[,"target"] == "Class_4"
index_C5 <- y[,"target"] == "Class_5"
index_C6 <- y[,"target"] == "Class_6"
index_C7 <- y[,"target"] == "Class_7"
index_C8 <- y[,"target"] == "Class_8"
index_C9 <- y[,"target"] == "Class_9"

ret <- vector();
ret[1] <- LogLoss((actualClasses[index_C1,]), (as.matrix(sapply(y_f[index_C1,-1], as.numeric))))
ret[2] <- LogLoss((actualClasses[index_C2,]), (as.matrix(sapply(y_f[index_C2,-1], as.numeric))))
ret[3] <- LogLoss((actualClasses[index_C3,]), (as.matrix(sapply(y_f[index_C3,-1], as.numeric))))
ret[4] <- LogLoss((actualClasses[index_C4,]), (as.matrix(sapply(y_f[index_C4,-1], as.numeric))))
ret[5] <- LogLoss((actualClasses[index_C5,]), (as.matrix(sapply(y_f[index_C5,-1], as.numeric))))
ret[6] <- LogLoss((actualClasses[index_C6,]), (as.matrix(sapply(y_f[index_C6,-1], as.numeric))))
ret[7] <- LogLoss((actualClasses[index_C7,]), (as.matrix(sapply(y_f[index_C7,-1], as.numeric))))
ret[8] <- LogLoss((actualClasses[index_C8,]), (as.matrix(sapply(y_f[index_C8,-1], as.numeric))))
ret[9] <- LogLoss((actualClasses[index_C9,]), (as.matrix(sapply(y_f[index_C9,-1], as.numeric))))

#Display cross validation results
local.score
ret

#save predictions as csv. Will be used for calibration
write.csv(Y_xB, file = "csvs/10foldCV_xB.csv", row.names = FALSE)

##TRAIN AND PREDICT TEST SET
test.set <- read.csv(file='../input/test.csv',head=TRUE,sep=",",stringsAsFactors=F)

X.test <- select(test.set, feat_1:feat_93)

#New features
X.test$rowSum <- rowSums(X.test)
X.test$maxVal <- apply(X.test[,1:93], 1, max)
X.test$zeros <- apply(X.test[,1:93],1,function(x) sum(x==0))
X.test$maxValDIVZeros <-X.test$maxVal/X.test$zeros

y.train <- as.factor(select(train.set, target)[,1])

#train/predict xgBoost
Y_xB <- mxgboost(data.frame(X, y[,2]), data.frame(X.test))

#formate submission file and save
m_values <- data.frame(test.set$id, Y_xB)
colnames(m_values) <- c("id", levels(y.train))
m_values <- arrange(m_values, id)
write.csv(m_values, file = "csvs/submit_xB.csv", row.names = FALSE)
