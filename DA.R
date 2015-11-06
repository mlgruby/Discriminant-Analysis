variance <- function(x, yLabel, method){
  # returns variance of a vector or covariance of a matrix
  # takes one argument x which can be a vector or matrix (numeric)
  # example varReturn(iris[,1]) or varReturn(iris[, -5])
  if(ncol(x) == 2) return(varReturn(x, yLabel, method)) else return(covReturn(x, yLabel, method))
}

covReturn <- function(x, yLabel, method){
  # returns covariance matrix 
  # takes two argument x as training Data Frame (Including label) and 
  # yLabel(Character) which is column name of label variable
  # example covReturn(iris, 'V5', 'lda')
  # calculation : Pr1(Y1)*X1 + Pr1(Y2)*X2 + ...
  colNum <- which(names(x) == yLabel)
  probDF <- piK(x[,colNum])
  covMat <- matrix(0, nrow = ncol(x[, -colNum]), ncol = ncol(x[, -colNum]))
  if (method == 'lda'){
    for (i in 1:nrow(probDF)){
      covMat <- covMat + probDF[i,2]*cov(x[x[,colNum] == probDF[i,1], -colNum])
    }  
  } else {
    covMat <- cov(x[, -colNum])
  }  
  return(covMat)
}

varReturn <- function(x, yLabel, method){
  # returns variance of a vector
  # takes two argument x as training Data Frame (Including label) and 
  # yLabel(Character) which is column name of label variable
  # example varReturn(iris[,c(1,5)], 'V5')
  # calculation : Pr1(Y1)*var(X1) + Pr1(Y2)*var(X2) + ...
  colNum <- which(names(x) == yLabel)
  probDF <- piK(x[,colNum])
  featureVar <- 0
  if(method == 'lda'){
    for (i in 1:nrow(probDF)){
      featureVar <- featureVar + probDF[i,2]*var(x[x[, colNum] == probDF[i,1],-colNum])
    }  
  } else {
    featureVar <- var(x[, -colNum])
  }
  return(featureVar)
}

piK <- function(yLabel){
  # returns a data frame
  # containing two columns, 'Label' and 'prob'
  # each row includes label and its coreesponding probability
  # example piK(iris[,5])
  piKDF <- data.frame(Label = as.numeric(0), prob = as.numeric(0))
  idx <- 1
  for (i in unique(yLabel)){
    piKDF[idx,1] <- i
    piKDF[idx,2] <- length(which(yLabel == i))/length(yLabel) # probability for each label
    idx <- idx+1
  }
  return(piKDF)
}

muK <- function(x, yi, colNum){
  # returns mean value for a matrix or for a vector
  if(ncol(x) == 2) return(mean(x[x[,colNum] == yi, -colNum])) else return(as.numeric(apply(x[x[, colNum] == yi, -colNum], 2, mean)))
}

ldaCalc <- function(test, train, yLabel, yk, commonVar){
  # returns a predicted label for a tes data point
  # takes five different argument
  # test datapoint, train data frame, label Name (Character)
  # yk Data frame containg unique label and there probability
  colNum <- which(names(train) == yLabel)
  yk$probCalc <- 0
  varInv <- solve(commonVar)
  for(i in yk[, 1]){
    yk[which(yk[, 1] == i),3] <- t(as.numeric(test[,-colNum]))%*%varInv%*%muK(train, i, colNum) -
      0.5*(t(muK(train, i, colNum))%*%varInv%*%muK(train, i, colNum)) + log(yk[which(yk[, 1] == i),2])
  }
  return(yk[which.max(yk[,3]), 1])
}

detReturn <- function(train, yLabel, method){
  # returns determinant of covariance matrix if p>2 or returns only variance
  # takes two argument training data frame and character Label Name
  # example detReturn(iris, 'V5')
  if (ncol(train) == 2){
    return(abs(variance(train, yLabel, method)))
  } else {
    return(det(variance(train, yLabel, method)))
  }
}

qdaCalc <- function(test, train, yk, yLabel, method){
  # returns predicted label for a test point using qda
  # takes four arguments test data point, training data frame, label probability data frame
  # and label name (Character)
  # example qdaCalc(iris.test, iris.train, yk, 'V5', 'lda')
  colNum <- which(names(train) == yLabel)
  yk$probCalc <- 0
  for(i in yk[, 1]){
    yk[which(yk[, 1] == i),3] <- -(0.5)*(log(detReturn(train[train[, colNum] == i,], yLabel, method))) - 
      (0.5)*(as.numeric(t(test[, -colNum]))%*%solve(variance(train[train[, colNum] == i, ], yLabel, method))%*%as.numeric(test[, -colNum])) + 
      (0.5)*(as.numeric(t(test[, -colNum]))%*%solve(variance(train[train[, colNum] == i, ], yLabel, method))%*%muK(train, i, colNum)) - 
      (0.5)*(t(muK(train, i, colNum))%*%solve(variance(train[train[, colNum] == i, ], yLabel, method))%*%muK(train, i, colNum)) + 
      log(yk[which(yk[, 1] == i),2])
  }
  return(yk[which.max(yk[,3]), 1])
}

dAnalysis <- function(test, train, yLabel, method){
  # returns predicted classification of a test data
  # takes three argument test Data Frame, Train Data Frame and Lnabel Name (Character)
  # example lda(test, train, yLabel)
  yk <- piK(train[,yLabel])
  commonVar <- variance(train, yLabel, method)
  pred <- 0
  if(method == 'lda'){
    for (i in 1:nrow(test)){
      pred[i] <- ldaCalc(test[i,], train, yLabel, yk, commonVar)
    }
  } else if (method == 'qda'){
    for (i in 1:nrow(test)){
      pred[i] <- qdaCalc(test[i,], train, yk, yLabel, method)
    }
  }else {
    print("not a valid method")
  }
  return(pred)
}

accuracy <- function(test.label, pred.label){
  # returns ratio of correct prediction and total test dataset
  # takes two arguments 
  # test.label is label for test data set
  # pred.label is label predicted by knn
  bool <- test.label == pred.label
  return(length(bool[bool == TRUE])/length(test.label))
}