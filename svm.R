library(e1071)
## svm
svm.model <- svm(Rented_Bike_Count ~ ., data = trainset, cost = 1000, gamma = 0.0001)
svm.pred <- predict(svm.model, testset[,-1])
svm.MSE <- crossprod(svm.pred - testset[,1]) / length(testindex)

#10-fold CV
n <- nrow(trainset)
sum.MSE <- numeric(10)
for (i in 1:10){
  #split training data and validation data
  v_index <- seq(from = i, to = n, by = 10)
  train <- trainset[-v_index,]
  valid <- trainset[v_index,]
  #fit model
  svm.model <- svm(Rented_Bike_Count ~ ., data = train, cost = 1000, gamma = 0.0001)
  pred <- predict(svm.model,valid[,-1])
  #calculate MSE
  sum.MSE[i] <- crossprod( pred - valid[,1]) / nrow(valid)
}
#mean of 10 MSEs
mean.MSE <- mean(sum.MSE)