library(e1071)

# training
## 10-fold CV function
  svm_10_cv <- function(trainset, C, G){
    n <- nrow(trainset)
    sum.MSE <- numeric(10)
    for (i in 1:10){
      v_index <- seq(from = i, to = n, by = 10)
      train <- trainset[-v_index,]
      valid <- trainset[v_index,]
      svm.model <- svm(Rented_Bike_Count ~ ., data = train, cost = C, gamma = G)
      pred <- predict(svm.model,valid[,-1])
      sum.MSE[i] <- crossprod( pred - valid[,1]) / nrow(valid)
    }
    mean.MSE <- mean(sum.MSE)
    return(mean.MSE)
  }

## function of grid search
  grid_search <- function(C_list, G_list){
    nc <- length(C_list)
    ng <- length(G_list)
    cv_mat <- matrix(0, nrow = nc * ng, ncol = 3)
    rowflag <- 1
    for (i in 1: nc){
      C <- C_list[i]
      for (j in 1: ng){
        G <- G_list[j]
        mean.MSE <- svm_10_cv (trainset, C, G)
        print( rowflag )
        cv_mat[rowflag,] <- c(C, G, mean.MSE)
        rowflag <- rowflag + 1
      }
    }
    return(cv_mat)
  }

## grid search to select cost and gamma
### grid search 1
C_list <- c(1,seq(10, 100, by =10))
G_list <- c(seq(0.01,0.1,by=0.01))
out_gs1 <- grid_search(C_list, G_list)
out_gs1[order(out_gs1[,3]),]

### grid search 2
C_list <- c(20,30)
G_list <- c(seq(0.09,0.12,by=0.01))
out_gs2 <- grid_search(C_list, G_list)
out_gs2[order(out_gs2[,3]),]

# testing
## svm with cost=20, gamma=0.11
svm.model <- svm(Rented_Bike_Count ~ ., data = trainset, cost = 20, gamma = 0.11)
svm.pred <- predict(svm.model, testset[,-1])
svm.MSE <- crossprod(svm.pred - testset[,1]) / length(testindex)
svm.RMSE <- sqrt(svm.MSE)
