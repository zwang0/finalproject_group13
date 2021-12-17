## ridge
x = data.matrix(trainset[,-1])
y_train = trainset[,"Rented_Bike_Count"]
lambda_seq<- 10^seq(2, -2, by = -.1)
ridge_fit <- glmnet(x, y_train, alpha = 0, lambda  = lambda_seq)
summary(ridge_fit)

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
plot(ridge_cv)

# Best lambda value
best_lambda <- ridge_cv$lambda.min
best_lambda
# Extracting the best model using K-cross validation
best_fit <- ridge_cv$glmnet.fit

# Rebuilding the model with optimal lambda value
best_ridge <- glmnet(x, y_train, alpha = 0, lambda = 3.162278)
coef(best_ridge)

# using the test dataset
ridge.pred <- predict(best_ridge,s = best_lambda, data.matrix(testset[,-1]))
ridge.MSE <- crossprod(ridge.pred - testset[,1]) / length(testindex)
sqrt(ridge.MSE)

# R squared formula
actual <- testset$Rented_Bike_Count
preds <- ridge.pred
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq

## Lasso
x = data.matrix(trainset[,-1])
y_train = trainset[,"Rented_Bike_Count"]
lambda_seq<- 10^seq(2, -3, by = -.1)
lasso_fit <- glmnet(x, y_train, alpha = 1, lambda  = lambda_seq)
summary(lasso_fit)

# Using cross validation glmnet
lasso_cv <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas)
plot(lasso_cv)
# Best lambda value
best_lambda <- lasso_cv$lambda.min
best_lambda
# Extracting the best model using K-cross validation
best_fit <- lasso_cv$glmnet.fit


# Rebuilding the model with optimal lambda value
best_lasso <- glmnet(x, y_train, alpha = 1, lambda = 1.584893)
coef(best_lasso)

# using the test dataset
lasso.pred <- predict(best_lasso,s = best_lambda, data.matrix(testset[,-1]))
lasso.MSE <- crossprod(lasso.pred - testset[,1]) / length(testindex)
sqrt(lasso.MSE)

# R squared formula
actual <- testset$Rented_Bike_Count
preds <- lasso.pred
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq
