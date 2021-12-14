# Principle Component Regression
# Zehua Wang, biostat625 final project

PCRegression = function(data) { # input: trainset
  # principle component analysis
  X = model.matrix(Rented_Bike_Count ~., data)[,-1]
  pca = prcomp(x = X, center = TRUE, scale = TRUE)
  pca.vars <- apply(pca$x, 2, var)
  pca.props = pca.vars / sum(pca.vars)
  pca.cumprops = cumsum(pca.props)
  pca.pc80 = min(which(pca.cumprops > 0.80))
  X.pca = pca$x[, 1:pca.pc80] # we choose the PC over 80
  pcr = lm(data$Rented_Bike_Count ~ X.pca)
  pcr.mse = mean((pcr$fitted.values - data$Rented_Bike_Count)^2)
  cat("MSE of Principle Component Regression is:", pcr.mse)
  return(pcr)
}
