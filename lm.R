linear.model <- lm(Rented_Bike_Count ~ Hour + Temperature + Humidity + Wind_speed + Visibility + Dew_point_temp + Solar_Rad + Rainfall + Snowfall + factor(Seasons) + factor(Holiday) + factor(Funct_Day) + factor(Month) + factor(Day) + factor(DWeek), data = trainset)

summary(linear.model)

#RMSE for training set
lm.train <- predict(linear.model)
lm.trian.MSE <- crossprod(lm.train - trainset[,1]) / nrow(trainset)
lm.trian.RMSE <- sqrt(lm.trian.MSE)

#RMSE for testing set
lm.pred <- predict(linear.model, testset[,-1])
lm.MSE <- crossprod(lm.pred - testset[,1]) / length(testindex)
lm.RMSE <- sqrt(lm.MSE)
