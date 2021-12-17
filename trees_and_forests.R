library(rpart)
library(rattle)
library(rpart.plot)
library(randomForest)



bike_data = read.csv('SeoulBikeData.csv')
head(bike_data)



bike_data$Date = as.Date(bike_data$Date, "%d/%m/%Y")
bike_data$Year = factor(format(bike_data$Date, "%Y"))
bike_data$Month = factor(months(bike_data$Date))
bike_data$Day = factor(format(bike_data$Date, "%d"))
bike_data$DWeek = factor(weekdays(bike_data$Date))
bike_data$Seasons=as.factor(bike_data$Seasons)
bike_data$Functioning.Day=as.factor(bike_data$Functioning.Day)
bike_data$Holiday=as.factor(bike_data$Holiday)



# split dataset
index <- 1:nrow(bike_data)
testindex <- sample(index, trunc(length(index)*0.2))
testset <- bike_data[testindex,-1]
trainset <- bike_data[-testindex,-1]



#classification trees
tree.hitters<-rpart(Rented.Bike.Count~.,
                    method="anova",
                    cp=0.02,
                    data=trainset)
fancyRpartPlot(tree.hitters,sub="")

#RMSE
MSE_root <-mean((trainset$Rented.Bike.Count -mean(trainset$Rented.Bike.Count))^2)
MSE_root
sqrt(MSE_root * tree.hitters$cptable[6,"xerror"])

  #for testset
tree.pred <- predict(tree.hitters, testset[,-1])
tree.MSE <- crossprod(tree.pred - testset[,1]) / length(testindex)
sqrt(tree.MSE)



#bagged Forest
set.seed(200)
bag.bike<-randomForest(Rented.Bike.Count~.,
                       mtry=12,
                       importance=TRUE,
                       ntree=200,
                       data=trainset,
                       compete=FALSE)
bag.bike

#CER:percent increase in OOB error
dotchart(importance(bag.bike)[,1])

#percent increase in Gini index
dotchart(importance(bag.bike)[,2])

#impact of variable  on count
partialPlot(bag.bike,trainset,x.var="Hour",ylab="Count")
partialPlot(bag.bike,trainset,x.var="Temperature..C.",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Humidity",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Solar.Radiation..MJ.M2.",ylab="Count")
partialPlot(bag.bike,trainset,x.var="Functioning.Day",ylab="Count")
partialPlot(bag.bike,trainset,x.var="Seasons",ylab="Count")
partialPlot(bag.bike,trainset,x.var="Snowfall..cm.",ylab="Count")

#RMSE
predict.OOB<-bag.bike$predicted
sqrt(mean((trainset$Rented.Bike.Count-predict.OOB)^2))

  #for testset
bag.pred <- predict(bag.bike, testset[,-1])
bag.MSE <- crossprod(bag.pred - testset[,1]) / length(testindex)
sqrt(bag.MSE)



#random Forest
set.seed(200)
rf.bike<-randomForest(Rented.Bike.Count~.,
                      mtry=sqrt(12),
                      importance=TRUE,
                      ntree=200,
                      data=trainset,
                      compete=FALSE)
rf.bike

#CER:percent increase in OOB error
dotchart(importance(rf.bike)[,1])

#percent increase in Gini index
dotchart(importance(rf.bike)[,2])

#impact of variable  on count
partialPlot(rf.bike,trainset,x.var="Hour",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Temperature..C.",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Humidity",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Functioning.Day",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Solar.Radiation..MJ.M2.",ylab="Count")
partialPlot(rf.bike,trainset,x.var="Seasons",ylab="Count")

#RMSE for traningset 
predict.OOB.1<-rf.bike$predicted
sqrt(mean((trainset$Rented.Bike.Count-predict.OOB.1)^2))

#for testset
rf.pred <- predict(rf.bike, testset[,-1])
rf.MSE <- crossprod(rf.pred - testset[,1]) / length(testindex)
sqrt(rf.MSE)
