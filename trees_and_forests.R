

#Random Forest
library(rpart)
library(rattle)
library(rpart.plot)
library(randomForest)

#classification trees

tree.hitters<-rpart(Rented.Bike.Count~.,
                    method="anova",
                    cp=0.02,
                    data=trainset)
fancyRpartPlot(tree.hitters,sub="")

n<-nrow(trainset)
rss_here <- rep(0,n) 
for(i in 1:n){
  tree.hittersCV <-rpart(Rented.Bike.Count~.,
                         method="anova",
                         cp=0.02,
                         data=trainset[-i,])

  yhat <- predict(tree.hittersCV, newdata = trainset[i,])
  
  rss_here[i] <- (trainset$Rented.Bike.Count[i] - yhat)^2
}
#RMSE for train
sqrt(mean(rss_here))
#for testset
tree.pred <- predict(tree.hitters, testset[,-1])
tree.MSE <- crossprod(tree.hitters - testset[,1]) / length(testindex)
sqrttree.MSE)

#bagged Forest

set.seed(200)
bag.bike<-randomForest(Rented.Bike.Count~.,
                       mtry=12,
                        importance=TRUE,
                        ntree=1000,
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
sqrt(rf.MSE)

#random Forest

set.seed(200)
rf.bike<-randomForest(Rented.Bike.Count~.,
                       mtry=sqrt(12),
                       importance=TRUE,
                       ntree=1000,
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
