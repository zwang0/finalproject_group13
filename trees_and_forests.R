library(rpart)
library(rattle)
library(rpart.plot)
library(randomForest)

#classification trees

tree.hitters<-rpart(Rented.Bike.Count~.,
                    method="anova",
                    cp=0.02,
                    data=testset)
fancyRpartPlot(tree.hitters,sub="")

n<-nrow(testset)
rss_here <- rep(0,n) 
for(i in 1:n){
  tree.hittersCV <-rpart(Rented.Bike.Count~.,
                         method="anova",
                         cp=0.02,
                         data=testset[-i,])

  yhat <- predict(tree.hittersCV, newdata = testset[i,])
  
  rss_here[i] <- (testset$Rented.Bike.Count[i] - yhat)^2
  }
  
sqrt(mean(rss_here))



#bagged Forest

set.seed(200)
bag.bike<-randomForest(Rented.Bike.Count~.,
                       mtry=12,
                        importance=TRUE,
                        ntree=1000,
                        data=testset,
                        compete=FALSE)
bag.bike
#CER:percent increase in OOB error
dotchart(importance(bag.bike)[,1])
#percent increase in Gini index
dotchart(importance(bag.bike)[,2])
#impact of variable  on count
partialPlot(bag.bike,testset,x.var="Hour",ylab="Count")
partialPlot(bag.bike,testset,x.var="Temperature..C.",ylab="Count")
partialPlot(rf.bike,testset,x.var="Humidity",ylab="Count")
partialPlot(rf.bike,testset,x.var="Solar.Radiation..MJ.M2.",ylab="Count")
partialPlot(bag.bike,testset,x.var="Functioning.Day",ylab="Count")
partialPlot(bag.bike,testset,x.var="Seasons",ylab="Count")
partialPlot(bag.bike,testset,x.var="Snowfall..cm.",ylab="Count")
#RMSE
predict.OOB<-bag.bike$predicted
sqrt(mean((testset$Rented.Bike.Count-predict.OOB)^2))


#random Forest

set.seed(200)
rf.bike<-randomForest(Rented.Bike.Count~.,
                       mtry=sqrt(12),
                       importance=TRUE,
                       ntree=1000,
                       data=testset,
                       compete=FALSE)
rf.bike
#CER:percent increase in OOB error
dotchart(importance(rf.bike)[,1])
#percent increase in Gini index
dotchart(importance(rf.bike)[,2])
#impact of variable  on count
partialPlot(rf.bike,testset,x.var="Hour",ylab="Count")
partialPlot(rf.bike,testset,x.var="Temperature..C.",ylab="Count")
partialPlot(rf.bike,testset,x.var="Humidity",ylab="Count")
partialPlot(rf.bike,testset,x.var="Functioning.Day",ylab="Count")
partialPlot(rf.bike,testset,x.var="Solar.Radiation..MJ.M2.",ylab="Count")

partialPlot(rf.bike,testset,x.var="Seasons",ylab="Count")

#RMSE
predict.OOB.1<-rf.bike$predicted
sqrt(mean((testset$Rented.Bike.Count-predict.OOB.1)^2))
