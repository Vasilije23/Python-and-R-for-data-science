library(ggplot2)
library(lubridate)
library(dplyr)
library(tidyr)
library(caret)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)

incidents <- read.csv(file= "C:/Users/nide/Desktop/Fakultet/Su Master/programski jezici DS/projekat/Police_Department_Incidents_-_Previous_Year__2016_.csv")

head(incidents)

na.omit(incidents) 

summary(incidents)

cat<-count(incidents,Category)
cat<-arrange(cat,desc(n))
plot(cat$n,type="h")

pdd<-count(incidents,PdDistrict)
pdd<-arrange(pdd,desc(n))
plot(pdd$n,type="h")

res<-count(incidents,Resolution)
res<-arrange(res,desc(n))
plot(res$n,type="h")

dof<-count(incidents,DayOfWeek)
dof<-arrange(dof)
plot(dof$n,type="h")

ggplot(incidents, aes(X,Y)) + geom_point(cex = .1)

dim(incidents)

features_extraction <- function(incidents){
  labels <- c("Category", "Location", "Address","X", "Y")
  features <- incidents[,labels]
  features$Category <- as.factor(features$Category)
  features$Category <- as.numeric(features$Category)
  features$Location <- as.factor(features$Location)
  features$Location <- as.numeric(features$Location)
  features$Address <- as.factor(features$Address)
  features$Address <- as.numeric(features$Address)
  return(features) }

set.seed(123)
split=sample.split(incidents,SplitRatio=0.8)
training<- subset(incidents,split== TRUE)
testing<- subset(incidents,split== FALSE)

summary(features_extraction(training))
summary(features_extraction(testing))

training$PdDistrict<- as.factor(training$PdDistrict)
training$PdDistrict<- as.numeric(training$PdDistrict)
summary(training$PdDistrict)


rf_model <- randomForest(features_extraction(training), training$PdDistrict, ntree=20)
rf_model

rtree <- rpart(training$PdDistrict ~ ., features_extraction(training), method  = "anova")
rtree
rpart.plot(rtree)





