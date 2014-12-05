#Outline
#1.Scale or normalize data
#2.Run hierachical clustering on data
#3.Run parition clustering on data
#4 validation

library(rpart)
library(randomForest)
library(xlsx)
library(AUCRF)
library(VSURF)
library(hopach)


####### TRAINING

test <-read.csv("C:/Users/dwoo57/Google Drive/Career/Data Mining Competitions/Crowdanalytix/Data/CAX_ExacerbationModeling_TRAIN_data.csv")

### set the variables
test$Exacebator<-factor(test$Exacebator)
test.na<-test
test.na.roughfix<-na.roughfix(test.na)

test.na.roughfix.dist<-distancematrix(test.na.roughfix[,332:1332],"cosangle")
dim(test.na.roughfix.dist)

test.na.roughfix.cluster<-hopach(test.na.roughfix[,332:1332],dmat=test.na.roughfix.dist )

test.na.roughfix.cluster$clust$k

table(test.na.roughfix.cluster$clust$labels)
test.na.roughfix.cluster$clust$sizes

makeoutput(test.na.roughfix,test.na.roughfix.cluster,file = "C:/Users/dwoo57/Documents/mydata.xlsx")

test.clusterd<-read.csv("C:/Users/dwoo57/Google Drive/Career/Data Mining Competitions/Crowdanalytix/Data/Analysis/V2_WCluster/CAX_ExacerbationModeling_TRAIN_data_Dzw_explore_add_cluster_subset.csv")

### set the variables
test.clusterd$Exacebator<-factor(test.clusterd$Exacebator)
test.clusterd$Cluster<-factor(test.clusterd$Cluster)
test.clusterd.na<-test.clusterd
test.clusterd.na.roughfix<-na.roughfix(test.clusterd.na)

attach(test.clusterd.na.roughfix)
fit.clustered <- AUCRF(Exacebator~., data=test.clusterd.na.roughfix)
summary(fit.clustered)
plot(fit)

# Random forecst implementation

test.imputed<-rfImpute(test.na$Exacebator ~ ., data = test.na)
fit_rf<-randomForest(test.imputed$Exacebator ~ ., data = test.imputed)

fit_rf<-randomForest(test.na.roughfix$Exacebator ~ ., data = test.na.roughfix, importance = TRUE)

data_var<-round(importance(fit_rf),2)
write.xlsx(data_var,"C:/Users/dwoo57/Documents/mydata.xlsx")

fit <- AUCRF(Exacebator~., data=test.na.roughfix)

summary(fit)
plot(fit)

fitCV<-AUCRFcv(fit)
summary(fitCV)

variables.impt<-OptimalSet(fit)
write.xlsx(variables.impt,"C:/Users/dwoo57/Documents/mydata.xlsx")

attach(test.na.roughfix)


### now do for reduced set

test_rd <-read.csv("C:/Users/dwoo57/Google Drive/Career/Data Mining Competitions/Crowdanalytix/Data/CAX_ExacerbationModeling_TRAIN_data_reduced_variables_random_forest.csv")
test_rd$Exacebator<-factor(test_rd$Exacebator)
test_rd.na.roughfix<-na.roughfix(test_rd)

fit_rd<-randomForest(test_rd.na.roughfix$Exacebator~.,data = test_rd.na.roughfix)


############# TEST: RESULTS - Output to test data
validation_rd <-read.csv("C:/Users/dwoo57/Google Drive/Career/Data Mining Competitions/Crowdanalytix/Data/CAX_ExacerbationModeling_Public_TEST_data.csv")
validation_rd.na.roughfix<-na.roughfix(validation_rd)


validation_rd$Exacebator<-predict(fit_rd,validation_rd.na.roughfix,type="prob")

validation_output<-validation_rd[,1:2]

summary(validation_rd$Exacebator)

write.csv(validation_output,"C:/Users/dwoo57/Documents/mydata_predict.csv")
