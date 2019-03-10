library(glmnet)
library(Metrics)
library(xgboost)
library(caret)
train=read.csv("train.csv",stringsAsFactors = FALSE)
test=read.csv("test.csv",stringsAsFactors = FALSE)
test$SalePrice=NA
train_ID=train$PID
test_ID=test$PID
combi=rbind(train,test)
combi$Sale_Price = log(combi$Sale_Price + 1)
combi=combi[,-1]
idx=which(is.na(combi$Garage_Yr_Blt))
combi[idx, 'Garage_Yr_Blt'] = combi[idx, 'Year_Built'] 
combi=subset(combi,select=-c(Longitude,Latitude))


combi=subset(combi,select =-c(MS_Zoning, Street, Alley,Lot_Shape,Land_Contour,Land_Slope, Condition_2, Roof_Matl,Heating,Pool_QC) )
combi$YrSold=as.character(combi$Year_Sold)
combi$MoSold=as.character(combi$Mo_Sold)

feature_classes=sapply(names(combi),function(x){class(combi[[x]])})
numeric_feats=names(feature_classes[feature_classes != "character"])

# get names of categorical features
categorical_feats=names(feature_classes[feature_classes == "character"])

# use caret dummyVars function for hot one encoding for categorical features
dummies=dummyVars(~.,combi[categorical_feats])
categorical_1_hot=predict(dummies,combi[categorical_feats])

combi <- cbind(combi[numeric_feats],categorical_1_hot)
n=dim(combi)[1]
tn_size=seq(1,0.7*n,by=1)
train_df=combi[tn_size,]
test_df=combi[-tn_size,]
l=length(numeric_feats)


#lasso
cv_lasso=cv.glmnet(as.matrix(train_df[,-l]),train_df[,l])

## Predictions
preds=predict(cv_lasso,newx=as.matrix(test_df[,-l]),s="lambda.min")
rmse(test_df$Sale_Price,preds)
y_prediction=round(exp(preds)-1,digits = 2)
mysubmission1 = cbind(test_ID,y_prediction)
write.table(mysubmission1, 'mysubmission1.txt', row.names = FALSE, col.names = c("PID","Sale_Price"), sep = ", ")

#randomforest
#library(randomForest)
#rf_model=train(Sale_Price~.,data=train_df,method='rf',nodesize=10,ntree=500,trControl=trainControl(method="oob"),tuneGrid = expand.grid(mtry = c(123)))
#preds=predict(rf_model,test_df)



#xgboost

xgbFit=xgboost(data=as.matrix(train_df[,-l]),nfold=5,label=as.matrix(train_df$Sale_Price),nrounds=2200,verbose=FALSE,objective='reg:linear',eval_metric='rmse',nthread=8,eta=0.01,gamma=0.0468,max_depth=6,min_child_weight=1.7817,subsample=0.5213,colsample_bytree=0.4603)
preds2 = predict(xgbFit,newdata=as.matrix(test_df[,-l]))
rmse(test_df$Sale_Price,preds2)
y_prediction1=round(exp(preds2)-1,digits = 2)
mysubmission2 = cbind(test_ID,y_prediction1)
write.table(mysubmission2, 'mysubmission2.txt', row.names = FALSE, col.names = c("PID","Sale_Price"), sep = ", ")
