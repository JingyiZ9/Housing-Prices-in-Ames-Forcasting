setwd("C:/Users/yong/Desktop/stat542")
mydata=read.csv("Ames_data.csv",stringsAsFactors = FALSE)

tn_size=sample(2930,0.7*2930)

mydata=rbind(train,test)
na.cols <- which(colSums(is.na(data)) > 0)
#sort(colSums(sapply(data[na.cols], is.na)), decreasing = TRUE)
#Garage_Yr_Blt has 159 missing values
#length(which(data$Garage_Yr_Blt == data$Year_Built)) #2227
idx=which(is.na(data$Garage_Yr_Blt))

#replace the missing value in garageyrblt with year bulit
mydata[idx, 'Garage_Yr_Blt'] = data[idx, 'Year_Built'] 
mydata=subset(mydata,select=-c(Longitude,Latitude))

#categorical  data
Num=sapply(train_df,is.numeric)
Num=train_df[,Num]

for (i in 1:(length(train_df)-1)) {
  if(is.factor(train_df[,i])){
    train_df[,i]=as.integer(train_df[,i])
  }
}


for (i in 1:(length(train_df)-1)) {
  if(is.factor(train_df[,i])){
    levels=cbind(names(train_df[,i]),table(train_df[,i]))
  }
}


#drop columns
domint_cata=c('MS_Zoning', 'Street', 'Alley','Lot_Shape','Land_Contour','Land_Slope', 'Condition_2', 'Roof_Matl','Heating,Pool_QC')


#split data
train_df=data[tn_size,]
test_df=data[-tn_size,]
write.csv(train,file="train.csv",quote=F,row.names = F)
write.csv(test,file="test.csv",quote=F,row.names = F)

#model 1
library(randomForest)

#install packages
mypackages = c("leaps", "glmnet")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

library(leaps)  # regsubsets
library(glmnet)

rf_model=train(Sale_Price~.,
               data=train_df,
               method='rf',
               nodesize=10,
               ntree=500,
               trControl=trainControl(method="oob"),
               tuneGrid = expand.grid(mtry = c(123)))

price_pre=predict(rf_model,test_df,type='raw')
rmse(test_df$Sale_Price,price_pre)

price_pre=data.frame(price_pre)

sqrt((sum(((price_pre)-(test_df$Sale_Price))^2))/879)

charcol=names(all[,sapply(all,is.character)])

rf_model1=randomForest(log(Sale_Price)~.,data=train_df[,-1],importance = T, ntree=500)
price_pre1=predict(rf_model1,test_df)
price_pre1=data.frame(price_pre1)

sqrt((sum((price_pre1-log(test_df$Sale_Price))^2))/879)

regresstionm=lm(Sale_Price~.,
   data=train_df[,-1])



correlations<- cor(Num[,-1],use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")
#model2
install.packages("xgboost")
library(xgboost)
train<- as.matrix(train_df, rownames.force=NA)
test<- as.matrix(test_df, rownames.force=NA)
train <- as(train, "sparseMatrix")
test <- as(test, "sparseMatrix")
# Never forget to exclude objective variable in 'data option'
train_Data <- xgb.DMatrix(data = train[,-1], label = train[,"SalePrice"])
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)

xgb.grid <- expand.grid(nrounds = 500,
                        max_depth = seq(6,10),
                        eta = c(0.01,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1),
                        min_child_weight=seq(1,10)
)

xgb_tune <-train(SalePrice ~.,
                 data=Training_Inner,
                 method="xgbTree",
                 metric = "RMSE",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid
)


library(glmnet)
install.packages("Metrics")
library(Metrics)
cv_lasso = cv.glmnet(as.matrix(Training[, -59]), Training[, 59])



##part 2

one_step_lasso = function(r, x, lam){
  xx = sum(x^2)
  xr = sum(r*x)
  b = (abs(xr) -lam/2)/xx
  b = sign(xr)*ifelse(b>0, b, 0)
  return(b)
}




mylasso=function(X,y,standardize=TRUE){
  n=dim(X)[1]
  p=dim(X)[2]
  b=rep(0,p)
  r=y
  if standardize==TRUE
  for (step in 1:50) {
    for (j in 1:p) {
      r=r+X[,j]*b[j]
      b[j] = one_step_lasso(r, X[, j], lam)
      
      
      
      
      
      
    }
    
  }
}






