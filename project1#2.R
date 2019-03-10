library(caret)

#preprocess
train=read.csv("train.csv",stringsAsFactors = FALSE)
test=read.csv("test.csv",stringsAsFactors = FALSE)
train_ID=train$PID
test_ID=test$PID
combi=rbind(train,test)
combi$Sale_Price = log(combi$Sale_Price + 1)
combi=combi[,-1]
idx=which(is.na(combi$Garage_Yr_Blt))
combi[idx, 'Garage_Yr_Blt'] = combi[idx, 'Year_Built'] 
combi=subset(combi,select=-c(Longitude,Latitude))

#domint_cata=c(MS_Zoning, Street, Alley,Lot_Shape,Land_Contour,Land_Slope, Condition_2, Roof_Matl,Heating,Pool_QC)
combi=subset(combi,select =-c(MS_Zoning, Street, Alley,Lot_Shape,Land_Contour,Land_Slope, Condition_2, Roof_Matl,Heating,Pool_QC) )
combi$YrSold=as.character(combi$Year_Sold)
combi$MoSold=as.character(combi$Mo_Sold)

feature_classes = sapply(names(combi),function(x){class(combi[[x]])})
numeric_feats =names(feature_classes[feature_classes != "character"])

# get names of categorical features
categorical_feats = names(feature_classes[feature_classes == "character"])

#hot one encoding for categorical features
dummies = dummyVars(~.,combi[categorical_feats])
categorical_1_hot = predict(dummies,combi[categorical_feats])

combi = cbind(combi[numeric_feats],categorical_1_hot)
n=dim(combi)[1]
test.id = seq(1, 2930, by=3)
train_df=combi[-test.id,]
test_df=combi[test.id,]
l=length(numeric_feats)

#lasso
one_step_lasso = function(r, x, lam){
  xx = sum(x^2)
  xr = sum(r*x)
  b = (abs(xr) -lam/2)/xx
  b = sign(xr)*ifelse(b>0, b, 0)
  return(b)
}
mylasso=function(X,y,lam,n.iter,standardize=TRUE){
  n=dim(X)[1]
  p=dim(X)[2]
  b=rep(0,p)
  r=y
  if(standardize==TRUE){
    X=scale(X,center=F,scale=T)
    x_mean=apply(X,2,mean)
    x_sd=apply(X,2,sd) 
  }
  for (step in 1:n.iter) {
    for (j in 1:p){
      r=r+X[,j]*b[j]
      b[j]=one_step_lasso(r,X[,j],lam)
      r=r-X[,j]*b[j]
    }
  }
  b0=mean(y)-sum(b*apply(X,2,mean))
  return(c(b0,b))
}


beta=mylasso(train_df[,-l],train_df[,l],0.0137,1500,standardize = F)
i=rep(1,(dim(test_df)[1]))
intercept=as.data.frame(i,row.names = "intercept")
test_prime=cbind(intercept,test_df[,-l])
y_prediction=as.matrix(test_prime)%*%as.matrix(beta)
rmse(test_df$Sale_Price,y_prediction)

y_prediction=round(exp(y_prediction)-1,digits = 2)
mysubmission3 = cbind(test_ID,y_prediction)
write.table(mysubmission3, 'mysubmission3.txt', row.names = FALSE, col.names = c('PID','Sale_Price'), sep = ", ")
