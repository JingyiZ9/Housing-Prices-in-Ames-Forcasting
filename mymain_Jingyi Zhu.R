
library(Metrics)

library(caret)

#############################################################################
#Part 1 dealing with the missing values preprocessing the training/test data
data <- read.csv("~/Desktop/542_project1/Ames_data.csv", stringsAsFactors=FALSE)
load("~/Desktop/542_project1/project1_testIDs.R")

#load the training and test data
train=read.csv("train.csv",stringsAsFactors = FALSE)
test=read.csv("test.csv",stringsAsFactors = FALSE)

#change the X matrix to a numerical matrix (no factors)
j <- 6
test.dat <- data[testIDs[,j], ]
train.dat <- data[-testIDs[,j], ]
n.train = dim(train.dat)[1]
n.test = dim(test.dat)[1]

train.y <- log(train.dat$Sale_Price)
train.x <- subset(train.dat,select=-c(PID, Sale_Price))
test.y <- log(test.dat$Sale_Price)
test.PID <- test.dat$PID
test.x <- subset(test.dat,select=-c(PID, Sale_Price))

#dealing with the missing values replace the missing value with zero
id1=which(is.na(train.x$Garage_Yr_Blt))
id2=which(is.na(test.x$Garage_Yr_Blt))
train.x[id1, 'Garage_Yr_Blt'] = 0
test.x[id2, 'Garage_Yr_Blt'] = 0

#change the X matrix to a numerical matrix (no factors)
PreProcessingMatrixOutput <- function(train.data, test.data){
  # generate numerical matrix of the train/test
  # assume train.data, test.data have the same columns
  categorical.vars <- colnames(train.data)[which(sapply(train.data, 
                                                        function(x) is.character(x)))]
  train.matrix <- train.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  test.matrix <- test.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  n.train <- nrow(train.data)
  n.test <- nrow(test.data)
  for(var in categorical.vars){
    mylevels <- sort(unique(train.data[, var]))
    m <- length(mylevels)
    tmp.train <- matrix(0, n.train, m)
    tmp.test <- matrix(0, n.test, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[train.data[, var]==mylevels[j], j] <- 1
      tmp.test[test.data[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    colnames(tmp.test) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
    test.matrix <- cbind(test.matrix, tmp.test)
  }
  return(list(train = as.matrix(train.matrix), test = as.matrix(test.matrix)))
}

total = PreProcessingMatrixOutput(train.x, test.x)
train.x <- total$train
test.x <- total$test

#############################################################################
#Part 2 XGBoost
library(xgboost)
set.seed(3610)
xgb.model <- xgboost(data = train.x, label = train.y, max.depth = 15, eta = 0.05, nround = 400, gamma=0.01,objective='reg:linear')
#tmp is the fitted values using test.x
tmp <- predict(xgb.model, test.x)
#calculate the test error
sqrt(mean((tmp - test.y)^2))


#########################################################################
#Part 3 linear regression with lasso
library(glmnet)
train.y <- log(train.dat$Sale_Price)
names(train.y) <- "Sale_Price"
train.x <- subset(train.dat,select=-c(PID, Sale_Price))
test.y <- log(test.dat$Sale_Price)
test.PID <- test.dat$PID
test.x <- subset(test.dat,select=-c(PID, Sale_Price))

#Remove the following variables
train.x <- subset(train.dat,select= -c(Street, Utilities, Condition_2, Roof_Matl, Heating,
                                       Pool_QC, Misc_Feature, Low_Qual_Fin_SF, Pool_Area, 
                                       Longitude, Latitude, PID, Sale_Price))
test.x <- subset(test.dat,select= -c(Street, Utilities, Condition_2, Roof_Matl, Heating,
                                       Pool_QC, Misc_Feature, Low_Qual_Fin_SF, Pool_Area, 
                                       Longitude, Latitude, PID, Sale_Price))

#Set "Mo_Sold" and "Year_Sold" as categorical variables
train.x$YrSold=as.character(train.x$Year_Sold)
train.x$MoSold=as.character(train.x$Mo_Sold)
test.x$YrSold=as.character(test.x$Year_Sold)
test.x$MoSold=as.character(test.x$Mo_Sold)

#In the function ProcessWinsorization, compute the upper 95% quantile of the train column, 
#denoted by M; then replace all values in the train and test that are bigger than M by M. 
ProcessWinsorization <- function(train.col, test.col, upper){
  train <- as.numeric(train.col)
  test <- as.numeric(test.col)
  M <- quantile(train,probs=upper)
  index = which(train > M)
  id = which(test > M)
  for(i in index){
    train[i] = M
  }
  for(i in id){
    test[i] = M
  }
  return(list(train = as.matrix(train), test = as.matrix(test)))
}

#Apply winsorization on the following numerical variables
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", 
                 "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', 
                 "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", 
                 "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")
for(var in winsor.vars){
  r <- ProcessWinsorization(train.x[, var], test.x[, var], 0.95)
  train.x[, var] <- r$train
  test.x[, var] <- r$test
}

#Call PreProcessingMatrixOutput to create the design matrix
total = PreProcessingMatrixOutput(train.x, test.x)
train.x <- total$train
test.x <- total$test
#dealing with the missing values replace the missing value with zero
id1=which(is.na(train.x[,'Garage_Yr_Blt']))
id2=which(is.na(test.x[,'Garage_Yr_Blt']))
train.x[id1, 'Garage_Yr_Blt'] = 0
test.x[id2, 'Garage_Yr_Blt'] = 0

#call glmnet
set.seed(100)
cv.out <- cv.glmnet(train.x, train.y, alpha = 1)
tmp <-predict(cv.out, s = cv.out$lambda.min, newx = test.x)
sqrt(mean((tmp - test.y)^2))


#############################################################
#Part4 GAM
library(mgcv)
train.y <- log(train.dat$Sale_Price)
train.x <- subset(train.dat,select=-c(PID, Sale_Price))
test.y <- log(test.dat$Sale_Price)
test.PID <- test.dat$PID
test.x <- subset(test.dat,select=-c(PID, Sale_Price))

#dealing with the missing values replace the missing value with zero
id1=which(is.na(train.x[,'Garage_Yr_Blt']))
id2=which(is.na(test.x[,'Garage_Yr_Blt']))
train.x[id1, 'Garage_Yr_Blt'] = 0
test.x[id2, 'Garage_Yr_Blt'] = 0

#Remove the following variables
train.x <- subset(train.dat,select= -c(Street, Utilities, Condition_2, Roof_Matl, Heating,
                                       Pool_QC, Misc_Feature, Low_Qual_Fin_SF, Pool_Area, 
                                       Longitude, Latitude, Mo_Sold, Year_Sold,
                                       PID, Sale_Price))

test.x <- subset(test.dat,select= -c(Street, Utilities, Condition_2, Roof_Matl, Heating,
                                     Pool_QC, Misc_Feature, Low_Qual_Fin_SF, Pool_Area, 
                                     Longitude, Latitude, Mo_Sold, Year_Sold,
                                     PID, Sale_Price))

#keep the linear term of those variables in the gam model
linear.vars <- c('BsmtFin_SF_1', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 
                 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 
                 'Kitchen_AbvGr', 'Fireplaces', 'Garage_Cars')

#get the list of numerical variables whose nonlinear terms will be included in the gam model
categorical.vars <- colnames(train.dat)[which(sapply(train.dat, 
                                                     function(x) is.character(x)))]
num.vars <- names(train.dat)
num.vars <- num.vars[num.vars != "Sale_Price"]
num.vars <- num.vars[! num.vars %in% categorical.vars]
num.vars <- num.vars[! num.vars %in% linear.vars]
remove.vars <- c('Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating',
                 'Pool_QC', 'Misc_Feature' , 'Low_Qual_Fin_SF', 'Pool_Area', 
                 'Longitude', 'Latitude', 'Mo_Sold', 'Year_Sold',
                 'PID', 'Sale_Price')
num.vars <- num.vars[! num.vars %in% remove.vars]

select.level.var = c('MS_SubClass__Duplex_All_Styles_and_Ages', 
                     'MS_SubClass__One_Story_1945_and_Older',
                     'MS_SubClass__Two_Story_PUD_1946_and_Newer',
                     'MS_Zoning__C_all', 'MS_Zoning__Residential_Medium_Density',
                     'Neighborhood__Crawford', 'Neighborhood__Edwards',
                     'Neighborhood__Green_Hills', 'Neighborhood__Meadow_Village',
                     'Neighborhood__Northridge', 'Neighborhood__Somerset', 
                     'Neighborhood__Stone_Brook', 'Overall_Qual__Above_Average',
                     'Overall_Qual__Average', 'Overall_Qual__Below_Average',
                     'Overall_Qual__Excellent', 'Overall_Qual__Fair', 'Overall_Qual__Good',
                     'Overall_Qual__Poor', 'Overall_Qual__Very_Excellent', 'Overall_Qual__Very_Good',
                     'Overall_Qual__Very_Poor', 'Overall_Cond__Above_Average', 'Overall_Cond__Average',
                     'Overall_Cond__Below_Average', 'Overall_Cond__Excellent', 'Overall_Cond__Fair',
                     'Overall_Cond__Good', 'Overall_Cond__Poor', 'Overall_Cond__Very_Good',
                     'Overall_Cond__Very_Poor', 'Sale_Condition__Abnorml', 'Sale_Condition__AdjLand',
                     'Sale_Condition__Alloca', 'Sale_Condition__Family', 'Sale_Condition__Normal',
                     'Sale_Condition__Partial', 'Sale_Type__COD', 'Sale_Type__Con', 'Sale_Type__ConLD',
                     'Sale_Type__ConLI', 'Sale_Type__ConLw', 'Sale_Type__CWD', 'Sale_Type__New',
                     'Sale_Type__Oth', 'Sale_Type__VWD', 'Sale_Type__WD', 'Bldg_Type__Duplex',
                     'Bldg_Type__OneFam', 'Bldg_Type__Twnhs', 'Bldg_Type__TwnhsE',
                     'Bldg_Type__TwoFmCon')

#generate select.binary.vars
m <- length(select.level.var)
tmp.train <- matrix(0, n.train, m)
tmp.test <- matrix(0, n.test, m)
colnames(tmp.train) <- select.level.var
colnames(tmp.test) <- select.level.var
for(i in 1:m){
  tmp <- unlist(strsplit(select.level.var[i], '__'))
  select.var <- tmp[1]
  select.level <- tmp[2]
  tmp.train[train.dat[, select.var]==select.level, i] <- 1
  tmp.test[test.dat[, select.var]==select.level, i] <- 1
}

select.binary.vars <- colnames(tmp.train, tmp.test)

#create training and test data frame
train.x <- data.frame(train.x[,linear.vars], tmp.train[,select.binary.vars], train.x[,num.vars],train.y)
test.x <-data.frame(test.x[,linear.vars], tmp.test[,select.binary.vars], test.x[,num.vars])
colnames(train.x)[82] <- "Sale_Price"

id1=which(is.na(train.x[,'Garage_Yr_Blt']))
id2=which(is.na(test.x[,'Garage_Yr_Blt']))
train.x[id1, 'Garage_Yr_Blt'] = 0
test.x[id2, 'Garage_Yr_Blt'] = 0
#call GAM
gam.formula <- paste0("Sale_Price ~ ", linear.vars[1])
for(var in c(linear.vars[-1], select.binary.vars))
  gam.formula <- paste0(gam.formula, " + ", var)
for(var in num.vars)
  gam.formula <- paste0(gam.formula, " + s(", var, ")")
gam.formula <- as.formula(gam.formula)

gam.model <- gam(gam.formula, data = train.x, method="REML")
tmp <- predict.gam(gam.model, newdata = test.x)
sqrt(mean((tmp - test.y)^2))







