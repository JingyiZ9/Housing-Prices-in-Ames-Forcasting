# Housing-Prices-in-Ames-Forcasting
https://www.kaggle.com/claudia9/house-price-prediction
## 01 Introduction
The project is aimed to predict housing price based on analyzing the housing data collected on residential properties sold
in Ames, Iowa between 2006 and 2010. The dataset has 2930 rows (i.e., houses) and 83 columns. Column 1 is "PID", the Parcel
identification number, the last column is the response variable, "Sale_Price", and the remaining 81 columns are explanatory
variables describing (almost) every aspect of residential homes.

## 02 Methodology
### Data Preprocessing
Before building models, we do some preparation. First, we separate test data and train data, as well as X and y in both of 
them. Second, we should fix the missing data. We could replace missing value by 0, 1, mean and median based on the
personalities of columns. We will use the third training/test splits to select the best model of minimum RSME, then use 10
set of splits to fit that best model.

### Model 1: Xgboosting
We make sure X matrix is a numeric matrix by transfer categorical data into numerical data and take away “PIN” column before
fit the model. The mean of 10 sets accuracy is 0.9609826.

### Model 2: Regression with lasso
First, we remove some unnecessary variables based on experience. For example, due to very close area, the variables “Longitude”
and “Latitude” can be dropped, and some variables, such as “Pool_Area”, are dropped because of specific values. Then, we 
transfer value from numeric to categoric data for both “Mo_Sold” and “Year_Sold”. Third, we take off outliers for numerical 
variables, where we assume 95% of the train column as the upper bound. Last, we fit the lasso model. The mean of these accuracy 
is 0.952038.

### Model 3: GAM model
To avoid curse of dimensionality when there are too many variables, we first remove some variables as below: 'Street', 
'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude',
'Latitude', 'Mo_Sold', 'Year_Sold', 'PID', 'Sale_Price'. Second, we take some numerical variables below to be linear terms: 
'BsmtFin_SF_1', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'Fireplaces',
'Garage_Cars', which only take specific values not enough to fit a nonparametric curve. Also, for other numerical variables, 
we get the list of numerical variables whose nonlinear terms will be included in the gam model. Third, we generate select 
binary variables referring to lasso result. As we have three parts of features, we fit the gam model. The mean of these 
accuracy is 0.9608281.

## 03 Conclusion
Comparing the RMSE of three models, we find boosting tree model is the best with high accuracy and the RMSE within the 
thresholds as required (RMSE of the first five samples is lower than 0.125, while RMSE of the last five samples is lower 
than 0.135).

