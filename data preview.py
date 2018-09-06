import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

mydf = pd.read_csv('.//all//train.csv', sep=',', na_values="NA")

mydf.isnull().sum().sort_values(ascending=False)
mydf['PoolQc'].fillna("NotApplicable")
mydf['MiscFeature'].fillna("NotApplicbple")
mydf["Alley"].fillna("NotApplicable")
mydf['Fence'].fillna("NotApplicable")
mydf['MiscFeature'].fillna("NotApplicable")
mydf['LotFrontage'].fillna("NotApplicable")

#####################################################
##imputation
mydf['PoolQC'].fillna("NotApplicable")
mydf['MiscFeature'].fillna("NotApplicbple")
mydf["Alley"].fillna("NotApplicable")
mydf['Fence'].fillna("NotApplicable")
mydf['MiscFeature'].fillna("NotApplicable")
mydf['GarageType'].fillna("NotApplicable")
mydf['GarageCond'].fillna("NotApplicable")
mydf['GarageQual'].fillna("NotApplicable")
mydf['GarageFinish'].fillna("NotApplicable")
mydf['GarageYrBlt'].fillna(0)

mydf['BsmtExposure'].fillna("NotApplicable")
mydf['BsmtFinType2'].fillna("NotApplicable")
mydf['BsmtFinType1'].fillna("NotApplicable")
mydf['BsmtCond'].fillna("NotApplicable")
mydf['BsmtQual'].fillna("NotApplicable")


mydf["MasVnrType"] = mydf["MasVnrType"].fillna("None")
mydf["MasVnrArea"] = mydf["MasVnrArea"].fillna(0)
mydf['Electrical'] = mydf['Electrical'].fillna(mydf['Electrical'].mode()[0])

#####################################################

from scipy import stats

UniqueNames = mydf["Neighborhood"].unique()

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = mydf[:][mydf["Neighborhood"]== key]


f_val, p_val = stats.f_oneway(mydf["LotFrontage"])
print ("One-way ANOVA P =", p_val )

mydf_dummy = pd.get_dummies(mydf, dummy_na=True)
y = mydf_dummy['SalePrice']
x = mydf_dummy.drop(['SalePrice'], axis=1)
x = x.drop(['Id'], axis=1)
x = x.fillna(x.mean())
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x, y)





test = pd.read_csv('.//all//test.csv', sep=',', na_values="NA")
test_dummy = pd.get_dummies(test, dummy_na=True)
test_dummy = test_dummy.drop(['Id'], axis=1)
train_copy = pd.DataFrame(columns=x.columns, index=test_dummy.index)
test_x = train_copy.combine_first(test_dummy)

test_x = test_x.fillna(test_x.mean())
test_x = test_x.fillna(0)
test_x = test_x[x.columns]
test_y_pred = regr.predict(test_x)

test_y_pred_df = pd.DataFrame(data=test_y_pred, index=test["Id"], columns=["SalePrice"])

test_y_pred_df.to_csv(".//all//pred.csv", sep=",")


from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1, max_iter=100000)
lasso.fit(x,y)
test_y_pred_lasso = lasso.predict(test_x)

test_y_pred_lasso_df = pd.DataFrame(data=test_y_pred_lasso, index=test["Id"], columns=["SalePrice"])
test_y_pred_lasso_df.to_csv(".//all//pred_lasso.csv", sep=",")

from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(activation='logistic',max_iter=20000,learning_rate_init=0.0005)
nn.fit(x,y)
test_y_pred_nn = nn.predict(test_x)

test_y_pred_nn_df = pd.DataFrame(data=test_y_pred_nn, index=test["Id"], columns=["SalePrice"])
test_y_pred_nn_df.to_csv(".//all//pred_n.csv", sep=",")
