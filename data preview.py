import pandas as pd
from sklearn import datasets, linear_model

mydf = pd.read_csv('.//all//train.csv', sep=',', na_values="NA")
mydf_dummy = pd.get_dummies(mydf, dummy_na=True)
y = mydf_dummy['SalePrice']
x = mydf_dummy.drop(['SalePrice'], axis=1)
x = x.fillna(x.mean())
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x, y)

# Make predictions using the training set

test = pd.read_csv('.//all//test.csv', sep=',', na_values="NA")
test_dummy = pd.get_dummies(test, dummy_na=True)
train_copy = pd.DataFrame(columns=x.columns, index=test_dummy.index)
test_x = train_copy.combine_first(test_dummy)

test_x = test_x.fillna(test_x.mean())
test_x = test_x.fillna(0)
test_x = test_x[x.columns]
test_y_pred = regr.predict(test_x)

test_y_pred_df = pd.DataFrame(data=test_y_pred, index=test_x["Id"])

test_y_pred_df.to_csv(".//all//pred.csv", sep=",")
