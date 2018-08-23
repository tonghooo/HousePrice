import pandas as pd
from sklearn import datasets, linear_model
mydf=pd.read_csv('.//all//train.csv', sep=',', na_values="NA")
mydf_dummy=pd.get_dummies(mydf,dummy_na=True)
y=mydf_dummy['SalePrice']
x= mydf_dummy.drop(['SalePrice'], axis=1)
x=x.fillna(x.mean())
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x, y)

# Make predictions using the training set
diabetes_y_pred = regr.predict(x)
diabetes_y_pred.to_csv