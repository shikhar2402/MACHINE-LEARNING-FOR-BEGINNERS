import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('AirQualityUCI.csv',delimiter=';')

dataset = dataset.drop(["Date", "Time"], axis=1)

dataset=dataset[:9357]
dataset.dropna()
dataset.dropna(how="all",axis=1,inplace=True)

cols = list(dataset.columns[:])

for col in cols:
    if dataset[col].dtype != 'float64':
        str_x = pd.Series(dataset[col]).str.replace(',','.')
        float_X = []
        for value in str_x.values:
            fv = float(value)
            float_X.append(fv)

            dataset[col] = pd.DataFrame(float_X)

features=list(dataset.columns)

y = dataset['C6H6(GT)']
X = dataset.drop("C6H6(GT)",axis=1)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=0)
from sklearn.preprocessing import StandardScaler

# Scaling the features using StandardScaler:
'''X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)
'''


#1 Linear regression
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
clf.score(X_test, y_test)


#2 Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
tree.score(X_test, y_test)



