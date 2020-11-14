# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:17:06 2020

@author: dineshy86
"""


# compare ensemble to each standalone models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot
import pandas as pd

train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample = pd.read_csv('Sample Submission.csv')
test['UnitPrice'] = -1

data = pd.concat([train,test])


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['day'] = data['InvoiceDate'].dt.day
data['month'] = data['InvoiceDate'].dt.month
data['year'] = data['InvoiceDate'].dt.year



['Invoice No','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']

data.drop(columns = ['InvoiceNo','InvoiceDate',Description,StockCode,],inplace= True)




train1= data[data['UnitPrice'] != -1]
test1 = data[data['UnitPrice'] == -1]



#level0 models
level0 = list()

level0.append(KNeighborsRegressor())
level0.append(DecisionTreeRegressor())
level0.append(SVR())






# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('knn', KNeighborsRegressor()))
	level0.append(('cart', DecisionTreeRegressor()))
	level0.append(('svm', SVR()))
	# define meta learner model
	level1 = LinearRegression()
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['knn'] = KNeighborsRegressor()
	models['cart'] = DecisionTreeRegressor()
	models['svm'] = SVR()
	models['stacking'] = get_stacking()
	return models

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train1['StockCode','Description','CustomerID',
                                                           'Country','Quantity'], 
                                                    train1['UnitPrice']
                                                    , test_size=0.2, random_state=42)
model, history = dt.fit(X_train, y_train, epochs=100)


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()