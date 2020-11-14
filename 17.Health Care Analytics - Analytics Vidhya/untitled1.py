# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:46:39 2020

@author: dineshy86
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('Train.csv')
test= pd.read_csv('Test.csv')
sample= pd.read_csv('sample_submission.csv')
test['PE'] = -1

train.corr()
test.isnull().sum()

sns.distplot(train['PE'])
X = train[['AT', 'V', 'AP', 'RH']]
y = train['PE']
train = train.reset_index()

data = pd.concat([train,test])

from sklearn.preprocessing import PolynomialFeatures
interactions = PolynomialFeatures(interaction_only=True)
X_interactions= interactions.fit_transform(data[['AT', 'V', 'AP', 'RH', 'PE']])
X_interactions = pd.DataFrame(X_interactions)
temp = pd.merge(data,X_interactions,left_index=True, right_index=True)

train
X = train[['AT', 'V', 'AP', 'RH']]
y = train['PE']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)


from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost 



from catboost import CatBoostRegressor

knnr_scld =
(make_pipeline(StandardScaler(),


estimators = [
    ('forest',RandomForestRegressor(n_estimators=1000,random_state=42)),
    ('lr',  CatBoostRegressor(120)),
    ('xgb', xgboost.XGBRegressor(750))
]
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=)


reg.fit(X_train, y_train)

from  sklearn.metrics import mean_squared_error
mean_squared_error(reg.predict(X_test), y_test)

submission = sample.copy()
submission['PE'] = reg.predict(test.drop(columns = ['PE']))
submission.to_csv('sklearn_stack2.csv',index = True)


