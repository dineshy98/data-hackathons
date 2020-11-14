# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:12:27 2020

@author: dineshy86
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('Train.csv')
test= pd.read_csv('Test.csv')
sample= pd.read_csv('sample_submission.csv')
test['shares'] = -1

train = train[train['shares'] < 20000]

sns.distplot(np.log(train['shares']))
train['shares'] = np.log(train['shares'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train.drop(columns = ['shares']),train['shares'])


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 1000,
                             n_jobs = -1,
                             verbose = 1
                             )
rfr.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(np.exp(y_test),np.exp(rfr.predict(X_test)))

sub = sample.copy()
sub['shares'] = xgbreg.predict(test.drop(columns = ['shares']))
sub.to_csv('base_xgbreg_out+trans.csv',index = False)



from xgboost import XGBRegressor

xgbreg = XGBRegressor(n_estimators = 1000)
xgbreg.fit(X_train,y_train)


mean_absolute_error(np.exp(y_test),np.exp(xgbreg.predict(X_test)))
