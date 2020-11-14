# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 02:19:48 2020

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


data1 = pd.concat([train,test])

train = data1[data1['PE']>-1]
test = data1[data1['PE']==-1]

pred = pd.read_csv('stacked.csv')
pd.concat([train,pred])
X = train.drop(columns = ['PE'])
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
from sklearn.linear_model import Lasso




estimators = [
    ('forest',RandomForestRegressor(n_estimators=500,random_state=42)),
    ('lr',  CatBoostRegressor(100)),
    ('xgb', xgboost.XGBRegressor(350))
]
reg = StackingRegressor(
    estimators=estimators,
    final_estimator= Lasso())



reg.fit(X_train, y_train)


from  sklearn.metrics import mean_squared_error
mean_squared_error(model.predict(X_test), y_test)

submission = sample.copy()
submission['PE'] = reg.predict(test.drop(columns = ['PE']))
submission.to_csv('sklearn_stack4.csv',index = True)



monitor = EarlyStopping(monitor = 'val_loss',min_delta = 1e-3,patience = 5,
                        verbose= 1,mode = 'auto',restore_best_weights = True)
history2 = model2.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1000,callbacks = [monitor])





from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(Dense(10,input_shape = (4,),activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(75,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation = 'linear'))


monitor = EarlyStopping(monitor = 'val_loss',min_delta = 1e-3,patience = 10,
                        verbose= 1,mode = 'auto',restore_best_weights = True)


model.compile(loss='mean_squared_error', optimizer='adam')
history2 = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1000)




















