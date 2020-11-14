# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:47:56 2020

@author: dineshy86
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample_submission_iA3afxn.csv')
test['Response'] = -1

train['Response'].value_counts()

pd.__version__

train.isnull().sum()


data = pd.concat([train,test])
cat_columns = list(data.select_dtypes(object).columns)
cat_columns.remove('Vehicle_Age')

data['Vehicle_Age'] = data['Vehicle_Age'].replace(dict(zip(data['Vehicle_Age'].unique(),[3,2,1])))

data = pd.concat([data.drop(columns = cat_columns),pd.get_dummies(data[cat_columns])],axis = 1)
data.drop(columns = ['id','Region_Code'],inplace = True)

train_mod = data[data['Response'] > -1]
test_mod = data[data['Response'] == -1]

train_mod = pd.concat([train_mod[train_mod['Response'] == 1],train_mod[train_mod['Response'] == 0].sample(46710,random_state = 10)])

from sklearn.model_selection import train_test_split
X_tr,X_tst,y_tr,y_tst  = train_test_split(train_mod.drop(columns = ['Response']),train_mod['Response'])

params_xg = {
        'n_estimators': [100],
        'max_depth':[5],
        'min_child_weight': [1],
        'gamma': [2],
        'subsample': [0.6],
        'colsample_bytree': [ 1.0]
        }

from xgboost import XGBClassifier
xgbc=XGBClassifier(gamma = 1,subsample = 0.6,colsample_bytree = 1,min_child_weight = 5,max_depth = 3,n_estimators=25)

xgbc.fit(train_mod.drop(columns = ['Response']),train_mod['Response'])

from sklearn.metrics import accuracy_score
accuracy_score(y_tst,xgbc.predict(X_tst))

xgbc.predict_proba(X_tst)

from lightgbm import LGBMClassifier
lgbmc = LGBMClassifier()
lgbmc.fit(train_mod.drop(columns = ['Response']),train_mod['Response'])
accuracy_score(y_tst,lgbmc.predict(X_tst))

tset_prob = (lgbmc.predict_proba(test_mod.drop(columns = ['Response'])) + xgbc.predict_proba(test_mod.drop(columns = ['Response'])))/2
submission = sample.copy()
submission['Response'] = np.argmax(tset_prob,axis =1)

submission.to_csv('xgb_lgbm_full_data.csv',index = False)








