# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 01:35:54 2020

@author: dineshy86
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/train.csv')
test= pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/test.csv')
sample= pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/sample.csv')
test['Crop_Damage'] = -1

train.isnull().sum()

train['Number_Weeks_Used'].fillna(train['Number_Weeks_Used'].mean(),inplace = True)
test['Number_Weeks_Used'].fillna(test['Number_Weeks_Used'].mean(),inplace = True)

train_mod = train.copy()
train_mod['Crop_Damage'].replace({1:0},inplace = True)


train_mod['Crop_Damage'].replace({2:1},inplace = True)

X = train_mod.drop(columns = ['ID','Crop_Damage'])
y = train_mod['Crop_Damage']

from imblearn.over_sampling import SMOTE
sm =SMOTE(random_state=42)
X_res_OS , Y_res_OS = sm.fit_resample(X,y)
pd.Series(Y_res_OS).value_counts()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X_res_OS, Y_res_OS, test_size = 0.20, random_state = 10)

X_ovr = pd.DataFrame(X_res_OS)
X_ovr.columns = ['Estimated_Insects_Count', 'Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category', 'Number_Doses_Week', 'Number_Weeks_Used',
       'Number_Weeks_Quit', 'Season']


import lightgbm as lgb

cate_features_name = ['Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category','Season']

d_train=lgb.Dataset(X_ovr, label=Y_res_OS)

model2 = lgb.train(params,d_train,2000, categorical_feature = cate_features_name)


pred = model2.predict(test.drop(columns = ['ID','Crop_Damage']))
pred = pd.Series(pred)
y_pred=pred.round(0)
y_pred = pd.DataFrame(y_pred)
two_cases = pd.DataFrame(y_pred[y_pred[0]>0].index) 

y_pred[y_pred[0]>0].index

'---------------------------------------------------------------------------------------------'

train_mod1 = train.copy()
train_mod1.drop(index = train_mod1[train_mod1['Crop_Damage'] == 2].index ,inplace = True)


X1 = train_mod1.drop(columns = ['ID','Crop_Damage'])
y1 = train_mod1['Crop_Damage']

from imblearn.over_sampling import SMOTE
sm =SMOTE(random_state=42)
X_res_OS1 , Y_res_OS1 = sm.fit_resample(X1,y1)
pd.Series(Y_res_OS1).value_counts()


X_ovr1 = pd.DataFrame(X_res_OS1)
X_ovr1.columns = ['Estimated_Insects_Count', 'Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category', 'Number_Doses_Week', 'Number_Weeks_Used',
       'Number_Weeks_Quit', 'Season']



import lightgbm as lgb

cate_features_name = ['Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category','Season']

d_train1=lgb.Dataset(X_ovr1, label=Y_res_OS1)

model3 = lgb.train(params,d_train1,2000, categorical_feature = cate_features_name)

pred1 = model3.predict(test.drop(index = two_cases[0],columns = ['ID','Crop_Damage']))
pred1 = pd.Series(pred1)
y_pred1=pred1.round(0)
y_pred1 = pd.DataFrame(y_pred1)
y_pred1.index = test.drop(index = two_cases[0],columns = ['ID','Crop_Damage']).index

results = pd.concat([y_pred,y_pred1])

sample['Crop_Damage'] = results[0].values

one_cases = pd.DataFrame( y_pred1[y_pred1[0]>0].index) 


ress = sample.copy()
ress[ress.index ]
ress['Crop_Damage'][two_cases[0]] = 2
y_pred1[y_pred1[0] == 0].index
ress['Crop_Damage'][y_pred1[y_pred1[0] == 0].index] = 0



ress.to_csv('final.csv',index = False)



params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='binary' #Binary target feature
params['metric']='binary_logloss' #metric for binary classification
params['max_depth']=10

res = pd.DataFrame()
res[two_cases[0]] == 2


