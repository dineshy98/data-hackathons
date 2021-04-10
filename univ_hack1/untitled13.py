# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:08:46 2021

@author: dineshy86
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


test = pd.read_csv('Test Data.csv')
test['risk_flag'] = -1
sample = pd.read_csv('Sample Prediction Dataset.csv')
train = pd.read_csv('Training Data.csv')


train.drop(columns = ['id'],inplace = True)
train.drop_duplicates(inplace = True)
train['id'] = -1



train.columns
train['risk_flag'].value_counts()
data = pd.concat([train,test])




#feature engineering
data.isnull().sum()


#1.feature_intraction

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state']


for lisst in [
        ['married','house_ownership'],
        ['house_ownership','car_ownership'],
        ['house_ownership','car_ownership','profession'],
        ['profession','city']
        ]:

    
    if len(lisst) == 2:
       data['_'.join(lisst)] = data[lisst[0]] +'_'+ data[lisst[1]]
    
    elif len(lisst) == 3:
        data['_'.join(lisst)] = data[lisst[0]] +  '_' + data[lisst[1]] + '_' + data[lisst[2]]




data.columns

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state',
       'married_house_ownership',
       'house_ownership_car_ownership',
       'house_ownership_car_ownership_profession', 'profession_city']


drop_cols = ['id' ]

data1 = data.drop(data[data['risk_flag'] == -1].index)
testdf = data[data['risk_flag'] == -1]


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score,f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = drop_cols,inplace = True)



import autogluon as ag
from autogluon import TabularPrediction as task
from sklearn.model_selection import train_test_split


train,test = train_test_split(data, test_size=0.2)
train_data = task.Dataset()
predictor = task.fit(train_data=train, label='TenYearCHD', eval_metric='accuracy')

y_pred = predictor.predict(test_data)

