# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 01:24:26 2021

@author: dineshy86
"""




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





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = drop_cols,inplace = True)

data1 = data.copy()







train_df = data1.loc[data1['risk_flag'] != -1]
test_df = data1.loc[data1['risk_flag'] == -1]
test_df.drop(columns = ['risk_flag'],inplace = True)


#smote

from collections import Counter
from numpy.random import RandomState
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTENC
import sklearn 
for i in cat_cols:
    le = sklearn.preprocessing.LabelEncoder()
    data1[i] = le.fit_transform(data1[i])

sm = SMOTENC( categorical_features=[3,4,5,6,7,8,12,13,14])
X_res, y_res = sm.fit_resample(train_df.drop(columns = ['risk_flag']), train_df['risk_flag'])


y_res.value_counts()



X_tr,X_tst,y_tr,y_tst = train_test_split(X_res,y_res,stratify=y_res)


from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=15000, depth=3, learning_rate=0.1,eval_metric = 'F1')
model.fit(X_tr, y_tr,cat_features=cat_cols,eval_set=(X_tst, y_tst))



roc_auc_score(y_tst,model.predict(X_tst))






sample['risk_flag'] = model.predict(test_df)
sample.to_csv('catboost_smote.csv',index = False)



