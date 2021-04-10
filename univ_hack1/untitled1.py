# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:38:21 2021

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







['id', 'income', 'age', 'experience', 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state', 'current_job_years',
       'current_house_years', 'risk_flag']



from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
train['Rented Bike Counts'] = ms.fit_transform(train[['Rented Bike Count']])




data.columns

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state',
       'married_house_ownership',
       'house_ownership_car_ownership',
       'house_ownership_car_ownership_profession', 'profession_city']


drop_cols = ['id' ]





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score



data1 = data.drop(data.loc[data['risk_flag'] == 0].sample(190000).index)
data1['risk_flag'].value_counts()
data.drop(columns = drop_cols,inplace = True)


train_df = data.loc[data['risk_flag'] != -1]
test_df = data.loc[data['risk_flag'] == -1]



X_tr,X_tst,y_tr,y_tst = train_test_split(train_df.drop(columns = ['risk_flag']),train_df['risk_flag'],stratify = train_df['risk_flag'])


from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=15000, depth=3, learning_rate=0.1)
model.fit(X_tr, y_tr,cat_features=cat_cols,eval_set=(X_tst, y_tst))



roc_auc_score(y_tst,model.predict(X_tst))


sample['risk_flag'] = model.predict(test_df.drop(columns = ['risk_flag']))
sample['risk_flag_proba'] = model.predict_proba(test_df.drop(columns = ['risk_flag']))[:,1]
model.classes_

sample.to_csv('catboost_+_feats_+15000itrs.csv',index = False)



sample['risk_flag'].value_counts()
