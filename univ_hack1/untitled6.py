# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 19:40:10 2021

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


import category_encoders as ce
encd = ce.cat_boost.CatBoostEncoder(cols=cat_cols,return_df=True)
data1 = encd.fit_transform(data,data['risk_flag'])

train_df = data1.loc[data1['risk_flag'] != -1]
test_df = data1.loc[data1['risk_flag'] == -1]




X_tr,X_tst,y_tr,y_tst = train_test_split(train_df.drop(columns = ['risk_flag']),train_df['risk_flag'],stratify = train_df['risk_flag'])

weighted_clf = RandomForestClassifier(max_depth=3 ,random_state=0,class_weight={0:0.08,1:0.92}).fit(X_tr, y_tr)
roc_auc_score(y_tst,weighted_clf.predict(X_tst))



sample['risk_flag_weighted_rfc'] = weighted_clf.predict(test_df.drop(columns = ['risk_flag']))
sample['risk_flag_proba_weighted_rfc'] = weighted_clf.predict_proba(test_df.drop(columns = ['risk_flag']))[:,1]
weighted_clf.classes_

sample['risk_flag'] = weighted_clf.predict(test_df.drop(columns = ['risk_flag']))
sample['risk_flag'].value_counts()

sample.to_csv('weighted_rfc.csv',index = False)


