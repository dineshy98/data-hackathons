# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:38:16 2021

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



#creating holdout dataset





data0 = data.loc[data['risk_flag'] == 0]
data1 = data.loc[data['risk_flag'] == 1]

data.drop(columns = drop_cols,inplace = True)

for i in range(7):
    
    globals()['data0%s' % i] = data0[31000*i:31000*(i+1)]


for i in range(7):
    
    globals()['train_df1%s' % i] = pd.concat([globals()['data0%s' % i],data1])
    globals()['train_df1%s' % i] = globals()['train_df1%s' % i].sample(frac=1).reset_index(drop=True)

for i in range(7):
    globals()['train_df1%s' % i].drop(columns = drop_cols,inplace = True)
    


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier



test_df = data1.loc[data1['risk_flag'] == -1]


for i in range(7):
    globals()['X_tr%s' % i],globals()['X_tst%s' % i],globals()['y_tr%s' % i],globals()['y_tst%s' % i] = train_test_split(globals()['train_df1%s' % i].drop(columns = ['risk_flag']),globals()['train_df1%s' % i]['risk_flag'],stratify = globals()['train_df1%s' % i]['risk_flag'])


score = {}

from catboost import CatBoostClassifier
for i in range(7):
    
    globals()['catboost%s' % i]=CatBoostClassifier()
    globals()['catboost%s' % i].fit(globals()['X_tr%s' % i], globals()['y_tr%s' % i],cat_features=cat_cols,eval_set=(globals()['X_tst%s' % i], globals()['y_tst%s' % i]))
    score['catboost{}'.format(i)] = roc_auc_score(globals()['y_tst%s' % i],globals()['catboost%s' % i].predict(globals()['X_tst%s' % i]))




#predictions


    
test_df = data.loc[data['risk_flag'] == -1]
predictions = pd.DataFrame()
predictions_proba = pd.DataFrame()

for i in range(7):
    predictions['catboost{}'.format(i)] = globals()['catboost%s' % i].predict(test_df.drop(columns = ['id','risk_flag']))
    
    


for i in range(7):
    predictions_proba['catboost{}'.format(i)] = globals()['catboost%s' % i].predict_proba(test_df.drop(columns = ['id','risk_flag']))[:,0]
    
k = predictions.sum(axis = 1)
zeros_proba = pd.DataFrame(predictions_proba.sum(axis = 1),columns = ['zero_proba_sum'])

final_preds = zeros_proba['zero_proba_sum'].apply(lambda x : 0 if x > 4 else 1)
final_preds = zeros_proba['zero_proba_sum']/7
final_preds.value_counts()


zeros = pd.DataFrame(predictions.sum(axis = 1),columns = ['zero_sum'])
final_preds1 = zeros['zero_sum'].apply(lambda x : 1 if x > 6 else 0)


sample['risk_flag'] = final_preds
sample['risk_flag'].value_counts()
sample.to_csv('ensembled_catboost_6_probas.csv',index = False)

globals()['catboost%s' % 1].

feat_imp = pd.DataFrame(list(test_df.drop(columns = ['id','risk_flag']).columns.values))
feat_imp['score'] = catboost1.get_feature_importance()
catboost1.feature_intraction

