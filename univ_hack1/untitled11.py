# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 01:21:35 2021

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





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = drop_cols,inplace = True)

data1 = data.copy()

train_df = data1.loc[data1['risk_flag'] != -1]
test_df = data1.loc[data1['risk_flag'] == -1]
test_df.drop(columns = ['risk_flag'],inplace = True)

X_train,X_valid,y_train,y_valid = train_test_split(train_df.drop('risk_flag',axis=1),train_df['risk_flag'],stratify=train_df['risk_flag'],random_state=22)


#modelling
best = ['profession_city']
counter= 0



from catboost import CatBoostClassifier

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
predictions = np.zeros((len(X_valid), 2))
oof_preds = np.zeros((len(test_df), 2))
feature_importance_df = pd.DataFrame()
final_preds = []
# random_state = [77,89,22,1007,1997,1890,2000,2020,8989,786,787,1999992,2021,7654]
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
        print("Fold {}".format(fold_))
        X_trn,y_trn = X_train.iloc[trn_idx],y_train.iloc[trn_idx]
        X_val,y_val = X_train.iloc[val_idx],y_train.iloc[val_idx]
        clf  = CatBoostClassifier(iterations=10000, depth=3, learning_rate=0.2,eval_metric="Logloss")
        clf.fit(X_trn, y_trn, eval_set=[(X_val,y_val)],cat_features=cat_cols)
        final_preds.append(log_loss(y_pred=clf.predict_proba(X_val),y_true=y_val))
        predictions += clf.predict_proba(X_valid)
        oof_preds += clf.predict_proba(test_df)
        counter = counter + 1



oof_preds = oof_preds/counter

sample['risk_flag'] = oof_preds[:,1]
sample['risk_flag'] = sample['risk_flag'].apply(lambda x : 0 if x < 0.5 else 1)

sample.to_csv('cat_sfk_loedata.csv',index = False)


print(sum(final_preds)/5)