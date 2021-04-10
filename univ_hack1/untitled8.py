# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 10:38:26 2021

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



#feature_engineering

#1
data['loc'] = data['city'] + data['state']
data.drop(columns = ['city','state'],inplace = True)

num_cols = ['income']



#2
for i in num_cols:
    data[i] = np.log(data[i]+1)

#3

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession']

data['profession'] = data['profession'].apply(lambda x : x.lower().replace(' ','_'))
data1 = pd.get_dummies(data , columns = cat_cols,drop_first = True)




#modelling


drop_cols = ['id','loc'] 

data1.drop(columns = drop_cols,inplace = True)
train_df = data1.loc[data1['risk_flag'] != -1]
test_df = data1.loc[data1['risk_flag'] == -1]



from sklearn.model_selection import train_test_split
X_tr,X_tst,y_tr,y_tst = train_test_split(train_df.drop(columns = ['risk_flag']),train_df['risk_flag'],stratify = train_df['risk_flag'])






from lightgbm import LGBMClassifier
model = LGBMClassifier(max_depth=0.1,
                       learning_rate=0.1, 
                       n_estimators=1000)

model.fit(X_tr,y_tr,
          eval_set=[(X_tr,y_tr),(X_tst, y_tst)],
          eval_metric='auc',
          early_stopping_rounds=100,
          verbose=200)

pred_y = model.predict_proba(x_val)[:,1]

