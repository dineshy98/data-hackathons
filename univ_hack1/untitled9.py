# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:35:51 2021

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




train.columns
train['risk_flag'].value_counts()
train.drop(train.loc[train['risk_flag'] == 0].sample(26300).index,inplace = True)


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
       'car_ownership', 'profession','loc']

data['profession'] = data['profession'].apply(lambda x : x.lower().replace(' ','_'))


drop_cols = ['id']




from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score


for i in cat_cols:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])


data.drop(columns = drop_cols,inplace = True)
train_df = data.loc[data['risk_flag'] != -1]
test_df = data.loc[data['risk_flag'] == -1]



X_tr,X_tst,y_tr,y_tst = train_test_split(train_df.drop(columns = ['risk_flag']),train_df['risk_flag'],stratify = train_df['risk_flag'])




from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X_tr, y_tr)

clf.feature_count_

k = clf.predict(X_tst)
roc_auc_score(y_tst,clf.predict(X_tst))


print(clf.predict(X[2:3]))




