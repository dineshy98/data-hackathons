# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:11:04 2020

@author: dineshy86
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample.csv')
test['Segmentation'] = -1
test['source'] = 'test'
train['source'] = 'train'

data = pd.concat([train,test])


CATEGORICAL = ['Gender', 'Ever_Married', 'Graduated', 'Profession','Family_Size', 'Var_1','Spending_Score']
NUMERICAL = ['Work_Experience','Age']

data['Ever_Married'].fillna(data['Ever_Married'].mode()[0],inplace = True)
data['Graduated'].fillna(data['Graduated'].mode()[0],inplace = True)
data['Profession'].fillna(data['Profession'].mode()[0],inplace = True)
data['Work_Experience'].fillna(data['Work_Experience'].mode()[0],inplace = True)
data['Family_Size'].fillna(data['Family_Size'].mode()[0],inplace = True)
data['Var_1'].fillna(data['Var_1'].mode()[0],inplace = True)

from sklearn.preprocessing import LabelEncoder
enc1 = LabelEncoder()
enc2 = LabelEncoder()
enc3 = LabelEncoder()
enc4 = LabelEncoder()
enc5 = LabelEncoder()
enc6 = LabelEncoder()

data['Gender'] = enc1.fit_transform(data['Gender'])
data['Ever_Married'] = enc1.fit_transform(data['Ever_Married'])
data['Graduated'] = enc1.fit_transform(data['Graduated'])
data['Profession'] = enc1.fit_transform(data['Profession'])
data['Var_1'] = enc1.fit_transform(data['Var_1'])
data['Spending_Score'] = enc1.fit_transform(data['Spending_Score'])

data.drop(columns = ['ID'],inplace= True)

data["Work_Experience"] = data["Work_Experience"].astype(np.int) 
data["Family_Size"] = data["Family_Size"].astype(np.int)

test1 = data[data['source'] == 'test']
train1 = data[data['source'] == 'train']

test1.drop(columns = ['Segmentation','source'],inplace= True)
train1.drop(columns = ['source'],inplace= True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train1.drop(columns = ['Segmentation']),train1['Segmentation'],train_size=0.8,
                                                  stratify = train['Segmentation'])

'----------------------   light gbm ----------------------------------------------------------------'

lgbm=LGBMClassifier(random_state=22,n_jobs=-1,max_depth=-1,min_data_in_leaf=17,num_leaves=67,
                   colsample_bytree=0.9,bagging_fraction=0.1,lambda_l2=1.1,n_estimators=5000)

lgbm.fit(X_train, y_train, eval_metric="logloss", eval_set=[(X_test,y_test)], verbose=True,early_stopping_rounds=100)
lgbm.classes_

'--------------------     cat_boost   ----------------------------------------------------------------'

CATEGORICAL = ['Gender', 'Ever_Married', 'Graduated', 'Profession','Family_Size', 'Var_1','Spending_Score']
NUMERICAL = ['Work_Experience','Age']

from catboost import CatBoostClassifier,Pool

train_dataset = Pool(data=X_train,label=y_train,cat_features=CATEGORICAL)
eval_dataset = Pool(data=X_test,label=y_test,cat_features=CATEGORICAL)

model = CatBoostClassifier(iterations=1000,
                           cat_features = CATEGORICAL,
                           depth=2,
                           loss_function='MultiClassOneVsAll'
                           )

model.fit(train_dataset,eval_set=eval_dataset)

preds_class = model.predict(eval_dataset)
preds_proba = model.predict_proba(eval_dataset)

model.classes_


' -----------------------  Naive Bayes classifier  --------------------------------------------'

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb.classes_

' -------------------------------------KNN classifier  -------------------------------------------'

# training a 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

knn.classes_

'--------------------- stacking ---------------------'

lgbm_prob = lgbm.predict_proba(X_test)
cat_prob= model.predict_proba(X_test)
gnb_prob= gnb.predict_proba(X_test)
knn_prob= knn.predict_proba(X_test)

pred = 0.30*lgbm_prob+0.10*cat_prob+0.5*gnb_prob+0.10*knn_prob

pd.Series(np.argmax(pred,axis = 1)).replace({0:'A',1:'B',2:'C',3:'D'})

accuracy_score(y_test,pd.Series(np.argmax(pred,axis = 1)).replace({0:'A',1:'B',2:'C',3:'D'}))
