# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:58:45 2020

@author: dineshy86
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample.csv')
test['Segmentation'] = -1
test['source'] = 'test'
train['source'] = 'train'

data = pd.concat([train,test])


train.columns


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




CATEGORICAL = ['Gender', 'Ever_Married', 'Graduated', 'Profession','Family_Size', 'Var_1','Spending_Score']
NUMERICAL = ['Work_Experience','Age']

from catboost import CatBoostClassifier,Pool

train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=CATEGORICAL)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=CATEGORICAL)

# Initialize CatBoostClassifier

model = CatBoostClassifier(iterations=1000,
                           cat_features = CATEGORICAL,
                           depth=2,
                           loss_function='MultiClassOneVsAll'
                           )

parameters = {'depth'         : [6,8,10],
                  'learning_rate' : [0.01, 0.05, 0.1],
                  'iterations'    : [30, 50]
                 }

model._tune_hyperparams(parameters,train_dataset)



model.fit(train_dataset,
          eval_set=eval_dataset)

preds_class = model.predict(eval_dataset)
preds_proba = model.predict_proba(eval_dataset)


preds_raw = model.predict(eval_dataset, 
                          prediction_type='RawFormulaVal')

model.predict_proba(test_x)


submission = sample.copy()
submission['Crop_Damage'] = model.predict(test_x)
submission.to_csv('cat3.csv',index = False)