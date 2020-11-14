# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:31:41 2020

@author: dineshy86
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/train.csv')
test= pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/test.csv')
sample= pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/sample.csv')
test['Crop_Damage'] = -1

train.isnull().sum()

train['Number_Weeks_Used'].fillna(train['Number_Weeks_Used'].mean(),inplace = True)
test['Number_Weeks_Used'].fillna(test['Number_Weeks_Used'].mean(),inplace = True)



data = pd.concat([train,test])

data['Crop_Soil'] = data['Crop_Type']*data['Soil_Type']
data['Enviroment_conditions'] = (data['Soil_Type'].replace({0:4})) * data['Season']
data['pest_crop'] = (data['Crop_Type'].replace({0:4}))* data['Pesticide_Use_Category']
data['Total_Doses'] = data['Number_Weeks_Used']*data['Number_Doses_Week']

train_mod = data[data['Crop_Damage'] > -1]
test_mod = data[data['Crop_Damage'] < 0]

y1 = train['Crop_Damage']
X1 = train.drop(columns = ['ID','Crop_Damage'])
test_x1 = test.drop(columns = ['ID','Crop_Damage'])

y = train_mod['Crop_Damage']
X = train_mod.drop(columns = ['ID','Crop_Damage'])
test_x = test_mod.drop(columns = ['ID','Crop_Damage'])




cat_features = ['Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category','Season', 'Crop_Soil',
       'Enviroment_conditions', 'pest_crop']


from catboost import CatBoostClassifier,Pool
model = CatBoostClassifier(iterations=100,
                           depth=2,
                           loss_function='MultiClassOneVsAll'
                           )

parameters = {'depth'         : [6,8,10],
                  'learning_rate' : [0.01, 0.05, 0.1],
                  'iterations'    : [30, 50]
                 }






















('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))






from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


estimators = [
    ('rf', RandomForestClassifier(n_estimators=1000, random_state=42)),
    ('knn',KNeighborsClassifier()),
    ('cat',model),
    ('gauss',GaussianNB())
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(max_iter =10000, random_state=42),verbose = 2,
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)


pd.Series(train_mod[['Crop_Type','Crop_Damage']].groupby(['Crop_Type']).count()/88858)


clf.fit(X_train, y_train)


submission = sample.copy()
submission['Crop_Damage'] = bc.predict(test_x)
submission.to_csv('bc1.csv',index = False)


from category_encoders import TargetEncoder
encoder = TargetEncoder()
t = encoder.fit_transform(train_mod['Crop_Type'], train_mod['Crop_Damage'])


from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(base_estimator =XGBClassifier(750),
                  n_estimators = 15,
                  verbose= 20,
                  bootstrap = False,
                  max_features = 1
                )

bc.fit(X, y)