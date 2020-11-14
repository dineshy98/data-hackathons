# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:48:10 2020

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


y = train_mod['Crop_Damage']
X = train_mod.drop(columns = ['ID','Crop_Damage'])
test_x = test_mod.drop(columns = ['ID','Crop_Damage'])


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.20, random_state = 10,stratify = y)



from catboost import CatBoostClassifier,Pool

cat_features = ['Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category','Season', 'Crop_Soil',
       'Enviroment_conditions', 'pest_crop']

train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=cat_features)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=cat_features)

# Initialize CatBoostClassifier

model = CatBoostClassifier(iterations=100,
                           cat_features = cat_features,
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


