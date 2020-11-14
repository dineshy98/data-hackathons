# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:12:36 2020

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
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample.csv')
test['Segmentation'] = -1
test['source'] = 'test'
train['source'] = 'train'

data = pd.concat([train,test])


CATEGORICAL = ['Gender','Ever_Married', 'Graduated', 'Profession','Family_Size', 'Var_1','Spending_Score']
NUMERICAL = ['Work_Experience','Age']

data[CATEGORICAL] = data[CATEGORICAL].astype(str)

for i in CATEGORICAL:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])
    
twmp = pd.get_dummies(data, columns = CATEGORICAL)
data = twmp



from sklearn.impute import KNNImputer
knn= KNNImputer()
data.drop(columns = ['Segmentation','ID']) = knn.fit_transform(data.drop(columns = ['Segmentation','ID'])).shape



test1 = data[data['source'] == 'test']
train1 = data[data['source'] == 'train']

test1.drop(columns = ['Segmentation','source'],inplace= True)
train1.drop(columns = ['source'],inplace= True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train1.drop(columns = ['Segmentation']),train1['Segmentation'],train_size=0.8,
                                                  stratify = train['Segmentation'])


from sklearn.impute import KNNImputer
knn= KNNImputer()
X_train = knn.fit_transform(X_train)
X_test = knn.fit_transform(X_test)


from lightgbm import LGBMClassifier
model2 = LGBMClassifier(n_estimators=300, max_features = .85, max_depth = 15, learning_rate = 1.1).fit(X_train, y_train)
accuracy_score(y_test,model2.predict(X_test))








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




model4 = pipeline.make_pipeline(impute.KNNImputer(n_neighbors = 10), ensemble.RandomForestClassifier(class_weight = 'balanced_subsample',
                    n_estimators = 200, max_depth = 20, criterion = 'entropy', max_features = .8, oob_score = True, random_state = 0)).fit(X, y)




