# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:21:08 2020

@author: dineshy86
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('S:/Codes/DataHackthon/Machinehack/#16 Playstore App Downloads Prediction/train.csv')
test= pd.read_csv('S:/Codes/DataHackthon/Machinehack/#16 Playstore App Downloads Prediction/test.csv')
test['Downloads'] = -1
sample= pd.read_csv('S:/Codes/DataHackthon/Machinehack/#16 Playstore App Downloads Prediction/sample_submission.csv')
                    
                    
data = pd.concat([train,test])

#preprocess + finding mean
import re

def prepro_sizes1(d):
    d = d.replace(',','')
    if re.findall("[a-zA-Z]+", d)[0] == 'M':
       d = float(d[:-1])*1024
    elif re.findall("[a-zA-Z]+", d)[0] == 'k':
       d = float(d[:-1])
    return d
    
temp_down = data['Size'].apply(prepro_sizes1)
varies_index = data[data['Size'] == 'Varies with device'].index
size_mean = temp_down.drop(index = varies_index).mean()
print(size_mean)

#preprocess + imputing mean
def prepro_sizes2(d):
    d = d.replace(',','')
    if re.findall("[a-zA-Z]+", d)[0] == 'M':
       d = float(d[:-1])*1024
    elif re.findall("[a-zA-Z]+", d)[0] == 'k':
       d = float(d[:-1])
    elif d == 'Varies with device':
       d = size_mean
    
    return d

data['Size'] = data['Size'].apply(prepro_sizes2)

data['Price'].replace({'Free':0},inplace = True)


#target encoding

train_mod = train.copy()

def cat_to_reg(inp):
    inp = inp.replace(',','')
    inp = inp.replace('+','')
    return inp

train_mod['Downloads'] = train_mod['Downloads'].apply(cat_to_reg)


data['Price'] = data['Price'].replace({'Free' : 0},inplace = True)
data['Price'] = pd.to_numeric(data['Price'])
data['Last_Updated_On'] = pd.to_datetime(data['Last_Updated_On'])
data['week'] = data['Last_Updated_On'].dt.week
data['year'] = data['Last_Updated_On'].dt.year
data.drop(columns = ['Last_Updated_On'],inplace = True)
data.drop(columns = ['Offered_By','Release_Version'],inplace = True)


dummies_cols = ['Category','Content_Rating','OS_Version_Required']
dummies_data = pd.get_dummies(data[dummies_cols])

data2 = pd.concat([data,dummies_data],axis = 1)
data2.drop(columns = dummies_cols,inplace = True)



train_mod.append(train_mod[train_mod['Downloads'] == '5,000,000,000+'])




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


test_mod = data2[data2['Downloads'] == -1]
train_mod = data2[data2['Downloads'] != -1]
train_mod.drop(index = train_mod[train_mod['Downloads'] == '5,000,000,000+'].index,inplace = True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_mod.drop(columns = ['Downloads']),train_mod['Downloads'],stratify = train_mod['Downloads'])

import xgboost
xgbc = xgboost.XGBClassifier(objective = "multi:softmax")

xgbc.fit(X_train,y_train)

from sklearn.metrics import log_loss,accuracy_score
y_pred = pd.DataFrame(xgbc.predict_proba(X_test),columns = xgbc.classes_)
log_loss(y_test,xgbc.predict_proba(X_test))
log_loss(y_test,y_pred)

xgbc.

sub = pd.DataFrame(xgbc.predict_proba(test_mod.drop(columns = ['Downloads'])),columns = xgbc.classes_)
sub2 = sub[['10+','50+','100+','500+','1,000+','5,000+','10,000+','50,000+','100,000+','500,000+','1,000,000+','5,000,000+','10,000,000+',
     '50,000,000+','100,000,000+', '500,000,000+','1,000,000,000+']]
sub2['5,000,000,000+'] = 0

sub2.to_csv('xgb_basic.csv',index = False)
