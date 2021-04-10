# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:00:59 2021

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
data['profession'] = data['profession'].apply(lambda x : x.lower().replace(' ','_'))
data['city'] = data['city'].apply(lambda x : x.lower().replace(' ','_'))
data['state'] = data['state'].apply(lambda x : x.lower().replace(' ','_'))


drop_cols = ['id']
data.drop(columns = drop_cols,inplace = True)
train_df = data.loc[data['risk_flag'] != -1]
test_df = data.loc[data['risk_flag'] == -1]



all_vals = []
for j in list(train_df.index):
    val = ''
    for i in train_df.drop(columns = ['risk_flag']).columns:
        val = val + str(train_df[i][j])
    
    all_vals.append(val)
   
ans = dict(zip(all_vals,train_df['risk_flag']))
    


all_vals_test = []
for j in list(test_df.index):
    val = ''
    for i in test_df.drop(columns = ['risk_flag']).columns:
        val = val + str(test_df[i][j])
    
    all_vals_test.append(val)



test_df['key'] = all_vals_test

def checkKey(dict, key):
      
    if key in dict.keys():
        print("Present, ", end =" ")
        print("value =", dict[key])
    else:
        print("Not present")
        
checkKey(ans, '73930905919singlerentednogeologistmaldawest_bengal413')


pred= []
for i in test_df['key']:
    if checkKey(ans, i) == 'Present':
        pred.append(ans[i])
    else:
        pred.append(-1)        
        


''+' fd'
    

train['all'] = 

