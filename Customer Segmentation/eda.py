# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 19:39:14 2020

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



data['Ever_Married'].fillna(data['Ever_Married'].mode()[0],inplace = True)
data['Graduated'].fillna(data['Graduated'].mode()[0],inplace = True)
data['Profession'].fillna(data['Profession'].mode()[0],inplace = True)
data['Work_Experience'].fillna(data['Work_Experience'].mode()[0],inplace = True)
data['Family_Size'].fillna(data['Family_Size'].mode()[0],inplace = True)
data['Var_1'].fillna(data['Var_1'].mode()[0],inplace = True)


test = data[data['source'] == 'test']
train = data[data['source'] == 'train']

test.drop(columns = ['Segmentation','source'],inplace= True)
train.drop(columns = ['source'],inplace= True)



[ 'Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',
       'Work_Experience', 'Spending_Score', 'Family_Size', 'Var_1',
       'Segmentation']

on = ['Ever_Married', 'Graduated', 'Profession',
        'Spending_Score', 'Var_1']

gen_seg = pd.crosstab(index = train['Gender'],columns = train['Segmentation'])
mar_seg = pd.crosstab(index = train['Ever_Married'],columns = train['Segmentation'])
age_seg = pd.crosstab(index = train['Age'],columns = train['Segmentation'])
grad_seg = pd.crosstab(index = train['Graduated'],columns = train['Segmentation'])
prof_seg = pd.crosstab(index = train['Profession'],columns = train['Segmentation'])
work_seg = pd.crosstab(index = train['Work_Experience'],columns = train['Segmentation'])
fam_seg = pd.crosstab(index = train['Family_Size'],columns = train['Segmentation'])


grp = train.drop(columns = ['ID','Family_Size','Age','Work_Experience']).groupby(['Gender', 'Ever_Married', 'Graduated', 'Profession',
               'Spending_Score', 'Var_1']).sum()

temp = train.drop(columns = ['ID','Family_Size','Gender','Age','Work_Experience']).groupby([ 'Ever_Married', 'Graduated', 'Profession',
               'Spending_Score', 'Var_1']).sum()['Segmentation'].apply(lambda x : x[0]).reset_index()

grp_test = test.drop(columns = ['ID','Family_Size','Gender']).groupby(['Ever_Married', 'Age', 'Graduated', 'Profession',
               'Spending_Score','Work_Experience', 'Var_1']).sum()



temp3 = pd.merge(left = test,right = temp,on = on,how='left').fillna(method='ffill')
temp3['Segmentation'].fillna(temp3['Segmentation'].mode()[0],inplace = True)
temp3.sort()

sub = sample.copy()
sub['Segmentation'] = temp3['Segmentation']
sub.to_csv('eda.csv',index = False)
