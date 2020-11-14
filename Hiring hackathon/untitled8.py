# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:57:32 2020

@author: dineshy86
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('Train.csv')
test= pd.read_csv('Test.csv')
sample= pd.read_csv('sample.csv')


train.describe()
train.isnull().sum()
train.info()




data = pd.concat([train,test])

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['day'] = data['InvoiceDate'].dt.day
data['month'] = data['InvoiceDate'].dt.month
data['year'] = data['InvoiceDate'].dt.year
    
train.set_index(pd.to_datetime(train['InvoiceDate'])).resample('D').mean()['UnitPrice'].plot()
