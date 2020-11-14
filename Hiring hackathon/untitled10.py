# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:03:34 2020

@author: dineshy86
"""




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample = pd.read_csv('Sample.csv')
test['UnitPrice'] = -1

data = pd.concat([train,test])
train['InvoiceDate'] = pd.to_datetime(train['InvoiceDate'])
eda = pd.DataFrame()
eda['mean'] = train.groupby('StockCode').mean()['UnitPrice']
eda['std'] = train.groupby('StockCode').std()['UnitPrice']


more_var_codes = eda.sort_values(by = ['std'],ascending = False).index[:10]

for i in more_var_codes:
    locals()['train_{}'.format(i)] = train.loc[train['StockCode'] == i]

train_1366.sort_values(by = 'InvoiceDate' , inplace = True)

train_1366.index = train_1366.InvoiceDate

train_1366['UnitPrice'].plot()
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train_1366['UnitPrice'], order=(1,1,1))
model_fit = model.fit(disp=0)

