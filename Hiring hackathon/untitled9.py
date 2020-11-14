# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:08:26 2020

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



train.columns
eda = pd.DataFrame()
eda['mean'] = train.groupby('StockCode').mean()['UnitPrice']
eda['std'] = train.groupby('StockCode').std()['UnitPrice']

eda['StockCode'] = eda.index

data[data['StockCode'] == 1512][['UnitPrice','InvoiceDate']].set_index('InvoiceDate').plot()

only_test = test[~test['StockCode'].isin(list(set(train['StockCode'])))]


from tqdm import tqdm

testorder = list(test['StockCode'])
ans = []
for i in tqdm(testorder):
    temp = eda.loc[i]['mean']
    ans.append(temp)

eda.loc[3516]['mean']

ans[0].item()
sub = sample.copy()
sub['UnitPrice'] = ans
sub.to_csv('pred_by_only_stockembedd5.csv',index = False)

train.loc[train['CustomerID']== 3516].loc[train['Description'] == 3738]
train.columns

dict(data[data['StockCode'] == 3516].items())
train[
 'StockCode': 1565    3516
 Name: StockCode, dtype: int64,
 'Description': 1565    3738
 Name: Description, dtype: int64,
 'Quantity': 1565    1
 Name: Quantity, dtype: int64,
 'InvoiceDate': 1565    2011-11-24 14:40:00
 Name: InvoiceDate, dtype: object,
 'UnitPrice': 1565   -1.0
 Name: UnitPrice, dtype: float64,
 'CustomerID': 1565    17364.0
 Name: CustomerID, dtype: float64,
 'Country': 1565    35
 Name: Country, dtype: int64]






from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(series, order=(2,1,0))
model_fit = model.fit(disp=0)

import tensorflow as tf

from keras.models import Model
from keras.layers import Embedding,Input,Concatenate,Flatten,Dense
from keras.callbacks import EarlyStopping

monitor = EarlyStopping(monitor = 'val_loss',patience = int(10))


input0 = Input((1,))


['StockCode','Description','CustomerID','Country']




stock_strenght = Embedding(input_dim = data['StockCode'].nunique()+50,output_dim = 5)(input0)

dense1 = Dense(5,name='Hidden1',activation = 'relu')(stock_strenght)
dense2 = Dense(4,name='Hidden2',activation = 'relu')(dense1)
dense3 = Dense(1,name='prediction',activation='linear')(dense2)

model1 = Model(inputs = [input0],outputs = [dense3])


model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model1.fit([train['StockCode']],
          [train['UnitPrice']],
           batch_size=64,
           validation_split = 0.2,
          epochs = 100,
          callbacks= [monitor]
            
)

train.columns

'The Great Indian Hiring Hackathon'.replace(' ','_').lower()

import matplotlib.pyplot as plt
from keras.utils import plot_model


plot_model(model1, to_file='embedding model.png')
data = plt.imread('embedding model.png')
plt.imshow(data)
plt.show()

pred = model1.predict([test['StockCode']])

sub = sample.copy()


pred = pred.reshape(pred.shape[0],pred.shape[1])
sub['UnitPrice'] = pred
sub.to_csv('pred_by_only_stockembedd5.csv',index = False)