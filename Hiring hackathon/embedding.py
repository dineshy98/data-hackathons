# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:06:31 2020

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


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['day'] = data['InvoiceDate'].dt.day
data['month'] = data['InvoiceDate'].dt.month
data['year'] = data['InvoiceDate'].dt.year


['Invoice No','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']

data.drop(columns = ['InvoiceNo','InvoiceDate'],inplace= True)


from sklearn.preprocessing import LabelEncoder
enc1 = LabelEncoder()
enc2 = LabelEncoder()
enc3 = LabelEncoder()
enc4 = LabelEncoder()

data['StockCode'] = enc1.fit_transform(data['StockCode'])
data['Description'] = enc2.fit_transform(data['Description'])
data['CustomerID'] = enc3.fit_transform(data['CustomerID'])
data['Country'] = enc4.fit_transform(data['Country'])


train1= data[data['UnitPrice'] != -1]
test1 = data[data['UnitPrice'] == -1]



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train1.drop(columns = ['UnitPrice']),train1['UnitPrice'],train_size=0.8,)



sns.distplot(data[data['StockCode'] == 3681]['UnitPrice'])







pred_by_emb = pd.read_csv('pred_by_embedd.csv')

' =================================== embeddings =========================================='

import tensorflow as tf

from keras.models import Model
from keras.layers import Embedding,Input,Concatenate,Flatten,Dense
from keras.callbacks import EarlyStopping

monitor = EarlyStopping(monitor = 'val_loss',patience = int(10))


input0 = Input((1,))
input1 = Input((1,))
input2= Input((1,))
input3= Input((1,))
input4= Input((1,))


['StockCode','Description','CustomerID','Country']

drop = list(train1.loc[train1['UnitPrice'] > 4000].index)
train1.drop(drop,inplace = True)


emb_Gender_out = Embedding(input_dim = data['StockCode'].nunique()+50,output_dim = 10)(input0)
emb_Vehicle_Age_out = Embedding(input_dim = data['Description'].nunique()+1,output_dim = 3)(input1)
emb_Vehicle_Damage_out = Embedding(input_dim = data['CustomerID'].nunique()+1,output_dim = 10)(input2)
emb_Region_Code_out = Embedding(input_dim = data['Country'].nunique()+1,output_dim = 5)(input3)


cat_concat = Concatenate()([emb_Gender_out,emb_Vehicle_Age_out,emb_Vehicle_Damage_out])

concate_flat = Flatten()(cat_concat)

cat_concat = Concatenate()([concate_flat,input4])



dense1 = Dense(5,name='Hidden1',activation = 'relu')(cat_concat)
dense2 = Dense(4,name='Hidden2',activation = 'relu')(dense1)
dense3 = Dense(1,name='prediction',activation='linear')(dense2)

model = Model(inputs = [input0,input1,input2,input3,input4],outputs = [dense3])


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
	
model.fit([train1['StockCode'],train1['Description'],train1['CustomerID'],train1['Country'],
           train1['Quantity']],
          [train1['UnitPrice']],
           batch_size=64,
           validation_split = 0.2,
          epochs = 100,
          callbacks = [monitor]
            
)

train.columns

'The Great Indian Hiring Hackathon'.replace(' ','_').lower()

import matplotlib.pyplot as plt
from keras.utils import plot_model


plot_model(model, to_file='embedding model.png')
data = plt.imread('embedding model.png')
plt.imshow(data)
plt.show()

pred = model.predict([test1['StockCode'],test1['Description'],test1['CustomerID'],test1['Country'],test1['Quantity']])

sub = sample.copy()

sub['UnitPrice'] = pred
sub1.to_csv('pred_by_emb_without_outlier_zeros.csv',index = False)

def negative_correction(x):
    temp = []
    for i in list(x):
            if i>= 0:
               temp.append(i)
            else:
               temp.append(0)
    return temp
    


sub1 = sub.copy()
sub1['UnitPrice'] = negative_correction(sub1['UnitPrice'])

sub1















