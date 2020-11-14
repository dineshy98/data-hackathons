# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 03:04:22 2020

@author: dineshy86
"""

from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(Dense(30,input_shape = (20,),activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation = 'linear'))

model.compile()
model.fit(X_train,y_train,epochs = 20)
