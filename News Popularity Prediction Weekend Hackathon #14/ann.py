# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:20:49 2020

@author: dineshy86
"""
import pandas as pd
Df = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample = pd.read_csv('sample_submission.csv')


from sklearn.preprocessing import scale as s
from sklearn.model_selection import train_test_split as t

X_train,X_test,y_train,y_test = t(Df.drop(columns = ['shares']),Df['shares'],test_size=0.2,random_state=0)




from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error,mean_squared_error
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


nn = Sequential()

#Input Layer
nn.add(Dense(58,input_dim = X_train.shape[1],kernel_initializer='normal',activation='relu'))

#Hidden Layer(s)
nn.add(Dense(200,activation='relu'))
nn.add(Dense(100,activation='relu'))
nn.add(Dense(50,activation='relu'))
nn.add(Dense(20,activation='relu'))


#Output Layer
nn.add(Dense(1,activation='linear'))

#Compilation
nn.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
nn.summary()


history = nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.2, verbose=1)
loss = history.history['loss']

from keras.optimizers import SGD
opt = SGD(lr=0.5,momentum=0.9)
nn.compile(loss='mean_squared_error',optimizer=opt)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,nn.predict(X_test))


sub = sample.copy()

sub['shares'] = nn.predict(test)
sub.to_csv('nn.csv',index = False)

nn.evaluate(test_x,test_y)
