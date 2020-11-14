# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:28:39 2020

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


train.columns


CATEGORICAL = ['Gender', 'Ever_Married', 'Graduated', 'Profession','Family_Size', 'Var_1','Spending_Score']
NUMERICAL = ['Work_Experience','Age']

data['Ever_Married'].fillna(data['Ever_Married'].mode()[0],inplace = True)
data['Graduated'].fillna(data['Graduated'].mode()[0],inplace = True)
data['Profession'].fillna(data['Profession'].mode()[0],inplace = True)
data['Work_Experience'].fillna(data['Work_Experience'].mode()[0],inplace = True)
data['Family_Size'].fillna(data['Family_Size'].mode()[0],inplace = True)
data['Var_1'].fillna(data['Var_1'].mode()[0],inplace = True)

from sklearn.preprocessing import LabelEncoder
enc1 = LabelEncoder()
enc2 = LabelEncoder()
enc3 = LabelEncoder()
enc4 = LabelEncoder()
enc5 = LabelEncoder()
enc6 = LabelEncoder()

data['Gender'] = enc1.fit_transform(data['Gender'])
data['Ever_Married'] = enc1.fit_transform(data['Ever_Married'])
data['Graduated'] = enc1.fit_transform(data['Graduated'])
data['Profession'] = enc1.fit_transform(data['Profession'])
data['Var_1'] = enc1.fit_transform(data['Var_1'])
data['Spending_Score'] = enc1.fit_transform(data['Spending_Score'])

data.drop(columns = ['ID'],inplace= True)

test1 = data[data['source'] == 'test']
train1 = data[data['source'] == 'train']

test1.drop(columns = ['Segmentation','source'],inplace= True)
train1.drop(columns = ['source'],inplace= True)




'-------------------------------------- EDA --------------------------------------------------------'

sns.distplot(train1['Segmentation'])
sns.pairplot(train1,hue = 'Segmentation')

sns.scatterplot(x = 'Work_Experience',y = 'Age',data =train1,hue = 'Segmentation')



















from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train1.drop(columns = ['Segmentation']),train1['Segmentation'],train_size=0.8,
                                                  stratify = train['Segmentation'])

# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, dtree_predictions)

accuracy_score(y_test, dtree_predictions)










from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'poly', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix 
cm_svc = confusion_matrix(y_test, svm_predictions) 












# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
  
# accuracy on X_test 
knn.score(X_test, y_test) 
print accuracy 
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 






# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
gnb.score(X_test, y_test) 
print accuracy 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions) 








