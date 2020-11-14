# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:10:43 2020

@author: dineshy86
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:13:14 2020

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

data.columns

train.columns
import pandas as pd
data = pd.read_csv('data.csv')

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


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train1.drop(columns = ['Segmentation']),train1['Segmentation'],train_size=0.8,
                                                  stratify = train['Segmentation'])


' =================================== embeddings =========================================='



from keras.models import Model
from keras.layers import Embedding,Input,Concatenate,Flatten,Dense
from keras.callbacks import EarlyStopping

monitor = EarlyStopping(verbose = 2,monitor = 'val_loss',patience = 5)



'Segmentation'


input1 = Input((1,))
input2= Input((1,))
input3= Input((1,))
input4= Input((1,))
input5= Input((1,))
input6= Input((1,))
input7 = Input((1,))

input8 = Input((1,))
input9 = Input((1,))

'Work_Experience','Age'
CATEGORICAL = ['Gender', 'Ever_Married', 'Graduated', 'Profession','Family_Size', 'Var_1','Spending_Score']

emb_Gender_out = Embedding(input_dim = train1['Gender'].nunique()+1,output_dim = 3)(input1)
emb_Ever_Married_out = Embedding(input_dim = train1['Ever_Married'].nunique()+1,output_dim = 3)(input2)
emb_Graduated_out = Embedding(input_dim = train1['Graduated'].nunique()+1,output_dim = 3)(input3)
emb_Profession_out = Embedding(input_dim = train1['Profession'].nunique()+1,output_dim = 3)(input4)
emb_Family_Size_out = Embedding(input_dim = train1['Family_Size'].nunique()+1,output_dim = 3)(input5)
emb_Var_1_out = Embedding(input_dim = train1['Var_1'].nunique()+1,output_dim = 3)(input6)
emb_Spending_Score_out = Embedding(input_dim = train1['Spending_Score'].nunique()+1,output_dim = 3)(input7)

cat_concat = Concatenate()([emb_Gender_out,emb_Ever_Married_out,emb_Graduated_out,emb_Profession_out,
                           emb_Family_Size_out,emb_Var_1_out,emb_Spending_Score_out])

concate_flat = Flatten()(cat_concat)

cat_concat = Concatenate()([concate_flat,input8,input9])



dense1 = Dense(12,name='Hidden1',activation = 'relu')(cat_concat)
dense2 = Dense(8,name='Hidden2',activation = 'relu')(dense1)
dense3 = Dense(4,name='prediction',activation = 'softmax')(dense2)

model = Model(inputs = [input1,input2,input3,input4,input5,input6,input7,input8,input9],outputs = [dense3])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
model.fit([train1['Gender'],train1['Ever_Married'],train1['Graduated'],train1['Profession'],
           train1['Family_Size'],train1['Var_1'],train1['Spending_Score'],train1['Work_Experience'],train1['Age']],
          [pd.get_dummies(train1['Segmentation'])],
           batch_size=64,
          epochs = 200
            
)



import matplotlib.pyplot as plt
from keras.utils import plot_model


plot_model(model, to_file='embedding model.png')
data = plt.imread('embedding model.png')
plt.imshow(data)
plt.show()

pred = np.argmax(model.predict([test1['Gender'],test1['Ever_Married'],test1['Graduated'],test1['Profession'],
           test1['Family_Size'],test1['Var_1'],test1['Spending_Score'],test1['Work_Experience'],
           test1['Age']]),axis = 1)

pd.DataFrame(pred).to_csv('PROB_pred_by_embedd.csv',index = False)

sub = sample.copy()

sub['Segmentation'] = pred
sub['Segmentation'].replace({0:'A',1:'B',2:'C',3:'D'},inplace = True)
sub.to_csv('pred_by_embedd.csv',index = False)







