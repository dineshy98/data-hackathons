
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample_submission_iA3afxn.csv.csv')
test['Response'] = -1

data = pd.concat([train,test])


train.columns


categorical = list(pd.DataFrame(data.select_dtypes(np.object)).columns)
numerical = list(pd.DataFrame(data.select_dtypes(np.number)).columns)

['Gender', 'Vehicle_Age', 'Vehicle_Damage']

from sklearn.preprocessing import LabelEncoder
enc1 = LabelEncoder()
enc2 = LabelEncoder()
enc3 = LabelEncoder()

data['Gender'] = enc1.fit_transform(data['Gender'])
data['Vehicle_Age'] = enc2.fit_transform(data['Vehicle_Age'])
data['Vehicle_Damage'] = enc3.fit_transform(data['Vehicle_Damage'])

data.drop(columns = ['id'],inplace= True)

train1= data[data['Response'] != -1]
test1 = data[data['Response'] == -1]



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train1.drop(columns = ['Response']),train1['Response'],train_size=0.8,
                                                  stratify = train1['Response'])

train.columns.count


' =================================== embeddings =========================================='



from keras.models import Model
from keras.layers import Embedding,Input,Concatenate,Flatten,Dense
from keras.callbacks import EarlyStopping

monitor = EarlyStopping(verbose = 2,monitor = 'val_loss',patience = 5)


input0 = Input((1,))
input1 = Input((1,))
input2= Input((1,))
input3= Input((1,))
input4= Input((1,))
input5= Input((1,))
input6= Input((1,))
input7 = Input((1,))
input8 = Input((1,))
input9 = Input((1,))

['Gender', 'Vehicle_Age', 'Vehicle_Damage','Region_Code']

emb_Gender_out = Embedding(input_dim = train1['Gender'].nunique()+1,output_dim = 3)(input0)
emb_Vehicle_Age_out = Embedding(input_dim = train1['Vehicle_Age'].nunique()+1,output_dim = 3)(input1)
emb_Vehicle_Damage_out = Embedding(input_dim = train1['Vehicle_Damage'].nunique()+1,output_dim = 3)(input2)
emb_Region_Code_out = Embedding(input_dim = train1['Region_Code'].nunique()+1,output_dim = 3)(input3)


cat_concat = Concatenate()([emb_Gender_out,emb_Vehicle_Age_out,emb_Vehicle_Damage_out])

concate_flat = Flatten()(cat_concat)

cat_concat = Concatenate()([concate_flat,input3,input4,input5,input6,input7,input8,input9,input0])



dense1 = Dense(12,name='Hidden1',activation = 'relu')(cat_concat)
dense2 = Dense(8,name='Hidden2',activation = 'relu')(dense1)
dense3 = Dense(1,name='prediction',activation = 'softmax')(dense2)

model = Model(inputs = [input0,input1,input2,input3,input4,input5,input6,input7,input8,input9],outputs = [dense3])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
model.fit([train1['Gender'],train1['Vehicle_Age'],train1['Vehicle_Damage'],train1['Region_Code'],
           
           train1['Driving_License'],train1['Previously_Insured'],
           train1['Annual_Premium'],train1['Age'],train1['Policy_Sales_Channel'],train1['Vintage']],
          [train1['Response']],
           batch_size=32,
          epochs = 200
            
)

train.columns


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



















