# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:29:10 2021

@author: dineshy86
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:17:03 2021

@author: dineshy86
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:09:32 2021

@author: dineshy86
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


test = pd.read_csv('Test Data.csv')
test['risk_flag'] = -1
sample = pd.read_csv('Sample Prediction Dataset.csv')
train = pd.read_csv('Training Data.csv')


train.columns
train['risk_flag'].value_counts()
data = pd.concat([train,test])



#feature engineering
data.isnull().sum()


#1.feature_intraction

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state']




data.columns

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state']


drop_cols = ['id' ]





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score,f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = drop_cols,inplace = True)

for i in cat_cols:
    data[i] = data[i].apply(lambda x : str(x).lower().replace('[','').replace(']','').replace(',',''))

data1 = pd.get_dummies(data,columns = cat_cols)

data1['risk_flag'].value_counts()


test_df = data1.loc[data1['risk_flag'] == -1].drop(columns = ['risk_flag'])
data1.reset_index(inplace = True,drop = True)
data1.drop(data1.loc[data1['risk_flag'] == -1].index,inplace = True)


data0_num = data1[data1['risk_flag'] == 0].reset_index(drop = True)
data1_num = data1[data1['risk_flag'] == 1].reset_index(drop = True)




for i in range(4):
    
    globals()['data0_num%s' % i] = data0_num[30996*i:30996*(i+1)]


for i in range(4):
    
    globals()['train_df1_num%s' % i] = pd.concat([globals()['data0_num%s' % i],data1_num])
    globals()['train_df1_num%s' % i] = globals()['train_df1_num%s' % i].sample(frac=1).reset_index(drop=True)




#for i in range(4):
 #   globals()['train_df1_num%s' % i].drop(columns = drop_cols,inplace = True)
    


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier




for i in range(4):
    globals()['X_tr_num%s' % i],globals()['X_tst_num%s' % i],globals()['y_tr_num%s' % i],globals()['y_tst_num%s' % i] = train_test_split(globals()['train_df1_num%s' % i].drop(columns = ['risk_flag']),globals()['train_df1_num%s' % i]['risk_flag'],stratify = globals()['train_df1_num%s' % i]['risk_flag'])



score = {}

from sklearn.ensemble import RandomForestClassifier
for i in range(4):
    
    globals()['rfc%s' % i]=RandomForestClassifier(n_estimators = 4000)
    globals()['rfc%s' % i].fit(globals()['X_tr_num%s' % i], globals()['y_tr_num%s' % i])
    score['rfc{}'.format(i)] = roc_auc_score(globals()['y_tst_num%s' % i],globals()['rfc%s' % i].predict(globals()['X_tst_num%s' % i]))
    print('rfc{}_trained'.format(i))

score_acc = {}
for i in range(4):
    score_acc['rfc{}'.format(i)] = accuracy_score(globals()['y_tst_num%s' % i],globals()['rfc%s' % i].predict(globals()['X_tst_num%s' % i]))

############### LEVEL 2 ###############################



#creating level2 training set

preds = pd.DataFrame()
for i in range(4):
    preds['rfc{}'.format(i)] = globals()['rfc%s' % i].predict(test_df)

preds['sum'] = preds.sum(axis = 1)


sample['risk_flag'] = preds['sum'].apply(lambda x : 1 if x > 3 else 0)

sample['risk_flag'].value_counts()

sample.to_csv('rfc_ensmbld_dummies.csv',index = False)

preds.to_csv('rfc_ensmbld_dummiespreds.csv',index = False)

preds = pd.read_csv('rfc_ensmbld_dummiespreds.csv')    
    






















train_df = data1.loc[data1['risk_flag'] != -1]
test_df = data1.loc[data1['risk_flag'] == -1]
test_df.drop(columns = ['risk_flag'],inplace = True)





X_tr,X_tst,y_tr,y_tst = train_test_split(train_df.drop(columns = ['risk_flag']),train_df['risk_flag'],stratify = train_df['risk_flag'])

weighted_clf = RandomForestClassifier(max_depth=3 ,random_state=0,class_weight={0:0.2,1:0.8}).fit(X_tr, y_tr)
roc_auc_score(y_tst,weighted_clf.predict(X_tst))






from sklearn.svm import SVC

# we can add class_weight='balanced' to add panalize mistake
svc_model = SVC(class_weight='balanced')

svc_model.fit(X_tr, y_tr)

svc_predict = svc_model.predict(X_tst)# check performance
print('ROCAUC score:',roc_auc_score(y_tst, svc_predict))
print('Accuracy score:',accuracy_score(y_tst, svc_predict))
print('F1 score:',f1_score(y_tst, svc_predict))



