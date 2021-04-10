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


for lisst in [
        ['married','house_ownership'],
        ['house_ownership','car_ownership'],
        ['house_ownership','car_ownership','profession'],
        ['profession','city']
        ]:

    
    if len(lisst) == 2:
       data['_'.join(lisst)] = data[lisst[0]] +'_'+ data[lisst[1]]
    
    elif len(lisst) == 3:
        data['_'.join(lisst)] = data[lisst[0]] +  '_' + data[lisst[1]] + '_' + data[lisst[2]]




data.columns

cat_cols = [ 'married', 'house_ownership',
       'car_ownership', 'profession', 'city', 'state',
       'married_house_ownership',
       'house_ownership_car_ownership',
       'house_ownership_car_ownership_profession', 'profession_city']


drop_cols = ['id' ]





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score,f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = drop_cols,inplace = True)


import category_encoders as ce
encd = ce.cat_boost.CatBoostEncoder(cols=cat_cols,return_df=True)
data1 = encd.fit_transform(data,data['risk_flag'])


sec_lev_data = pd.concat([data.loc[data['risk_flag'] == 0].sample(10000),data.loc[data['risk_flag'] == 1].sample(10000)])


test_df = data.loc[data['risk_flag'] == -1].drop(columns = ['risk_flag'])
data.drop(data.loc[data['risk_flag'] == -1].index,inplace = True)


data.drop(sec_lev_data.index,inplace = True)
data.reset_index(drop = True,inplace = True)


data0_cat = data[data['risk_flag'] == 0].reset_index(drop = True)
data1_cat = data[data['risk_flag'] == 1].reset_index(drop = True)


#cat_enabled_models

data['risk_flag'].value_counts()

def make_datasets(df):
    
    test_df = df.loc[df['risk_flag'] == -1].drop(columns = ['risk_flag'])
    df0 = df[df['risk_flag'] == 0].reset_index(drop = True)
    df1 = df[df['risk_flag'] == 1].reset_index(drop = True)
    
    return test_df,df0,df1


for i in range(4):
    
    globals()['data0_cat%s' % i] = data0_cat[20996*i:20996*(i+1)]


for i in range(4):
    
    globals()['train_df1_cat%s' % i] = pd.concat([globals()['data0_cat%s' % i],data1_cat])
    globals()['train_df1_cat%s' % i] = globals()['train_df1_cat%s' % i].sample(frac=1).reset_index(drop=True)




#for i in range(4):
 #   globals()['train_df1_cat%s' % i].drop(columns = drop_cols,inplace = True)
    


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier





for i in range(4):
    globals()['X_tr_cat%s' % i],globals()['X_tst_cat%s' % i],globals()['y_tr_cat%s' % i],globals()['y_tst_cat%s' % i] = train_test_split(globals()['train_df1_cat%s' % i].drop(columns = ['risk_flag']),globals()['train_df1_cat%s' % i]['risk_flag'],stratify = globals()['train_df1_cat%s' % i]['risk_flag'])



score = {}

from catboost import CatBoostClassifier
for i in range(4):
    
    globals()['catboost%s' % i]=CatBoostClassifier(iterations = 2000,eval_metric = 'F1')
    globals()['catboost%s' % i].fit(globals()['X_tr_cat%s' % i], globals()['y_tr_cat%s' % i],cat_features=cat_cols,eval_set=(globals()['X_tst_cat%s' % i], globals()['y_tst_cat%s' % i]))
    score['catboost{}'.format(i)] = roc_auc_score(globals()['y_tst_cat%s' % i],globals()['catboost%s' % i].predict(globals()['X_tst_cat%s' % i]))

############### LEVEL 2 ###############################



#creating level2 training set

preds = pd.DataFrame()
for i in range(4):
    
    preds['catboost{}'.format(i)] = globals()['catboost%s' % i].predict_proba(sec_lev_data.drop(columns = ['risk_flag']))[:,1]



sec_lev_data.reset_index(inplace = True,drop = True)
preds['risk_flag'] = sec_lev_data['risk_flag']
preds = preds.sample(frac= 1)


from catboost import CatBoostClassifier

catboost5 = CatBoostClassifier(iterations = 2000,eval_metric = 'F1')

X_tr_lev2,X_tst_lev2,y_tr_lev2,y_tst_lev2 = train_test_split(preds.drop(columns = ['risk_flag']),preds['risk_flag'],stratify = preds['risk_flag'])


from catboost import CatBoostClassifier
catboost5=CatBoostClassifier(iterations = 2000,eval_metric = 'F1')
catboost5.fit(X_tr_lev2, y_tr_lev2,eval_set=(X_tst_lev2, y_tst_lev2))







#predicting test


test_preds = pd.DataFrame()
for i in range(4):
    
    test_preds['catboost{}'.format(i)] = globals()['catboost%s' % i].predict_proba(test_df)[:,1]

sample['risk_flag'] =  catboost5.predict(test_preds)
sample.to_csv('catboost_ensembled_4stacking.csv',index = False)


counts = pd.DataFrame()
counts['1s'] = preds.sum(axis = 1)
counts['0s'] = 4 - counts['1s']


sample['risk_flag'] = counts['1s'].apply(lambda x : 1 if x > 3 else 0)

sample['risk_flag'].value_counts()


sample2 = pd.read_csv('catboost_ensembled_frop_duplicated_4_3_1s.csv')



confusion_matrix(sample['risk_flag'],sample2['risk_flag'])





sample['risk_flag'] = catboost0.predict(test_df.drop(columns = ['risk_flag']))
sample.to_csv('catboost.csv',index = False)



    
    






















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


