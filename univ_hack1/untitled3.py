# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 01:37:06 2021

@author: dineshy86
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:38:35 2021

@author: dineshy86
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:38:21 2021

@author: dineshy86
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


test = pd.read_csv('Test Data.csv')
test['risk_flag'] = -1
sample = pd.read_csv('Sample Prediction Dataset.csv')
train = pd.read_csv('Training Data.csv')

train.drop(columns = ['id'],inplace = True)
train.drop_duplicates(inplace = True)
train['id'] = -1

train.columns
train['risk_flag'].value_counts()


data = pd.concat([train,test])
data['risk_flag'].value_counts()



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



#creating holdout dataset





data0 = data.loc[data['risk_flag'] == 0]
data1 = data.loc[data['risk_flag'] == 1]

data.drop(columns = drop_cols,inplace = True)

for i in range(7):
    
    globals()['data0%s' % i] = data0[8500*i:8500*(i+1)]


for i in range(7):
    
    globals()['train_df1%s' % i] = pd.concat([globals()['data0%s' % i],data1])
    globals()['train_df1%s' % i] = globals()['train_df1%s' % i].sample(frac=1).reset_index(drop=True)

for i in range(7):
    globals()['train_df1%s' % i].drop(columns = drop_cols,inplace = True)
    


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier





for i in range(7):
    globals()['X_tr%s' % i],globals()['X_tst%s' % i],globals()['y_tr%s' % i],globals()['y_tst%s' % i] = train_test_split(globals()['train_df1%s' % i].drop(columns = ['risk_flag']),globals()['train_df1%s' % i]['risk_flag'],stratify = globals()['train_df1%s' % i]['risk_flag'])



test_df = data.loc[data['risk_flag'] == -1]


score = {}

from catboost import CatBoostClassifier
for i in range(1):
    
    globals()['catboost%s' % i]=CatBoostClassifier(eval_metric='AUC')
    globals()['catboost%s' % i].fit(globals()['X_tr%s' % i], globals()['y_tr%s' % i],cat_features=cat_cols,eval_set=(globals()['X_tst%s' % i], globals()['y_tst%s' % i]))
    score['catboost{}'.format(i)] = roc_auc_score(globals()['y_tst%s' % i],globals()['catboost%s' % i].predict(globals()['X_tst%s' % i]))


sample['risk_flag'] = catboost0.predict(test_df.drop(columns = ['risk_flag']))
sample.to_csv('catboost.csv',index = False)


for i in range(7):
    globals()['X_tr%s' % i],globals()['X_tst%s' % i],globals()['y_tr%s' % i],globals()['y_tst%s' % i] = train_test_split(globals()['train_df1%s' % i].drop(columns = ['risk_flag']),globals()['train_df1%s' % i]['risk_flag'],stratify = globals()['train_df1%s' % i]['risk_flag'])




weighted_clf = RandomForestClassifier(max_depth=3 ,random_state=0,class_weight={0:0.08,1:1}).fit(X_tr, y_tr)
roc_auc_score(y_tst,weighted_clf.predict(X_tst))



sample['risk_flag_weighted_rfc'] = weighted_clf.predict(test_df.drop(columns = ['risk_flag']))
sample['risk_flag_proba_weighted_rfc'] = weighted_clf.predict_proba(test_df.drop(columns = ['risk_flag']))[:,1]
weighted_clf.classes_

sample.to_csv('weighted_rfc.csv',index = False)




#balancedrfc

from imblearn.ensemble import BalancedRandomForestClassifier
brfc = BalancedRandomForestClassifier(n_estimators=500,random_state=0).fit(X_tr,y_tr)
roc_auc_score(y_tst,brfc.predict(X_tst))


sample['risk_flag'] = brfc.predict(test_df.drop(columns = ['risk_flag']))
sample['risk_flag_proba'] = brfc.predict_proba(test_df.drop(columns = ['risk_flag']))[:,1]
weighted_clf.classes_




print("F1 Score for Balanced Random Forest Classifier is ", f1_score(y_test,brfc.predict(X_test)))
print("Accuracy  Score for Balanced Random Fo
      
      
      

submission = pd.read_csv('catboost_+_feats_+15000itrs.csv')
sample['risk_flag_cat'] = submission['risk_flag']
sample['risk_flag_proba_cat'] = submission['risk_flag_proba']

sample['risk_flag'].value_counts()
