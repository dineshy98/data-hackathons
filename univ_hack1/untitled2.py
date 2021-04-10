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
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = drop_cols,inplace = True)


import category_encoders as ce
encd = ce.cat_boost.CatBoostEncoder(cols=cat_cols,return_df=True)
data1 = encd.fit_transform(data,data['risk_flag'])

train_df = data1.loc[data1['risk_flag'] != -1]
test_df = data1.loc[data1['risk_flag'] == -1]t



X_tr,X_tst,y_tr,y_tst = train_test_split(train_df.drop(columns = ['risk_flag']),train_df['risk_flag'],stratify = train_df['risk_flag'])

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
      
      
#catboost



submission = pd.read_csv('catboost_+_feats_+15000itrs.csv')
sample['risk_flag_cat'] = submission['risk_flag']
sample['risk_flag_proba_cat'] = submission['risk_flag_proba']

sample['risk_flag'].value_counts()



#rfc_dropped
train_df1 = train_df.copy()
train_df1.drop(train_df1.loc[train_df1['risk_flag'] == 0].sample(190000).index,inplace = True)

X_tr1,X_tst1,y_tr1,y_tst1 = train_test_split(train_df1.drop(columns = ['risk_flag']),train_df1['risk_flag'],stratify = train_df1['risk_flag'])

weighted_clf_bal = RandomForestClassifier(max_depth=3 ,random_state=0).fit(X_tr1, y_tr1)
roc_auc_score(y_tst1,weighted_clf_bal.predict(X_tst1))


sample['risk_flag_rfc_baldata'] = weighted_clf_bal.predict(test_df.drop(columns = ['risk_flag']))
sample['risk_flag_proba_rfc_baldata'] = weighted_clf_bal.predict_proba(test_df.drop(columns = ['risk_flag']))[:,1]



