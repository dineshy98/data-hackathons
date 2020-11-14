# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/train.csv')
test= pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/test.csv')
sample= pd.read_csv('S:/Codes/DataHackthon/17.Health Care Analytics - Analytics Vidhya/sample.csv')
test['Crop_Damage'] = -1

train.isnull().sum()

train['Number_Weeks_Used'].fillna(train['Number_Weeks_Used'].mean(),inplace = True)
test['Number_Weeks_Used'].fillna(test['Number_Weeks_Used'].mean(),inplace = True)



'--------------------EDA ---------------------------------------------------------------'

cross1 = pd.crosstab(index = train['Estimated_Insects_Count'],columns = train['Crop_Damage']).reset_index()
sns.pairplot(train,hue = 'Crop_Damage')

sns.scatterplot(data = train,x = 'Estimated_Insects_Count',y = 'Number_Weeks_Used',hue = 'Crop_Damage')

sns.relplot(x="Estimated_Insects_Count",
y="Number_Weeks_Used",col = 'Crop_Damage',hue = 'Crop_Type',
data=train,
kind="scatter")

train.corr()

'-------------------Features generation --------------------------------------------------'

data = pd.concat([train,test])
data['Crop_Soil'] = data['Crop_Type']*data['Soil_Type']

data['Enviroment_conditions'] = (data['Soil_Type'].replace({0:4})) * data['Season']

data['pest_crop'] = (data['Crop_Type'].replace({0:4}))* data['Pesticide_Use_Category']

data['Total_Doses'] = data['Number_Weeks_Used']*data['Number_Doses_Week']

train_mod = data[data['Crop_Damage'] > -1]

test_mod = data[data['Crop_Damage'] < 0]







'-----------------------modelling 2 --------------------------------------------------------'

from imblearn.over_sampling import SMOTE
sm =SMOTE(random_state=42)
X_res_OS , Y_res_OS = sm.fit_resample(train.drop(columns = ['ID','Crop_Damage']),train['Crop_Damage'])
pd.Series(Y_res_OS).value_counts()





from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000)
rfc.fit(X_res_OS,Y_res_OS)






























'----------------------------------modelling --------------------------------------------------'

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train.drop(columns = ['ID','Crop_Damage']),
                                                     train['Crop_Damage'])



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000)
rfc.fit(X_train,y_train)
cv_results = pd.DataFrame()

from sklearn.model_selection import cross_validate
cv_rfc = cross_validate(rfc, X_train, y_train, cv=5, scoring='accuracy',verbose = 5)

cv_results['cv_rfc'] = cv_rfc['test_score']



from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn import metrics



def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))





lg = lgb.LGBMClassifier(silent=False,objective = 'multiclass')
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, verbose=5)
grid_search.fit(X_train,y_train)
grid_search.best_estimator_

d_train = lgb.Dataset(X_train, label=y_train)
params = {"max_depth": 25, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300,
          "num_class" : 3, metric : "multi_logloss"}

# Without Categorical Features
model2 = lgb.train(params, d_train)
auc2(model2, train, test.drop(columns = ['ID']))


lg = lgb.LGBMClassifier()

params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=10
params['num_class']=3


#With Catgeorical Features
cate_features_name = ['Crop_Type', 'Soil_Type',
       'Pesticide_Use_Category','Season']

d_train=lgb.Dataset(X_train, label=y_train)

model2 = lgb.train(params,d_train, categorical_feature = cate_features_name)
auc2(model2, train, test)


y_pred_1=model2.predict(X_test)
y_pred_1 = [np.argmax(line) for line in y_pred_1]




,objective = 'multiclass'



temp = model2.predict(test.drop(columns = ['ID']))
temp = [np.argmax(line) for line in temp]

submission = sample.copy()
submission['Crop_Damage'] = temp
submission.to_csv('{}.csv'.format('light_gbm'),index = False)




def submit(estimator,submission_name):
    submission = sample.copy()
    submission['Crop_Damage'] = estimator.predict(test.drop(columns = ['ID']))
    submission.to_csv('{}.csv'.format(submission_name),index = False)

'-------------------------submission ---------------------------------------------------------'


def submit(estimator,submission_name):
    submission = sample.copy()
    submission['Crop_Damage'] = estimator.predict(test.drop(columns = ['ID','Crop_Damage']))
    submission.to_csv('{}.csv'.format(submission_name),index = False)
    
    return submission as 'results_{}'.format(estimator)



submit(rfc,'over_rfr')
