# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:58:35 2020

@author: dineshy86
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:39:58 2020

@author: dineshy86
"""
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('Train.csv')
test= pd.read_csv('Test.csv')
sample= pd.read_csv('sample_submission.csv')
test['shares'] = -1

data = pd.concat([train,test])




day_cat = ['weekday_is_monday', 'weekday_is_tuesday',
       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
       'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend']



ohe_data = data[day_cat]
array1 = np.array(ohe_data)
ohe = np.argmax(array1, axis=1)+1
data['day_type'] = ohe

drop_columns = list(set(day_cat+corr_columns))
drop_columns = set(day_cat.extend(corr_columns))
drop_columns.remove('shares')
data.drop(columns = drop_columns,inplace = True)

train_mod = data[data['shares'] > -1]
test_mod = data[data['shares'] == -1]
test_mod.drop(columns = ['shares'],inplace = True)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_mod.drop(columns = ['shares']),train_mod['shares'])


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 1000,
                             n_jobs = -1,
                             verbose = 1
                             )
rfr.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,rfr.predict(X_test))

sub = sample.copy()

sub['shares'] = lgbm.predict(test.drop(columns = ['shares']))
sub.to_csv('lgbm.csv',index = False)




'---------------------lightgbm ---------------------------------------------------------------------'
X = train.drop(columns = ['shares'])
y = train['shares']

from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(verbose=1)
lgbm.fit(X,y)


