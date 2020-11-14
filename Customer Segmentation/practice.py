# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:50:41 2020

@author: dineshy86
"""
import pandas as pd


spa_train = pd.read_csv('C:/Users/dines/Downloads/Compressed/SpaData.csv')


spa_test = spa_train.sample(800)




# fitting knn algo

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(spa_train.drop(columns = ['GTOccupancy','TimeStamp']),spa_train['GTOccupancy'])

Rejuvenate = pd.DataFrame()
Rejuvenate['ID'] = spa_test['ID']
Rejuvenate['Occupancy'] = knn.predict(spa_test.drop(columns = ['GTOccupancy','TimeStamp']))


spa_train.TimeStamp.week



from sklearn.ensemble import RandomForest

finalOutput.columns = ['id', 'taste']
finalOutput.to_csv("/code/wine_prediction.csv", index = False)


































import pandas as pd


wine_train = pd.read_csv('C:/Users/dines/Downloads/Compressed/wine_train.csv')

wine_test = wine_train.sample(700)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

rfr.fit(wine_train.drop(columns = ['quality']),wine_train['quality'])

df = pd.DataFrame()

df['quality'] = rfr.predict(wine_test.drop(columns = ['quality']))


k = ['bad' if x < 7 else x  for x in df['quality']]

result = []
for i in list(df['quality']):
    if (i < 7):
        result.append('bad')
    if (i == 7):
        result.append('normal')   
    if (i > 7):
        result.append('good')
        



k = ['normal' if x == 7 else x  for x in k]
k = ['good' if x > 7 else x  for x in k]
    



import pandas as pd

xtrain = pd.read_csv("/data/training/wine_train.csv", header = 0)
xtest = pd.read_csv("/data/test/wine_test.csv", header = 0)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

rfr.fit(xtrain.drop(columns = ['quality']),xtrain['quality'])

df = pd.DataFrame()
df['quality'] = rfr.predict(xtest.drop(columns = ['quality']))


result = []
for i in list(df['quality']):
    if (i < 7):
        result.append('bad')
    if (i == 7):
        result.append('normal')   
    if (i > 7):
        result.append('good')

wine_prediction = pd.DataFrame()
wine_prediction['taste'] = result


print(wine_prediction)




