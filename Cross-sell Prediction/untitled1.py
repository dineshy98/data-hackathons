# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:39:37 2020

@author: dineshy86
"""
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



  
estimators = [
    ('knn1', KNeighborsClassifier(n_neighbors = 5)),
    ('knn2', KNeighborsClassifier(n_neighbors = 10)),
    ('xgbc', XGBClassifier(gamma = 1,subsample = 0.6,colsample_bytree = 1,min_child_weight = 5,max_depth = 3,n_estimators=25)),
    ('lgbc', LGBMClassifier()),
    ('dtr1', DecisionTreeClassifier(splitter = 'best')),
    ('dtr2', DecisionTreeClassifier(splitter = 'random'))

]

classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=750),
    passthrough = True,
    verbose = 50,
    n_jobs = -1
)

classifier.fit(X_tr, y_tr)

accuracy_score(y_tst,classifier.predict(X_tst))


def submission(estimator,filename):
    sub = sample.copy()
    sub['Response'] = estimator.predict(test_mod.drop(columns = ['Response']))
    sub.to_csv('{}.csv'.format(filename),index = False)

submission(classifier,'stack')
