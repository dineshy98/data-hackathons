# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:45:09 2020

@author: dineshy86
"""


params_xg = {
        'n_estimators': [100,750,1500],
        'max_depth':[5,9,15],
        'min_child_weight': [1, 20],
        'gamma': [0.5,2, 5],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.7, 1.0]
        }


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,log_loss,accuracy_score
score_card = pd.Series()

def best_grid_model(estimator,X_tr,X_tst,y_tr,y_tst,parmas_grid):
    
    grid = GridSearchCV(estimator=estimator,
                    param_grid=parmas_grid,
                    cv=5,  
                    verbose=10, 
                    n_jobs=-1,
                    scoring='neg_log_loss',
                    refit = True)
    
    grid.fit(X_tr,y_tr)
    print('Best Parmaters are :',grid.best_params_)
    print('ReFitting Best Parameters to New Estimator ...............','\n')
    print('Best Score is :',grid.best_score_)
    best_estimator = estimator
    best_estimator.set_params(**grid.best_params_)
    best_estimator.fit(X_tr,y_tr)
    
    
    best_accuracy_score = accuracy_score(y_tst,best_estimator.predict(X_tst))
    print('Accuracy for X_test is :',best_accuracy_score,'\n' )
    
    score_card['best_{}'.format(estimator)] = best_accuracy_score
    
    return grid,best_estimator

from xgboost import XGBClassifier
grid_xgb,best_xgb = best_grid_model(XGBClassifier(objective = 'binary:logistic'),X_tr,X_tst,y_tr,y_tst,params_xg)

submission(best_xgb,'hyper_xgb')
