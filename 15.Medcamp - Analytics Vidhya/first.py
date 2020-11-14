# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 02:30:38 2020

@author: dineshy86
"""


import pandas as pd

Data_Dictionary = pd.read_excel('Data_Dictionary.xlsx')
First_Health_Camp_Attended  =pd.read_csv('First_Health_Camp_Attended.csv')
Health_Camp_Detail = pd.read_csv('Health_Camp_Detail.csv')
Patient_Profile = pd.read_csv('Patient_Profile.csv')
sample_submmission = pd.read_csv('sample_submmission.csv')
Second_Health_Camp_Attended = pd.read_csv('Second_Health_Camp_Attended.csv')
Third_Health_Camp_Attended = pd.read_csv('Third_Health_Camp_Attended.csv')
train = pd.read_csv('Train.csv')
test = pd.read_csv('test_l0Auv8Q.csv')

train['Registration_Date'].replace({'nan','15-Mar-05'},inplace = True)
train['Registration_Date'].fillna('15-Mar-05',inplace = True)
test['Outcome'] = -1


train = pd.merge(train, First_Health_Camp_Attended.drop('Unnamed: 4', axis=1), how='left', on=['Patient_ID', 'Health_Camp_ID'], indicator='camp1_merge_ind')
train = pd.merge(train, Second_Health_Camp_Attended, how='left', on=['Patient_ID', 'Health_Camp_ID'], indicator='camp2_merge_ind')
train = pd.merge(train, Third_Health_Camp_Attended, how='left', on=['Patient_ID', 'Health_Camp_ID'], indicator='camp3_merge_ind')
train = pd.merge(train, Health_Camp_Detail, how='left', on='Health_Camp_ID', indicator='healthcamp_merge_ind')
train = pd.merge(train, Patient_Profile, how='left', on='Patient_ID', indicator='patient_merge_ind')

train['Outcome'] = 0
train.loc[(train['camp1_merge_ind']=='both') | 
                     (train['camp2_merge_ind']=='both') |
                     ((train['camp3_merge_ind']=='both') & (train['Number_of_stall_visited']>0))
                     ,'Outcome'] = 1


train.drop(columns = ['Donation', 'Health_Score', 'camp1_merge_ind',
       'Health Score', 'camp2_merge_ind', 'Number_of_stall_visited',
       'Last_Stall_Visited_Number', 'camp3_merge_ind', 'Camp_Start_Date',
       'Camp_End_Date', 'Category1', 'Category2', 'Category3',
       'healthcamp_merge_ind', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age',
       'First_Interaction', 'City_Type', 'Employer_Category',
       'patient_merge_ind'],inplace = True)








#number of events attended

tot_pat_attend = pd.concat([First_Health_Camp_Attended['Patient_ID'],Second_Health_Camp_Attended['Patient_ID']])
tot_pat_attend = pd.concat([tot_pat_attend,Third_Health_Camp_Attended['Patient_ID']])
tot_pat_attend = tot_pat_attend.value_counts()
tot_pat_attend.reset_index()

First_Health_Camp_Attended['Outcome'] = 1
Second_Health_Camp_Attended['Outcome'] = 1
Third_Health_Camp_Attended['Outcome'] = 1 

attended = pd.concat([First_Health_Camp_Attended[['Outcome','Patient_ID', 'Health_Camp_ID']],Second_Health_Camp_Attended[['Outcome','Patient_ID', 'Health_Camp_ID']]])
attended = pd.concat([attended,Third_Health_Camp_Attended[['Outcome','Patient_ID', 'Health_Camp_ID']]])


train = pd.merge(train,attended,on = ['Patient_ID', 'Health_Camp_ID'],how='left').fillna(0)

'------------------------------------- PREPROCESSING --------------------------------------------------'

def make_date(column):
    date = pd.DataFrame()
    mnt_dict = {'Aug':8,'Nov':11, 'Dec':12,'Jan':1,'Feb':2,'Apr':4,'May':5,'Sep':9,'Oct':10,'Jun':6,'Jul':7,'Mar':3}
    date['day'] = column.apply(lambda x : x.split('-')[0])
    date['month'] = column.apply(lambda x : x.split('-')[1]).replace(mnt_dict)
    date['year'] = column.apply(lambda x : int(x.split('-')[2]))+2000
    
    return pd.to_datetime(date[['year','month','day']])






#patience profile
from sklearn.preprocessing import LabelEncoder

Patient_Profile.fillna('None',inplace = True)

emp_enc = LabelEncoder()
city_enc = LabelEncoder()

Patient_Profile['Employer_Category'] = emp_enc.fit_transform(Patient_Profile['Employer_Category'])
Patient_Profile['City_Type'] = city_enc.fit_transform(Patient_Profile['City_Type'])

Patient_Profile.replace({'None':-1},inplace = True)

Patient_Profile['First_Interaction'] = make_date(Patient_Profile['First_Interaction'])
for i in ['Income','Education_Score','Age']:
    Patient_Profile[i] = pd.to_numeric(Patient_Profile[i])


#camp_detail

cat1 = LabelEncoder()
cat2 = LabelEncoder()

Health_Camp_Detail['Category1'] = cat1.fit_transform(Health_Camp_Detail['Category1'])
Health_Camp_Detail['Category2'] = cat2.fit_transform(Health_Camp_Detail['Category2'])
list(Health_Camp_Detail['Camp_Start_Date'].apply(lambda x : x.split('-')[1]).unique())

Health_Camp_Detail['Camp_End_Date'] = make_date(Health_Camp_Detail['Camp_End_Date'])
Health_Camp_Detail['Camp_Start_Date'] = make_date(Health_Camp_Detail['Camp_Start_Date'])

import datetime as dt
Health_Camp_Detail['period'] = Health_Camp_Detail['Camp_End_Date'].apply(lambda x: dt.datetime.toordinal(x))-Health_Camp_Detail['Camp_Start_Date'].apply(lambda x: dt.datetime.toordinal(x))


' ------------------merging camp and pat into train--------------------------------'


data = pd.concat([train,test])
data = pd.merge(data,Health_Camp_Detail,on = ['Health_Camp_ID'])
data = pd.merge(data,Patient_Profile,on = ['Patient_ID'])


data['Registration_Date'] = make_date(data['Registration_Date'])


data['time_to_attend'] = data['Camp_Start_Date'].apply(lambda x: dt.datetime.toordinal(x))-data['Registration_Date'].apply(lambda x: dt.datetime.toordinal(x))



'--------------------variable model ------------------------------------------------------------'
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train[['Var1', 'Var2','Var3', 'Var4', 'Var5']],copy['attended'])

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000,
                             n_jobs = -1,
                             verbose = 1
                             )
rfc.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,rfc.predict(X_test))

sub = sample_submmission.copy()

sub['Outcome'] = rfc.predict(test[['Var1', 'Var2','Var3', 'Var4', 'Var5']])
sub.to_csv('rfr_with_variables.csv',index = False)




'----------------model --------------------------------------------------------'
drop_columns = ['Patient_ID', 'Health_Camp_ID','Registration_Date', 'Camp_Start_Date','First_Interaction', 'Camp_End_Date',]
data.drop(columns = drop_columns,inplace = True)


train2 = data[data['Outcome']>-1]
test2 = data[data['Outcome'] == -1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train2.drop(columns = ['Outcome']),train2['Outcome'])

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000,
                             n_jobs = -1,
                             verbose = 1
                             )
rfc.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test,rfc.predict(X_test))

sub2 = sample_submmission.copy()

sub2['Outcome'] = rfc.predict(test2.drop(columns = ['Outcome']))
sub2.to_csv('rfr_with_features.csv',index = False)
roc_auc_score(y_test,rfc.predict(X_test))


'---------------model 3 ------------------------------'

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=1000)

gbc.fit(X_train,y_train)


sub3 = sample_submmission.copy()

sub3['Outcome'] = gbc.predict(test2.drop(columns = ['Outcome']))
sub3.to_csv('gbc.csv',index = False)


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,gbc.predict(X_test))


from xgboost import XGBClassifier
gbc = XGBClassifier(n_estimators=1000)

gbc.fit(X_train,y_train)


sub3 = sample_submmission.copy()

sub3['Outcome'] = gbc.predict(test2.drop(columns = ['Outcome']))
sub3.to_csv('gbc.csv',index = False)


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,gbc.predict(X_test))
