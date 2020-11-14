# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:05:58 2020

@author: dineshy86
"""
import pandas as pd
import numpy as np

ranks = pd.read_excel('S:/Codes/FileHandler.xlsx')

for i in list(ranks.columns):
    ranks[i].replace('nan',np.nan)


p = "If the Closing/Opening Rank has a suffix 'P', it indicates that the corresponding rank is from preparatory Rank List."

rem1 = []
rem1.extend([0,1])
for i in range(len(ranks)):
    if ranks['Unnamed: 0'][i] == p:
        rem1.append(i)
        rem1.append(i+1)
        rem1.append(i+2)
rem1 = rem1[:-2]
rem1df = ranks.iloc[rem1]

ranks.drop(index = rem1,inplace = True)



joins = []
for i in range(len(ranks)):
    joins.append(str(ranks['Unnamed: 0'][i]) + str(ranks['Unnamed: 1'][i])+str(ranks['Unnamed: 2'][i]))


joins = []
for i in range(len(ranks)):
    if all(ranks['Unnamed: 0'][i].isnull(),ranks['Unnamed: 1'][i].isnull(),ranks['Unnamed: 2'][i].isnull()):
    
    elif (ranks['Unnamed: 0'][i].isnull() = False,ranks['Unnamed: 1'][i].isnull()= True,ranks['Unnamed: 2'][i].isnull()= True):
         
        return ranks['Unnamed: 0'][i]
    elif (ranks['Unnamed: 0'][i].isnull() = False,ranks['Unnamed: 1'][i].isnull()= False,
          ranks['Unnamed: 2'][i].isnull()= False): 
        return str(ranks['Unnamed: 0'][i]) + ' ' + str(ranks['Unnamed: 1'][i]) +' '+
          str(ranks['Unnamed: 2'][i])



ranks = ranks.reset_index(drop = True)

lis1 = []
for i in range(len(ranks)):
    if str(ranks['Joint Seat Allocation 2019'][i]).count('Bachelor') == 1:
       lis1.append(i)
    if str(ranks['Joint Seat Allocation 2019'][i]).count('Master') == 1:
       lis1.append(i)
        
r1 = ranks.iloc[lis1]


lis2 = []

for i in range(len(ranks)):
    if str(ranks['Unnamed: 2'][i]).count('Bachelor') == 1:
       lis2.append(i)
    elif str(ranks['Unnamed: 2'][i]).count('Master') == 1:
       lis2.append(i)
        
r2 = ranks.iloc[lis2]



lis3 = []

for i in range(len(ranks)):
    if str(ranks['Unnamed: 1'][i]).count('Bachelor') == 1:
       lis3.append(i)
    elif str(ranks['Unnamed: 1'][i]).count('Master') == 1:
       lis2.append(i)
       
       
       
r3 = ranks.iloc[lis3]


lis = lis1+lis2+lis3

tempr = ranks[~ranks.index.isin(lis)]

for i in r1.columns:
    r1[i] = r1[i].fillna('')



l = []
for i in list(r1.index):
    l.append(str(r1['Unnamed: 0'][i])+' '+str(r1['Unnamed: 1'][i])+' '+str(r1['Unnamed: 2'][i]))













If the Closing/Opening Rank has a suffix 'P', it indicates that the corresponding rank is from preparatory Rank List.
"Round No"

False | True



'(nkswl'.count('(')
        