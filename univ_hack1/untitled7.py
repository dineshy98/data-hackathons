# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 21:40:55 2021

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
