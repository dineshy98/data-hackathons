# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:33:16 2020

@author: dineshy86
"""

import pandas as pd
data = pd.read_csv('data.csv')

corelation = data.corr()[-3:]


