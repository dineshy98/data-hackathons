# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:34:20 2020

@author: dineshy86
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample.csv')


train.describe()
train.isnull().sum()
train.info()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from google.colab import drive
drive.mount('/content/drive/')

train = pd.read_csv('/content/drive/My Drive/Datasets/#16 Playstore App Downloads Prediction/Train.csv')
test= pd.read_csv('/content/drive/My Drive/Datasets/#16 Playstore App Downloads Prediction/Test.csv')
sample= pd.read_csv('/content/drive/My Drive/Datasets/#16 Playstore App Downloads Prediction/Sample_Submission.csv')


from catboost import CatBoostRegressor

    




























