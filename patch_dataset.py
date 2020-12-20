import numpy as np                      # math operations
import pandas as pd                     # dataset handling


###############################################################################
# LOADING DATASET
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('READING dataset.csv')

dataset = pd.read_csv('dataset.csv',index_col="Date_Time", sep=';', low_memory=False)
dataset.index = pd.to_datetime(dataset.index)
print(dataset.head())
print(dataset.tail())
print('')
print('missing variables:', end=' ')
print(dataset.isnull().sum().sum())

###############################################################################
# VISUALIZE MISSING VAR
###############################################################################



###############################################################################
# ESTIMATE MISSING DATA
###############################################################################




print('end.')
