import matplotlib.pyplot as plt         # plots
import numpy as np                      # math operations
import pandas as pd                     # dataset handling
import seaborn as sns                   # detailed plotting
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf                 # machiene learning
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.preprocessing import RobustScaler

print('Tensorflow Version ' + tf.__version__)

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
# FEATURE SCALING
###############################################################################



###############################################################################
# CREATE FITTING DATASET
###############################################################################


###############################################################################
# MODEL BUILDING
###############################################################################




print('end.')
