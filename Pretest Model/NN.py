import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.preprocessing import RobustScaler

print('Tensorflow Version ' + tf.__version__)

###############################################################################
# LOADING DATA
###############################################################################
url = 'data.csv'
column_names = ['datetime','Heater', 'Window', 'Fan', 'Temp', 'Velocity','CO2']

raw_dataset = pd.read_csv(url, names=column_names, parse_dates=['datetime'], index_col="datetime",
                          na_values='?', comment='\t',
                          sep=';', skipinitialspace=True)
print(raw_dataset.shape)


dataset = raw_dataset.copy()
print(dataset.tail())
print('')
print('missing variables (dropped):')
print(dataset.isna().sum())
dataset = dataset.dropna()

# when categorical needes to be splitted
# dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
# dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
# dataset.tail()

###############################################################################
# VISUALIZE DATA
###############################################################################

plot_cols = ['Heater', 'Window', 'Fan', 'Temp', 'Velocity','CO2']
plot_features = dataset[plot_cols]
plot_features.index = dataset.index
_ = plot_features.plot(subplots=True)
plt.show()


###############################################################################
# DATASET SEPERATION
###############################################################################
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size

train_dataset = dataset.iloc[0:train_size]
test_dataset = dataset.iloc[train_size:len(dataset)]

print('')
print('Training Dataset')
print(len(train_dataset))
#print(train_dataset.tail())
print('')
print('Test Dataset')
print(len(test_dataset))
#print(test_dataset.tail())

#sns.pairplot(train_dataset[['Temp', 'Velocity', 'CO2']], diag_kind='kde')
#plt.show()

###############################################################################
# FEATURE SCALING
###############################################################################
#f_columns = ['Temp', 'Velocity','CO2']
#f_transformer = RobustScaler()
#f_transformer = f_transformer.fit(train_dataset[f_columns].to_numpy())

#train_dataset.loc[:, f_columns] = f_transformer.transform(train_dataset[f_columns].to_numpy())

#test_dataset.loc[:, f_columns] = f_transformer.transform(test_dataset[f_columns].to_numpy())

print('')
print('scaling features...')
print('Training Dataset')
print(train_dataset.tail())
print('')
print('Test Dataset')
print(test_dataset.tail())

###############################################################################
# CREATE FITTING DATASET
###############################################################################

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 1
# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train_dataset, train_dataset.Temp, time_steps)
X_test, y_test = create_dataset(test_dataset, test_dataset.Temp, time_steps)

print(X_train.shape, y_train.shape)
print('input shape: ' + str(X_train.shape[1]) + 'x' + str(X_train.shape[2]))

###############################################################################
# MODEL BUILDING
###############################################################################

#model = tf.keras.Sequential([
#  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(6,)),  # input shape required
#  tf.keras.layers.Dense(10, activation=tf.nn.relu),
#  tf.keras.layers.Dense(6)
#])


model = tf.keras.Sequential()
model.add(
  tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
      units=128,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)

#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32,input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


history = lstm_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)



plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();
plt.show()



lstm_model.evaluate(X_test, y_test)

y_pred = lstm_model.predict(X_test)



plt.plot(np.arange(0, len(y_train)), y_train, 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred, 'r', label="prediction")
plt.ylabel('Value')
plt.xlabel('Time Step')
plt.legend()
plt.show();



print('end.')
