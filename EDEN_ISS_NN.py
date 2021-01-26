import os
import io
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


from sklearn.preprocessing import RobustScaler

print('Tensorflow Version ' + tf.__version__)


###############################################################################
# SELECTING MODEL TARGET
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('Select the model target:')
print('1) System Variables: Time Controlled')
print('2) System Variables: Environment Controlled')
print('3) Plant')
print('4) identify time controlled variables')

num_1 = int(input("Select option: "))
options = {1 : 'System Variables/Time Controlled',
           2 : 'System Variables/Environment Controlled',
           3 : 'Plant',
           4 : 'identify'
}
model_target = options[num_1]


###############################################################################
# LOADING DATA
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('Select dataset to load:')
print('1) dataset_full.csv')
print('2) dataset_time_controlled.csv')
print('3) dataset_environment_controlled.csv')
print('4) pretest.csv')

num = int(input("Select option: "))
options = {1 : 'dataset_full.csv',
           2 : 'dataset_time_controlled.csv',
           3 : 'dataset_environment_controlled.csv',
           4 : 'pretest.csv',
}
URL = 'datasets/' + options[num]

print('READING ' + URL, end=' ')
df = pd.read_csv(URL, parse_dates=['Date_Time'], index_col="Date_Time",
                          na_values='NaN', comment='\t',
                          sep=';', skipinitialspace=True)
print(df.shape)

if options[num] == 'dataset_full.csv':
    time_controlled_features = ['L1-2L BLUE','L1-2R BLUE','L1-4L BLUE','L1-4R BLUE','R4-4R BLUE','R4-4L BLUE','L2-1L BLUE','L2-1R BLUE','L2-2L BLUE','L2-2R BLUE','L2-3L BLUE','L2-3R BLUE','L2-4L BLUE','L2-4R BLUE','L3-1L BLUE','L3-2L BLUE','L3-1R BLUE','L3-2R BLUE','L3-3L BLUE','L3-3R BLUE','L3-4L BLUE','L3-4R BLUE','L4-1L BLUE','L4-1R BLUE','L4-2L BLUE','L4-2R BLUE','L4-3L BLUE','L4-3R BLUE','L4-4L BLUE','L4-4R BLUE','R1-2R BLUE','R1-2L BLUE','R1-4R BLUE','R1-4L BLUE','R2-2R BLUE','R2-2L BLUE','R2-4R BLUE','R2-4L BLUE','R3-2/4R BLUE','R3-2/4L BLUE','R4-2R BLUE','R4-2L BLUE','L1-2L RED','L1-2R RED','L1-4L RED','L1-4R RED','R4-4R RED','R4-4L RED','L2-1L RED','L2-1R RED','L2-2L RED','L2-2R RED','L2-3L RED','L2-3R RED','L2-4L RED','L2-4R RED','L3-1L RED','L3-2L RED','L3-1R RED','L3-2R RED','L3-3L RED','L3-3R RED','L3-4L RED','L3-4R RED','L4-1L RED','L4-1R RED','L4-2L RED','L4-2R RED','L4-3L RED','L4-3R RED','L4-4L RED','L4-4R RED','R1-2R RED','R1-2L RED','R1-4R RED','R1-4L RED','R2-2R RED','R2-2L RED','R2-4R RED','R2-4L RED','R3-2/4R RED','R3-2/4L RED','R4-2R RED','R4-2L RED','L1-2L FAR RED','L1-2R FAR RED','L1-4L FAR RED']

    test_features = ['FEG AIR FLOW','NDS-BASE DOSING PUMP ','NDS-ACID SOLENOID','PDS TEMP CONTROL BOX','SUBFLOOR CPO 2','AMS-FEG-FAN AIR LOOP 1','AMS-FEG-FAN AIR LOOP 2','SES TEMP AIR IN 1','FEG HUMIDITY TARGET','NDS-ACID DOSING PUMP ','NDS-PUMP FW ','FLOW METER TANK 2','SES TEMP AIR IN 2','NDS-SOLENOID FW TANK 1','NDS-REC PUMP TANK 1','NDS-REC PUMP TANK 2','NDS-SOLENOID FW TANK 2','NDS-A DOSING PUMP','NDS-B DOSING PUMP','SES DOOR STATUS']
    df = df[test_features]

df.astype('float64')
print(df.tail())
print('')
print('missing variables:', end=' ')
print(df.isna().sum().sum())

# short to one feature
if options[num] == 'dataset_time_controlled.csv':
    df = df.iloc[:, 0:1]



###############################################################################
# VISUALIZE DATA
###############################################################################

# plot first 6 columns
#plot_features = df.iloc[:, 0:6]
# plot all
plot_features = df
print(plot_features)
plot_features.index = df.index
_ = plot_features.plot(subplots=True)
plt.show()


###############################################################################
# DATASET SEPERATION
###############################################################################

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# usually input features = output features
num_input_features = df.shape[1]
num_output_features = df.shape[1]

#if options[num] == 'dataset_full.csv':
#    num_input_features = df.shape[1]
#    num_output_features = df.shape[1] - len(time_controlled_features)

print()
print('FEATURES')
print('input: ', end=' ')
print(num_input_features)
print('output: ', end=' ')
print(num_output_features)


###############################################################################
# FEATURE SCALING
###############################################################################
print('')
print('scale features?  [y]/ n', end=' ', flush=True)

scale_user = input()
if ( scale_user == 'y' ) or ( scale_user == '' ) :
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    print('-> features scaled.')
else:
    print('-> features not scaled.')




###############################################################################
# WINDOW CLASS
###############################################################################
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])



###############################################################################
# SPLIT WINDOW
###############################################################################

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

###############################################################################
#  TRAIN | VAL | TEST   PLOT
###############################################################################

print('visualizing first feature: ', end=' ')
print(df.columns)
print('')

def plot(self, model=None, plot_col=df.columns, max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time Steps')

WindowGenerator.plot = plot


###############################################################################
# MAKE DATASET
###############################################################################

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

###############################################################################
# USE
###############################################################################

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example




###############################################################################
# MULTI STEP MODELS
###############################################################################


MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history



OUT_STEPS = 288
multi_window = WindowGenerator(input_width=288,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

#multi_window.plot()
#
#plt.show()
#plt.pause(0.001)


print('┌────────────────┐')
print('│ BASELINE MODEL │')
print('└────────────────┘')
class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
#multi_window.plot(last_baseline)
#plt.show()


class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

def compute_repeat():
    print('┌────────────────┐')
    print('│  REPEAT MODEL  │')
    print('└────────────────┘')

    global repeat_baseline

    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                            metrics=[tf.metrics.MeanAbsoluteError()])

    multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
    multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
    multi_window.plot(repeat_baseline)

    plt.show()
    plt.pause(0.001)


def compute_linear():
    print('┌────────────────┐')
    print('│  LINEAR MODEL  │')
    print('└────────────────┘')

    global multi_linear_model

    multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_input_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    history = compile_and_fit(multi_linear_model, multi_window)

    IPython.display.clear_output()
    multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
    multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_linear_model)

    plt.show()
    plt.pause(0.001)

def compute_dense():
    print('┌────────────────┐')
    print('│  DENSE MODEL   │')
    print('└────────────────┘')

    global multi_dense_model

    multi_dense_model = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation='relu'),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_input_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    history = compile_and_fit(multi_dense_model, multi_window)

    IPython.display.clear_output()
    multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
    multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_dense_model)

    plt.show()
    plt.pause(0.001)

def compute_conv():
    print('┌────────────────┐')
    print('│ CONVOLUTIONAL  │')
    print('└────────────────┘')

    global multi_conv_model

    CONV_WIDTH = 3
    multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_input_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    history = compile_and_fit(multi_conv_model, multi_window)

    IPython.display.clear_output()

    multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
    multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_conv_model)

    plt.show()
    plt.pause(0.001)

def compute_lstm():
    print('┌────────────────┐')
    print('│   LSTM MODEL   │')
    print('└────────────────┘')

    global multi_lstm_model

    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_input_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    history = compile_and_fit(multi_lstm_model, multi_window)

    IPython.display.clear_output()

    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model)

    plt.show()
    plt.pause(0.001)


# for Auto LSTM
class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_input_features)

# for Auto LSTM
def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

# for Auto LSTM
def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the lstm state
  prediction, state = self.warmup(inputs)

  # Insert the first prediction
  predictions.append(prediction)

  # Run the rest of the prediction steps
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions



def compute_auto_lstm():
    print('┌────────────────┐')
    print('│   AUTO LSTM    │')
    print('└────────────────┘')

    feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
    FeedBack.warmup = warmup

    prediction, state = feedback_model.warmup(multi_window.example[0])
    prediction.shape

    FeedBack.call = call

    print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

    history = compile_and_fit(feedback_model, multi_window)

    IPython.display.clear_output()

    multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
    multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(feedback_model)

    plt.show()
    plt.pause(0.001)



def compute_all():
    compute_repeat()
    compute_linear()
    compute_dense()
    compute_conv()
    compute_lstm()
    #compute_auto_lstm()



print('Select Model to compute:')
print('0) Repeat')
print('1) Linear')
print('2) Dense')
print('3) Convolutional')
print('4) LSTM')
print('5) Auto LSTM')
print('6) all')
num = int(input("Select option: "))
options = {0 : compute_repeat,
           1 : compute_linear,
           2 : compute_dense,
           3 : compute_conv,
           4 : compute_lstm,
           5 : compute_auto_lstm,
           6 : compute_all,

}
options[num]()

###############################################################################
# PERFORMANCE
###############################################################################

x = np.arange(len(multi_performance))
width = 0.3


metric_name = 'mean_absolute_error'

metric_index = last_baseline.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show()

print('')
print('┌──── summary ─────┐')
for name, value in multi_performance.items():
  print('│ ' + f'{name:8s}: {value[1]:0.4f}' + ' │')
print('└──────────────────┘')

with io.open('./models/' + model_target + '/summary.txt', "a", encoding="utf-8") as f:
    f.write("┌──── summary ─────┐\n")
    for name, value in multi_performance.items():
      f.write('│ ' + f'{name:8s}: {value[1]:0.4f}' + ' │\n')
    f.write('└──────────────────┘\n')
    f.write('\n')
    f.close()

###############################################################################
# SAVING MODELS
###############################################################################
print('')
print('saving at ' + './models/' + model_target)

#multi_lstm_model.save('./models/' + model_target + '/REPEAT.h5', save_format="tf")

try:
    multi_linear_model.save_weights('models/' + model_target + '\LINEAR.h5')
    print('LINEAR.h5 saved')
except: print('', end='')
try:
    multi_dense_model.save_weights('models/' + model_target + '\DENSE.h5')
    print('DENSE.h5 saved')
except: print('', end='')
try:
    multi_conv_model.save_weights('models/' + model_target + '\CONV.h5')
    print('CONV.h5 saved')
except: print('', end='')
try:
    multi_lstm_model.save_weights('models/' + model_target + '\LSTM.h5')
    print('LSTM.h5 saved')
except: print('', end='')
try:
    feedback_model.save_weights('models/' + model_target + '\AR_LSTM.h5')
    print('AR_LSTM.h5 saved')
except: print('', end='')


###############################################################################
# LOADING MODELS
###############################################################################

#res50_model = load_model('my_model.h5')
#res50_model.summary()
#res50_model.get_weights()
print('end.')
end = input()
