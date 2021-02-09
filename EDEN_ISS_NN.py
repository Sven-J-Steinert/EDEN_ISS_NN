import os
import io
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mpl.rcParams['figure.figsize'] = (6, 6)
mpl.rcParams['axes.grid'] = False


from sklearn.preprocessing import RobustScaler


# SETTINGS

IN_STEPS = 288
OUT_STEPS = IN_STEPS  # 288

MAX_EPOCHS = 1



print('')
print('Tensorflow Version ' + tf.__version__ )
print('')
print('┌────────────────────────────────────────────────────────────────────────────────────────┐')
print('│                                                                                        │')
print('│    ███████ ██████  ███████ ███    ██     ██ ███████ ███████     ███    ██ ███    ██    │')
print('│    ██      ██   ██ ██      ████   ██     ██ ██      ██          ████   ██ ████   ██    │')
print('│    █████   ██   ██ █████   ██ ██  ██     ██ ███████ ███████     ██ ██  ██ ██ ██  ██    │')
print('│    ██      ██   ██ ██      ██  ██ ██     ██      ██      ██     ██  ██ ██ ██  ██ ██    │')
print('│    ███████ ██████  ███████ ██   ████     ██ ███████ ███████     ██   ████ ██   ████    │')
print('│                                                                                        │')
print('│    build on           CUDA 11.0    cuDNN 8.1.0    Tensorflow 2.4.1                     │')
print('└────────────────────────────────────────────────────────────────────────────────────────┘')
print('')
print('Select purpose:')
print('┌───────────────────────────┐')
print('│ 1) Train Network          │')
print('│ 2) Load Network           │')
print('└───────────────────────────┘')

num = int(input("Select option: "))
options = {1 : 'train',
           2 : 'load',
}
purpose = options[num]

###############################################################################
# SELECTING MODEL TARGET
###############################################################################
print('')
print('Select the model target:')
print('┌───────────────────────────┐')
print('│ 0) Identify Constants     │')
print('│ 1) Time Controlled        │')
print('│ 2) Environment Controlled │')
print('└───────────────────────────┘')

num = int(input("Select option: "))
options = {0 : 'Identify Constants',
           1 : 'Time Controlled',
           2 : 'Environment Controlled',
           3 : 'plot'
}
model_target = options[num]
path = model_target

###############################################################################
# LOADING DATA
###############################################################################
if model_target == 'Identify Constants':
    file = 'dataset_full_with_constants.csv'

if model_target == 'Time Controlled':
    file = 'dataset_time_controlled.csv'

if model_target == 'plot':
    file = 'dataset_full_long.csv'

if model_target == 'Environment Controlled':
    print('')
    print('Select model type:')
    print('┌───────────────────────────┐')
    print('│ 0) explain with EC only   │')
    print('│ 1) explain with TC and EC │')
    print('└───────────────────────────┘')

    num = int(input("Select option: "))
    options = { 0 : '/EC',
                1 : '/TC and EC',}
    model_type = options[num]
    path = path + model_type
    if model_type == '/EC':
        file = 'dataset_environment_controlled.csv'
    if model_type == '/TC and EC':
        file = 'dataset_full.csv'
else:
    model_type  = ''

URL = 'datasets/' + file
print('')
print('READING ' + URL, end=' ')
df = pd.read_csv(URL, parse_dates=['Date_Time'], index_col="Date_Time",
                          na_values='NaN', comment='\t',
                          sep=';', skipinitialspace=True)
print(df.shape)

if model_target == 'Time Controlled':
    OUT_FEATURES = None
    df = df.iloc[:, 0:1]
    print('')
    print('SELECTED first feature', end=' ')
    print(df.columns.values.tolist()[0], end=' ')
    print(df.shape)

if model_target == 'Environment Controlled':
    all_features = df.columns.tolist()
    print('')
    print('FEATURES total:' , end='                   ')
    print("{:4.0f}".format(len(all_features)))
    time_controlled_features = []
    if model_type == '/TC and EC':
        time_controlled_features = ['L1-2L BLUE','L1-2R BLUE','L1-4L BLUE','L1-4R BLUE','R4-4R BLUE','R4-4L BLUE','L2-1L BLUE','L2-1R BLUE','L2-2L BLUE','L2-2R BLUE','L2-3L BLUE','L2-3R BLUE','L2-4L BLUE','L2-4R BLUE','L3-1L BLUE','L3-2L BLUE','L3-1R BLUE','L3-2R BLUE','L3-3L BLUE','L3-3R BLUE','L3-4L BLUE','L3-4R BLUE','L4-1L BLUE','L4-1R BLUE','L4-2L BLUE','L4-2R BLUE','L4-3L BLUE','L4-3R BLUE','L4-4L BLUE','L4-4R BLUE','R1-2R BLUE','R1-2L BLUE','R1-4R BLUE','R1-4L BLUE','R2-2R BLUE','R2-2L BLUE','R2-4R BLUE','R2-4L BLUE','R3-2/4R BLUE','R3-2/4L BLUE','R4-2R BLUE','R4-2L BLUE','L1-2L RED','L1-2R RED','L1-4L RED','L1-4R RED','R4-4R RED','R4-4L RED','L2-1L RED','L2-1R RED','L2-2L RED','L2-2R RED','L2-3L RED','L2-3R RED','L2-4L RED','L2-4R RED','L3-1L RED','L3-2L RED','L3-1R RED','L3-2R RED','L3-3L RED','L3-3R RED','L3-4L RED','L3-4R RED','L4-1L RED','L4-1R RED','L4-2L RED','L4-2R RED','L4-3L RED','L4-3R RED','L4-4L RED','L4-4R RED','R1-2R RED','R1-2L RED','R1-4R RED','R1-4L RED','R2-2R RED','R2-2L RED','R2-4R RED','R2-4L RED','R3-2/4R RED','R3-2/4L RED','R4-2R RED','R4-2L RED','L1-2L FAR RED','L1-2R FAR RED','L1-4L FAR RED']

    print('FEATURES Time Controlled:' , end='         ')
    print("{:4.0f}".format(len(time_controlled_features)))

    environment_controlled_features = list(set(all_features) - set(time_controlled_features))
    OUT_FEATURES = environment_controlled_features
    print('FEATURES Environment Controlled:' , end='  ')
    print("{:4.0f}".format(len(environment_controlled_features)), end=' ')
    if len(all_features) == len(time_controlled_features)+len(environment_controlled_features):
        print('match')
    else:
        print('mismatch')


###############################################################################
# VISUALIZE DATA
###############################################################################

if model_target == 'Identify Constants':
    plot_names = ['FEG AIR FLOW','NDS-BASE DOSING PUMP ','NDS-ACID SOLENOID','PDS TEMP CONTROL BOX','SUBFLOOR CPO 2','AMS-FEG-FAN AIR LOOP 1','AMS-FEG-FAN AIR LOOP 2','SES TEMP AIR IN 1','FEG HUMIDITY TARGET','NDS-ACID DOSING PUMP ','NDS-PUMP FW ','FLOW METER TANK 2','SES TEMP AIR IN 2','NDS-SOLENOID FW TANK 1','NDS-REC PUMP TANK 1','NDS-REC PUMP TANK 2','NDS-SOLENOID FW TANK 2','NDS-A DOSING PUMP','NDS-B DOSING PUMP','SES DOOR STATUS']
    plot_features = df[plot_names]
    _ = plot_features.plot(subplots=True, figsize=(10,14))
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.1 )
    plt.xticks(rotation=0)
    plt.savefig('./figures/' + model_target + '/feature.svg')
    plt.show()
    print('end.')
    exit(0)

if model_target == 'plot':
    plot_features =  df.iloc[:, 0:1]
    _ = plot_features.plot(subplots=True, figsize=(10,1))
    #_ = plot_features.plot(subplots=True, figsize=(10,1))
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.1 )
    plt.xticks(rotation=0)
    plt.savefig('./figures/plot.svg')
    plt.show()
    print('end.')
    exit(0)

###############################################################################
# DATASET SEPERATION
###############################################################################

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]


if model_target == 'Time Controlled':
    num_input_features = df.shape[1]
    num_output_features = df.shape[1]

if model_target == 'Environment Controlled':
    num_input_features = df.shape[1]
    num_output_features = len(environment_controlled_features)


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
if model_target == 'Time Controlled':
    plot_col = 'L1-2L BLUE'
if model_target == 'Environment Controlled':
    plot_col = 'FEG CO2 1'
    #plot_col = 'FEG TEMPERATURE 1'

# df.columns[0]   # first feature
def plot(self, model=None, plot_col=plot_col, max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(14, 8))
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
  plt.xticks(rotation=0)

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
      batch_size=554,) # full training samples: 33246

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



multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=OUT_FEATURES)

multi_window_baseline = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)


###############################################################################
# CONTROL SHAPE WITH EXAMPLE
###############################################################################

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:multi_window.total_window_size]),
                           np.array(train_df[1000:1000+multi_window.total_window_size]),
                           np.array(train_df[2000:2000+multi_window.total_window_size])])


multi_window_inputs, multi_window_labels = multi_window.split_window(example_window)
print('')
print('All shapes are: (batch, time, features)')
print('──────────────────────────────')
print(f'  Window shape: {example_window.shape}')
print(f'  Inputs shape: {multi_window_inputs.shape}')
print(f'  labels shape: {multi_window_labels.shape}')
print('──────────────────────────────')



###############################################################################
# COMPILE AND FIT
###############################################################################

# patience 4
def compile_and_fit(model, window, patience=0, load=False):
  checkpoint_path = "cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min', restore_best_weights=True)

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  if not load:
      history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping,cp_callback])
  if load:
      history = None

  return history

###############################################################################


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

if model_target == 'Time Controlled':
    multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
    multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)

if model_target == 'Environment Controlled':
    multi_val_performance['Last'] = last_baseline.evaluate(multi_window_baseline.val)
    multi_performance['Last'] = last_baseline.evaluate(multi_window_baseline.test, verbose=0)


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

    plt.savefig('./models/' + model_target + '/REPEAT.svg')

#####################################################################################
# Neural Network MODELS
####################################################################################


def create_linear_model():
  model = tf.keras.Sequential([
      # Take the last time-step.
      # Shape [batch, time, features] => [batch, 1, features]
      tf.keras.layers.Lambda(lambda x: x[:, -1:, :] , input_shape=(IN_STEPS,num_input_features)),
      # Shape => [batch, 1, out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS*num_output_features,
                            kernel_initializer=tf.initializers.zeros),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
  ])

  return model



def compute_linear():
    print('┌────────────────┐')
    print('│  LINEAR MODEL  │')
    print('└────────────────┘')

    global multi_linear_model

    multi_linear_model = create_linear_model()
    multi_linear_model.summary()

    if purpose == 'train':
        history = compile_and_fit(multi_linear_model, multi_window)

    if purpose == 'load':
        print('LOADING ' + './models/' + path + '/LINEAR.h5')
        multi_linear_model = tf.keras.models.load_model('./models/' + path + '/LINEAR.h5')

    IPython.display.clear_output()
    multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
    multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_linear_model)

    plt.savefig('./models/' + model_target + model_type + '/LINEAR.svg')
    multi_linear_model.save('models/' + model_target + model_type + '\LINEAR.h5', overwrite=True, save_format="tf")
    print('LINEAR.h5 saved')


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
        tf.keras.layers.Dense(OUT_STEPS*num_output_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    if purpose == 'train':
        history = compile_and_fit(multi_dense_model, multi_window)
        print('SUMMARY')
        multi_linear_model.summary()

    if purpose == 'load':
        history = compile_and_fit(multi_dense_model, multi_window, load=True)
        print('LOADING ' + './models/' + path + '/DENSE.h5')
        multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val, verbose=0)
        multi_dense_model.load_weights('./models/' + path + '/DENSE.h5')

    IPython.display.clear_output()
    multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
    multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_dense_model)

    plt.savefig('./models/' + model_target + model_type + '/DENSE.svg')


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
        tf.keras.layers.Dense(OUT_STEPS*num_output_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    if purpose == 'train':
        history = compile_and_fit(multi_conv_model, multi_window)
        print('SUMMARY')
        multi_linear_model.summary()

    if purpose == 'load':
        history = compile_and_fit(multi_conv_model, multi_window, load=True)
        print('LOADING ' + './models/' + path + '/CONV.h5')
        multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val, verbose=0)
        multi_conv_model.load_weights('./models/' + path + '/CONV.h5')

    IPython.display.clear_output()

    multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
    multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_conv_model)

    plt.savefig('./models/' + model_target + model_type + '/CONV.svg')


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
        tf.keras.layers.Dense(OUT_STEPS*num_output_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_output_features])
    ])

    if purpose == 'train':
        history = compile_and_fit(multi_lstm_model, multi_window)
        print('SUMMARY')
        multi_linear_model.summary()

    if purpose == 'load':
        history = compile_and_fit(multi_lstm_model, multi_window, load=True)
        print('LOADING ' + './models/' + path + '/LSTM.h5')
        multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val, verbose=0)
        multi_lstm_model.load_weights('./models/' + path + '/LSTM.h5')

    IPython.display.clear_output()

    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model)

    plt.savefig('./models/' + model_target + model_type + '/LSTM.svg')



# for Auto LSTM
class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_output_features)

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

    if purpose == 'train':
        history = compile_and_fit(feedback_model, multi_window)
        print('SUMMARY')
        multi_linear_model.summary()

    if purpose == 'load':
        history = compile_and_fit(feedback_model, multi_window, load=True)
        print('LOADING ' + './models/' + path + '/AR_LSTM.h5')
        multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val, verbose=0)
        feedback_model.load_weights('./models/' + path + '/AR_LSTM.h5')

    IPython.display.clear_output()

    multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
    multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(feedback_model)

    plt.savefig('./models/' + model_target + model_type + '/AR_LSTM.svg')




def compute_all():
    if model_target == 'Time Controlled':
        compute_repeat()
    compute_linear()
    compute_dense()
    compute_conv()
    compute_lstm()
    if model_target == 'Time Controlled':
        compute_auto_lstm()


print('')
print('Select Model to compute:')
print('┌──────────────────┐')
if model_target == 'Time Controlled':
    print('│ 0) Repeat        │')
print('│ 1) Linear        │')
print('│ 2) Dense         │')
print('│ 3) Convolutional │')
print('│ 4) LSTM          │')
if model_target == 'Time Controlled':
    print('│ 5) Auto LSTM     │')
print('│ 6) all           │')
print('└──────────────────┘')

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

plt.show()
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
plt.savefig('./models/' + model_target + model_type + '/performance.svg')


print('')
print('┌──── Test MAE ────┐')
for name, value in multi_performance.items():
  print('│ ' + f'{name:8s}: {value[1]:0.4f}' + ' │')
print('└──────────────────┘')

with io.open('./models/' + model_target + model_type + '/summary.txt', "a", encoding="utf-8") as f:
    f.write("┌──── Test MAE ────┐\n")
    for name, value in multi_performance.items():
      f.write('│ ' + f'{name:8s}: {value[1]:0.4f}' + ' │\n')
    f.write('└──────────────────┘\n')
    f.write('\n')
    f.close()

###############################################################################
# SAVING MODELS
###############################################################################
print('')
print('SAVING at ' + './models/' + model_target + model_type)

#multi_lstm_model.save('./models/' + model_target + '/REPEAT.h5', save_format="tf")

try:
    multi_linear_model.save_weights('models/' + model_target + model_type + '\LINEAR.h5')
    print('LINEAR.h5 saved')
except: print('', end='')
try:
    multi_dense_model.save_weights('models/' + model_target + model_type + '\DENSE.h5')
    print('DENSE.h5 saved')
except: print('', end='')
try:
    multi_conv_model.save_weights('models/' + model_target + model_type + '\CONV.h5')
    print('CONV.h5 saved')
except: print('', end='')
try:
    multi_lstm_model.save_weights('models/' + model_target + model_type + '\LSTM.h5')
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
plt.show()
print('end.')
