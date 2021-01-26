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
df_raw = pd.read_csv(URL, parse_dates=['Date_Time'], index_col="Date_Time",
                          na_values='NaN', comment='\t',
                          sep=';', skipinitialspace=True)
print(df_raw.shape)

if options[num] == 'dataset_full.csv':
    time_controlled_features = ['L1-2L BLUE','L1-2R BLUE','L1-4L BLUE','L1-4R BLUE','R4-4R BLUE','R4-4L BLUE','L2-1L BLUE','L2-1R BLUE','L2-2L BLUE','L2-2R BLUE','L2-3L BLUE','L2-3R BLUE','L2-4L BLUE','L2-4R BLUE','L3-1L BLUE','L3-2L BLUE','L3-1R BLUE','L3-2R BLUE','L3-3L BLUE','L3-3R BLUE','L3-4L BLUE','L3-4R BLUE','L4-1L BLUE','L4-1R BLUE','L4-2L BLUE','L4-2R BLUE','L4-3L BLUE','L4-3R BLUE','L4-4L BLUE','L4-4R BLUE','R1-2R BLUE','R1-2L BLUE','R1-4R BLUE','R1-4L BLUE','R2-2R BLUE','R2-2L BLUE','R2-4R BLUE','R2-4L BLUE','R3-2/4R BLUE','R3-2/4L BLUE','R4-2R BLUE','R4-2L BLUE','L1-2L RED','L1-2R RED','L1-4L RED','L1-4R RED','R4-4R RED','R4-4L RED','L2-1L RED','L2-1R RED','L2-2L RED','L2-2R RED','L2-3L RED','L2-3R RED','L2-4L RED','L2-4R RED','L3-1L RED','L3-2L RED','L3-1R RED','L3-2R RED','L3-3L RED','L3-3R RED','L3-4L RED','L3-4R RED','L4-1L RED','L4-1R RED','L4-2L RED','L4-2R RED','L4-3L RED','L4-3R RED','L4-4L RED','L4-4R RED','R1-2R RED','R1-2L RED','R1-4R RED','R1-4L RED','R2-2R RED','R2-2L RED','R2-4R RED','R2-4L RED','R3-2/4R RED','R3-2/4L RED','R4-2R RED','R4-2L RED','L1-2L FAR RED','L1-2R FAR RED','L1-4L FAR RED']

df_raw.astype('float64')
print(df_raw.tail())
print('')
print('missing variables:', end=' ')
print(df_raw.isna().sum().sum())

g = open("eval_last.csv", "w")
f = open("eval_repeat.csv", "w")

for x_counter in range(0,df_raw.shape[1]):

    # short to one feature
    df = df_raw.iloc[:, x_counter:x_counter+1]

    ###############################################################################
    # VISUALIZE DATA
    ###############################################################################

    # plot first 6 columns
    #plot_features = df.iloc[:, 0:6]
    # plot all
    #plot_features = df
    #print(plot_features)
    #plot_features.index = df.index
    #_ = plot_features.plot(subplots=True, legend=False)
    #plt.show()


    ###############################################################################
    # DATASET SEPERATION
    ###############################################################################

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.1)]
    val_df = df[int(n*0.1):int(n*0.2)]
    test_df = df

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

    train_mean = df.mean()
    train_std = df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    print('-> features scaled.')





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
    print(df.columns[0])
    print('')

    def plot(self, model=None, plot_col=df.columns[0], max_subplots=3):
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

    #multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
    multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)


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

        #multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
        multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)




    compute_repeat()

    ###############################################################################
    # PERFORMANCE
    ###############################################################################

    x = np.arange(len(multi_performance))
    width = 0.3


    metric_name = 'mean_absolute_error'

    metric_index = repeat_baseline.metrics_names.index('mean_absolute_error')
    #val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in multi_performance.values()]



    print('')
    print('┌──── summary ─────┐')
    for name, value in multi_performance.items():
      print('│ ' + f'{name:8s}: {value[1]:0.4f}' + ' │')
    print('└──────────────────┘')

    print('SD = ', end=' ')
    print(train_std)
    print(train_mean)
    try:
        sd = f'{train_std.iat[0]:0.2f}'
        mean = f'{train_mean.iat[0]:0.2f}'
        print(sd)
        print(mean)
        pass
    except Exception as e:
        raise

    for name, value in multi_performance.items():

        if name == 'Last':
            g.write(df_raw.columns[x_counter] + ';' + mean + ';' + sd + ';' + f'{value[1]:0.4f}' + '\n')


        if name == 'Repeat':
            f.write(df_raw.columns[x_counter] + ';' + mean + ';' + sd + ';' + f'{value[1]:0.4f}' + '\n')



    print('end.')
    print(x_counter)
    x_counter = x_counter + 1

f.close()
g.close()
