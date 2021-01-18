# EDEN_ISS_NN
Neural Network for timeseries forecasting of the EDEN ISS System

order of executions to recontruct from the ground up:
```
run build_timeframe.py       -> timeframe.csv
run build_dataset.py         -> dataset.csv
run build_EDEN_ISS_NN.py     -> model_dense ; model_conv ; model_lstm
```

Use the pre-trained Models
```
run Model_Launcher.py
```
