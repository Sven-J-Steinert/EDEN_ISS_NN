import numpy as np                      # math operations
import pandas as pd                     # dataset handling

###############################################################################
# LOADING DATA
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING General MTF", end=' ', flush=True)
url_GMTF = 'data_GeneralMTF.csv'
column_names_GMTF = ['SampleNumber','Date','Time','TCS TEMP OUT EXIT',
    'TCS TEMP OUT WINDOW','PDS TEMP CONTROL BOX','SUBFLOOR CPO 1',
    'SUBFLOOR CPO 2','SES DOOR STATUS']

raw_data_GMTF = pd.read_csv(url_GMTF, names=column_names_GMTF, dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_GMTF.shape, end=' ', flush=True)
dataset_GMTF = raw_data_GMTF.copy()

print('EDIT', end=' ', flush=True)
datetime_GMTF = pd.to_datetime(dataset_GMTF.index)
datetime_GMTF = pd.DatetimeIndex(((datetime_GMTF.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
datetime_GMTF = pd.DatetimeIndex(((datetime_GMTF.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_GMTF.index = datetime_GMTF

dataset_GMTF = dataset_GMTF[~dataset_GMTF.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_GMTF['SampleNumber'] # delete SampleNumber

print(dataset_GMTF.shape)
print('')
print(dataset_GMTF.head())
print(dataset_GMTF.tail())
print('')
print('missing variables:', end=' ')
print(dataset_GMTF.isnull().sum().sum())

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING AMS", end=' ', flush=True)
url_AMS = 'data_AMS.csv'
column_names_AMS = ['﻿SampleNumber','Date','Time','SES TEMPERATURE','CPO HUMIDITY',
    'SES TEMPERATURE 1','SES HUMIDITY 1','SES PAR LIGHT 1','SES CO2 1',
    'SES CALCULATED VPD 1','SES TEMP AIR IN 1','SES TEMP AIR IN 2',
    'SES TEMPERATURE 2','SES HUMIDITY 2','FEG TEMPERATURE 1',
    'SES TEMPERATURE TARGET','SES HUMIDITY TARGET','AMS-SES-FAN AIR IN',
    'AMS-SES-VALVE AIR IN','AMS-SES-HUMIDIFIER AIR IN','AMS-SES-HEATER AIR IN',
    'FEG HUMIDITY 1','FEG PAR LIGHT 1','FEG CO2 1','FEG TEMPERATURE 2',
    'FEG HUMIDITY 2','FEG PAR LIGHT 2','FEG CO2 2','FEG LVL SWITCH COND MAX',
    'FEG TEMPERATURE AMS 1','FEG HUMIDITY AMS 1','FEG TEMPERATURE AMS 2',
    'FEG HUMIDITY AMS 2','FEG TEMPERATURE PHM 1','FEG HUMIDITY PHM 1',
    'FEG TEMPERATURE PHM 2','FEG HUMIDITY PHM 2','FEG TEMPERATURE PHM 3',
    'FEG HUMIDITY PHM 3','FEG TEMPERATURE PHM 4','FEG HUMIDITY PHM 4',
    'FEG AIR FLOW','FEG OXYGEN','FEG CALCULATED VPD 1','FEG CALCULATED VPD 2',
    'FEG CO2 TARGET','FEG TEMPERATURE TARGET','FEG HUMIDITY TARGET',
    'AMS-FEG-FANS CIRC L1/L2','AMS-FEG-FANS CIRC L3/L4',
    'AMS-FEG-FANS CIRC R1/R2','AMS-FEG-FANS CIRC R3/R4','AMS-FEG-FAN AIR LOOP 1',
    'AMS-FEG-FAN AIR LOOP 2','AMS-FEG-UV LAMP AIR','AMS-FEG-HEATER AIR']

raw_data_AMS = pd.read_csv(url_AMS, names=column_names_AMS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_AMS.shape, end=' ', flush=True)

dataset_AMS = raw_data_AMS.copy()

print('EDIT', end=' ', flush=True)
datetime_AMS = pd.to_datetime(dataset_AMS.index)
datetime_AMS = pd.DatetimeIndex(((datetime_AMS.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
datetime_AMS = pd.DatetimeIndex(((datetime_AMS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_AMS.index = datetime_AMS

dataset_AMS = dataset_AMS[~dataset_AMS.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_AMS['﻿SampleNumber']

print(dataset_AMS.shape)
print('')
print(dataset_AMS.head())
print(dataset_AMS.tail())
print('')
print('missing variables:', end=' ')
print(dataset_AMS.isnull().sum().sum())

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING NDS", end=' ', flush=True)
url_NDS = 'data_NDS.csv'
column_names_NDS = [ 'SampleNumber','Date','Time','TEMP 1 TANK 1','TEMP 2 TANK 1',
    'LEVEL SENSOR TANK 1','pH 1 TANK 1','pH 2 TANK 1','EC 1 TANK 1','EC 2 TANK 1',
    'FLOW METER TANK 1','VOLUME TANK 1','EC Setpoint','pH Setpoint',
    'L1 HP PUMP IRRIGATION SCHEDULE #1','L1 HP PUMP IRRIGATION SCHEDULE #2',
    'L2 HP PUMP IRRIGATION SCHEDULE #1','L2 HP PUMP 2 IRRIGATION SCHEDULE #2',
    'TEMP 1 TANK 2','TEMP 2 TANK 2','pH 1 TANK 2','pH 2 TANK 2','EC 1 TANK 2',
    'EC 2 TANK 2','LEVEL SENSOR TANK 2','FLOW METER TANK 2','VOLUME TANK 2',
    'EC Setpoint 2','pH Setpoint 2','L3 HP PUMP IRRIGATION SCHEDULE #1',
    'L3 HP PUMP IRRIGATION SCHEDULE #2','L4 HP PUMP IRRIGATION SCHEDULE #1',
    'L4 HP PUMP IRRIGATION SCHEDULE #2','NDS-REC PUMP TANK 1',
    'NDS-SOLENOID FW TANK 1','NDS-A DOSING PUMP','NDS-B DOSING PUMP',
    'NDS-REC PUMP TANK 2','NDS-SOLENOID FW TANK 2','NDS-C DOSING PUMP',
    'NDS-D DOSING PUMP','NDS-PUMP FW ','NDS-ACID SOLENOID',
    'NDS-BASE SOLENOID','NDS-ACID DOSING PUMP ','NDS-BASE DOSING PUMP ',
    'NDS-OZONE GENERATOR','R1 HP PUMP IRRIGATION SCHEDULE #1',
    'R1 HP PUMP IRRIGATION SCHEDULE #2','R2 HP PUMP IRRIGATION SCHEDULE #1',
    'R2 HP PUMP IRRIGATION SCHEDULE #2','R3 HP PUMP IRRIGATION SCHEDULE #1',
    'R3 HP PUMP IRRIGATION SCHEDULE #2','R4 HP PUMP IRRIGATION SCHEDULE #1',
    'R4 HP PUMP IRRIGATION SCHEDULE #2' ]

raw_data_NDS = pd.read_csv(url_NDS, names=column_names_NDS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_NDS.shape, end=' ', flush=True)
dataset_NDS = raw_data_NDS.copy()


print('EDIT', end=' ', flush=True)
datetime_NDS = pd.to_datetime(dataset_NDS.index)
datetime_NDS = pd.DatetimeIndex(((datetime_NDS.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
datetime_NDS = pd.DatetimeIndex(((datetime_NDS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_NDS.index = datetime_NDS

dataset_NDS = dataset_NDS[~dataset_NDS.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_NDS['SampleNumber']

print(dataset_NDS.shape)
print('')
print(dataset_NDS.head())
print(dataset_NDS.tail())
print('')
print('missing variables:', end=' ')
print(dataset_NDS.isnull().sum().sum())


print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING ILS", end=' ', flush=True)
url_ILS = 'data_ILS.csv'
column_names_ILS = [ 'SampleNumber','Date','Time','L1-2L','L1-2R','L1-4L','L2-4 R',
    'L2-4L','R4-4R','L2-1L','L2-1R','R4-4L','L2-2L','L2-2R','L2-3L','L2-3R',
    'L2-4R','L3-1L','L3-1R','L3-2L','L3-2R','L3-3L','L3-3R','L3-4L','L3-4R',
    'L4-1L','L4-1R','L4-2L','L4-2R','L4-3L','R4-2L','R4-2R','R3-4L','R3-4R',
    'R2-4L','L4-3R','L4-4L','L4-4R','R1-2R','R1-2L','R1-4R','R1-4L','R2-4R',
    'R2-2L','R2-2R','L1-2L BLUE','L1-2R BLUE','L1-4L BLUE','L1-4R BLUE',
    'R4-4R BLUE','R4-4L BLUE','L2-1L BLUE','L2-1R BLUE','L2-2L BLUE',
    'L2-2R BLUE','L2-3L BLUE','L2-3R BLUE','L2-4L BLUE','L2-4R BLUE',
    'L3-1L BLUE','L3-2L BLUE','L3-1R BLUE','L3-2R BLUE','L3-3L BLUE',
    'L3-3R BLUE','L3-4L BLUE','L3-4R BLUE','L4-1L BLUE','L4-1R BLUE',
    'L4-2L BLUE','L4-2R BLUE','L4-3L BLUE','L4-3R BLUE','L4-4L BLUE',
    'L4-4R BLUE','R1-2R BLUE','R1-2L BLUE','R1-4R BLUE','R1-4L BLUE',
    'R2-2R BLUE','R2-2L BLUE','R2-4R BLUE','R2-4L BLUE','R3-2/4R BLUE',
    'R3-2/4L BLUE','R4-2R BLUE','R4-2L BLUE','L1-2L RED','L1-2R RED',
    'L1-4L RED','L1-4R RED','R4-4R RED','R4-4L RED','L2-1L RED',
    'L2-1R RED','L2-2L RED','L2-2R RED','L2-3L RED','L2-3R RED',
    'L2-4L RED','L2-4R RED','L3-1L RED','L3-2L RED','L3-1R RED',
    'L3-2R RED','L3-3L RED','L3-3R RED','L3-4L RED','L3-4R RED',
    'L4-1L RED','L4-1R RED','L4-2L RED','L4-2R RED','L4-3L RED',
    'L4-3R RED','L4-4L RED','L4-4R RED','R1-2R RED','R1-2L RED',
    'R1-4R RED','R1-4L RED','R2-2R RED','R2-2L RED','R2-4R RED',
    'R2-4L RED','R3-2/4R RED','R3-2/4L RED','R4-2R RED','R4-2L RED',
    'L1-2L FAR RED','L1-2R FAR RED','L1-4L FAR RED' ]

raw_data_ILS = pd.read_csv(url_ILS, names=column_names_ILS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_ILS.shape, end=' ', flush=True)
dataset_ILS = raw_data_ILS.copy()

print('EDIT', end=' ', flush=True)
datetime_ILS = pd.to_datetime(dataset_ILS.index)
datetime_ILS = pd.DatetimeIndex(((datetime_ILS.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
datetime_ILS = pd.DatetimeIndex(((datetime_ILS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_ILS.index = datetime_ILS

dataset_ILS = dataset_ILS[~dataset_ILS.index.duplicated(keep='first')]  # remove duplicate indexes


del dataset_ILS['SampleNumber']

print(dataset_ILS.shape)
print('')
print(dataset_ILS.head())
print(dataset_ILS.tail())
print('')
print('missing variables:', end=' ')
print(dataset_ILS.isnull().sum().sum())


print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING timeframe.csv")
timeframe = pd.read_csv('timeframe.csv', index_col="Date_Time", sep=';', low_memory=False)
timeframe.index = pd.to_datetime(timeframe.index)
print(timeframe.shape)
print(timeframe.head())
print(timeframe.tail())
print('')
print('missing variables:', end=' ')
print(timeframe.isnull().sum().sum())

###############################################################################
# COMBINE DATA
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('COMBINING to dataset', end=' ', flush = True)
dataset = pd.concat([timeframe, dataset_GMTF, dataset_AMS, dataset_NDS, dataset_ILS], axis=1, sort=False, join='outer')
dataset.index = pd.to_datetime(dataset.index)
dataset.index.name = 'Date_Time'
del dataset['Placeholder']
print(dataset.shape, end=' ', flush = True)
print(' | order: GTMF AMS NDS ILS')
print(dataset.head(10))
print(dataset.tail(10))
print('')
print('missing variables:', end=' ')
print(dataset.isnull().sum().sum())

###############################################################################
# WRITE DATASET
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('WRITING dataset.csv', end=' ', flush = True)
dataset.to_csv('dataset.csv', sep=';', encoding='utf-8')
print('         done.')
print('READING dataset.csv', end=' ', flush = True)

dataset_readin = pd.read_csv('dataset.csv',index_col="Date_Time", sep=';', low_memory=False)
dataset_readin.index = pd.to_datetime(dataset_readin.index)
if dataset_readin.equals(dataset):
    print('     verified.')
else: print('     mismatch.')


print('end.')
