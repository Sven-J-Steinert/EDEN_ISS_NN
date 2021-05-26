import numpy as np                      # math operations
import pandas as pd                     # dataset handling


###############################################################################
# LOADING DATA
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING timeframe.csv", end=' ')
timeframe = pd.read_csv('timeframe.csv', index_col="Date_Time", sep=';', low_memory=False)
timeframe.index = pd.to_datetime(timeframe.index)
print(timeframe.shape)
print('')
print(timeframe.head())
print(timeframe.tail())


print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING General MTF", end=' ', flush=True)
url_GMTF = 'data\data_GeneralMTF.csv'

dataset_GMTF = pd.read_csv(url_GMTF, dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='NOTVerified',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False, decimal=',' , thousands='.')

print(dataset_GMTF.shape, end=' ', flush=True)
print('NAN:', end=' ', flush=True)
print(dataset_GMTF.isnull().sum().sum(), end=' ', flush=True)


print('| EDIT', end=' ', flush=True)
datetime_GMTF = pd.to_datetime(dataset_GMTF.index)
datetime_GMTF = pd.DatetimeIndex(((datetime_GMTF.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_GMTF.index = datetime_GMTF

dataset_GMTF = dataset_GMTF[~dataset_GMTF.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_GMTF['SampleNumber'] # delete SampleNumber

print(dataset_GMTF.shape, end=' ', flush=True)


dataset_GMTF = pd.concat([timeframe, dataset_GMTF], axis=1, sort=False, join='outer')
dataset_GMTF.index = pd.to_datetime(dataset_GMTF.index)
dataset_GMTF.index.name = 'Date_Time'
del dataset_GMTF['Placeholder']


print('APPLYED ON timeframe', end=' ', flush=True)
print(dataset_GMTF.shape)

print('')
print(dataset_GMTF.head())
print(dataset_GMTF.tail())
print('')
print('NAN number', end=' ')
print(dataset_GMTF.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_GMTF.isnull().sum().sum()/dataset_GMTF.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_GMTF.isnull().sum().sum()/dataset_GMTF.shape[1])*100) / dataset_GMTF.shape[0]), end=' ')
print('%')

dataset_GMTF.astype('float64')
dataset_GMTF.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
print('filled. | dropped remaining NAN:', end=' ')
print(dataset_GMTF.isnull().sum().sum(), end=' ')
dataset_GMTF = dataset_GMTF.dropna()

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING AMS", end=' ', flush=True)
url_AMS = 'data\data_AMS.csv'

dataset_AMS = pd.read_csv(url_AMS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='NOTVerified',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False, decimal=',' , thousands='.')

print(dataset_AMS.shape, end=' ', flush=True)
print('NAN:', end=' ', flush=True)
print(dataset_AMS.isnull().sum().sum(), end=' ', flush=True)

print('| EDIT', end=' ', flush=True)
datetime_AMS = pd.to_datetime(dataset_AMS.index)
#datetime_AMS = pd.DatetimeIndex(((datetime_AMS.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
datetime_AMS = pd.DatetimeIndex(((datetime_AMS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_AMS.index = datetime_AMS

dataset_AMS = dataset_AMS[~dataset_AMS.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_AMS['SampleNumber']

print(dataset_AMS.shape, end=' ', flush=True)

dataset_AMS = pd.concat([timeframe, dataset_AMS], axis=1, sort=False, join='outer')
dataset_AMS.index = pd.to_datetime(dataset_AMS.index)
dataset_AMS.index.name = 'Date_Time'
del dataset_AMS['Placeholder']

print('APPLYED ON timeframe', end=' ', flush=True)
print(dataset_AMS.shape)

print('')
print(dataset_AMS.head())
print(dataset_AMS.tail())
print('')
print('NAN number', end=' ')
print(dataset_AMS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_AMS.isnull().sum().sum()/dataset_AMS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_AMS.isnull().sum().sum()/dataset_AMS.shape[1])*100) / dataset_AMS.shape[0]), end=' ')
print('%')

dataset_AMS.astype('float64')
dataset_AMS.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
print('filled. | dropped remaining NAN:', end=' ')
print(dataset_AMS.isnull().sum().sum(), end=' ')
dataset_AMS = dataset_AMS.dropna()

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING ILS", end=' ', flush=True)
url_ILS = 'data\data_ILS.csv'

dataset_ILS = pd.read_csv(url_ILS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='NOTVerified',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False, decimal=',' , thousands='.')

print(dataset_ILS.shape, end=' ', flush=True)
print('NAN:', end=' ', flush=True)
print(dataset_ILS.isnull().sum().sum(), end=' ', flush=True)

print('| EDIT', end=' ', flush=True)
datetime_ILS = pd.to_datetime(dataset_ILS.index)
#datetime_ILS = pd.DatetimeIndex(((datetime_ILS.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
datetime_ILS = pd.DatetimeIndex(((datetime_ILS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_ILS.index = datetime_ILS

dataset_ILS = dataset_ILS[~dataset_ILS.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_ILS['SampleNumber']
print(dataset_ILS.shape, end=' ', flush=True)

dataset_ILS = pd.concat([timeframe, dataset_ILS], axis=1, sort=False, join='outer')
dataset_ILS.index = pd.to_datetime(dataset_ILS.index)
dataset_ILS.index.name = 'Date_Time'
del dataset_ILS['Placeholder']

print('APPLYED ON timeframe', end=' ', flush=True)
print(dataset_ILS.shape)


print('')
print(dataset_ILS.head())
print(dataset_ILS.tail())
print('')
print('NAN number', end=' ')
print(dataset_ILS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_ILS.isnull().sum().sum()/dataset_ILS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_ILS.isnull().sum().sum()/dataset_ILS.shape[1])*100) / dataset_ILS.shape[0]), end=' ')
print('%')

dataset_ILS.astype('float64')
dataset_ILS.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
print('filled. | dropped remaining NAN:', end=' ')
print(dataset_ILS.isnull().sum().sum(), end=' ')
dataset_ILS = dataset_ILS.dropna()

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING NDS", end=' ', flush=True)
url_NDS = 'data\data_NDS.csv'

dataset_NDS = pd.read_csv(url_NDS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='NOTVerified',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False, decimal=',' , thousands='.')

# delete SCHEDULES - meaning unclear
del dataset_NDS['L1 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['L1 HP PUMP IRRIGATION SCHEDULE #2']
del dataset_NDS['L2 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['L2 HP PUMP 2 IRRIGATION SCHEDULE #2']
del dataset_NDS['L3 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['L3 HP PUMP IRRIGATION SCHEDULE #2']
del dataset_NDS['L4 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['L4 HP PUMP IRRIGATION SCHEDULE #2']
del dataset_NDS['R1 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['R1 HP PUMP IRRIGATION SCHEDULE #2']
del dataset_NDS['R2 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['R2 HP PUMP IRRIGATION SCHEDULE #2']
del dataset_NDS['R3 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['R3 HP PUMP IRRIGATION SCHEDULE #2']
del dataset_NDS['R4 HP PUMP IRRIGATION SCHEDULE #1']
del dataset_NDS['R4 HP PUMP IRRIGATION SCHEDULE #2']

print(dataset_NDS.shape, end=' ', flush=True)
print('NAN:', end=' ', flush=True)
print(dataset_NDS.isnull().sum().sum(), end=' ', flush=True)


print('| EDIT', end=' ', flush=True)
datetime_NDS = pd.to_datetime(dataset_NDS.index)
datetime_NDS = pd.DatetimeIndex(((datetime_NDS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_NDS.index = datetime_NDS

dataset_NDS = dataset_NDS[~dataset_NDS.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_NDS['SampleNumber']

print(dataset_NDS.shape, end=' ', flush=True)

dataset_NDS = pd.concat([timeframe, dataset_NDS], axis=1, sort=False, join='outer')
dataset_NDS.index = pd.to_datetime(dataset_NDS.index)
dataset_NDS.index.name = 'Date_Time'
del dataset_NDS['Placeholder']

print('APPLYED ON timeframe', end=' ', flush=True)
print(dataset_NDS.shape)


print('')
print(dataset_NDS.head())
print(dataset_NDS.tail())
print('')
print('NAN number', end=' ')
print(dataset_NDS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_NDS.isnull().sum().sum()/dataset_NDS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_NDS.isnull().sum().sum()/dataset_NDS.shape[1])*100) / dataset_NDS.shape[0]), end=' ')
print('%')

dataset_NDS.astype('float64')
dataset_NDS.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
print('filled. | dropped remaining NAN:', end=' ')
print(dataset_NDS.isnull().sum().sum(), end=' ')
dataset_NDS = dataset_NDS.dropna()

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING TCS", end=' ', flush=True)
url_TCS = 'data\data_TCS.csv'

raw_data_TCS = pd.read_csv(url_TCS, dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='NOTVerified',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False, decimal=',' , thousands='.')

print(raw_data_TCS.shape, end=' ', flush=True)
dataset_TCS = raw_data_TCS.copy()


print('| EDIT', end=' ', flush=True)
datetime_TCS = pd.to_datetime(dataset_TCS.index)
datetime_TCS = pd.DatetimeIndex(((datetime_TCS.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # round to 5 minute steps
dataset_TCS.index = datetime_TCS

dataset_TCS = dataset_TCS[~dataset_TCS.index.duplicated(keep='first')]  # remove duplicate indexes

del dataset_TCS['SampleNumber'] # delete SampleNumber

print(dataset_TCS.shape, end=' ', flush=True)

dataset_TCS = pd.concat([timeframe, dataset_TCS], axis=1, sort=False, join='outer')
dataset_TCS.index = pd.to_datetime(dataset_TCS.index)
dataset_TCS.index.name = 'Date_Time'
del dataset_TCS['Placeholder']

print('APPLYED ON timeframe', end=' ', flush=True)
print(dataset_TCS.shape)

print('')
print(dataset_TCS.head())
print(dataset_TCS.tail())
print('')
print('missing')
print('number', end=' ')
print(dataset_TCS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_TCS.isnull().sum().sum()/dataset_TCS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_TCS.isnull().sum().sum()/dataset_TCS.shape[1])*100) / dataset_TCS.shape[0]), end=' ')
print('%', end='  ---> DROPPED!')
print('')


###############################################################################
# COMBINE DATA
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('COMBINING to final dataset', end=' ', flush = True)
dataset = pd.concat([dataset_GMTF, dataset_AMS, dataset_ILS, dataset_NDS], axis=1, join='inner')
dataset.index = pd.to_datetime(dataset.index)
dataset.index.name = 'Date_Time'
dataset.astype('float64')
dataset.dropna()


print(dataset.shape, end=' ', flush = True)
print(' | order: GTMF AMS ILS NDS   (TCS dropped)', end =' ')
from_datetime = pd.to_datetime('')
to_datetime = pd.to_datetime('')

print(' | apply time cut')
from_datetime = pd.to_datetime('01.03.2018',dayfirst=True)
to_datetime = pd.to_datetime('13.08.2018',dayfirst=True)

dataset = dataset.loc[from_datetime:to_datetime]

print(dataset.head(10))
print(dataset.tail(10))
print('')
print('leftover NAN (should be zero):', end=' ')
print(dataset.isnull().sum().sum(), end=' | ')
print("%.2f" % ((dataset.isnull().sum().sum()/(dataset.shape[1]*dataset.shape[0]))*100), end=' ')
print('%')

###############################################################################
# REMOVE CONSTANTS or CORRUPTED
###############################################################################

dataset_cache = dataset.copy()
del dataset_cache['CPO HUMIDITY']
del dataset_cache['SES TEMPERATURE 2']
del dataset_cache['SES HUMIDITY 2']
del dataset_cache['AMS-SES-FAN AIR IN']
del dataset_cache['AMS-SES-VALVE AIR IN']
del dataset_cache['AMS-SES-HUMIDIFIER AIR IN']
del dataset_cache['AMS-SES-HEATER AIR IN']
del dataset_cache['FEG LVL SWITCH COND MAX']
del dataset_cache['FEG  TEMPERATURE PHM 2']
del dataset_cache['FEG HUMIDITY PHM 2']
del dataset_cache['FEG TEMPERATURE PHM 3']
del dataset_cache['FEG HUMIDITY PHM 3']
del dataset_cache['FEG TEMPERATURE PHM 4']
del dataset_cache['FEG HUMIDITY PHM 4']
del dataset_cache['FLOW METER TANK 1']
del dataset_cache['EC Setpoint']
del dataset_cache['pH Setpoint']
del dataset_cache['EC Setpoint 2']
del dataset_cache['pH Setpoint 2']
del dataset_cache['SES TEMPERATURE']
del dataset_cache['SES HUMIDITY TARGET']
del dataset_cache['AMS-FEG-FANS CIRC L1.L2']
del dataset_cache['AMS-FEG-FANS CIRC L3.L4']
del dataset_cache['AMS-FEG-FANS CIRC R1.R2']
del dataset_cache['AMS-FEG-FANS CIRC R3.R4']
del dataset_cache['AMS-FEG-UV LAMP AIR']
del dataset_cache['NDS-BASE SOLENOID']
del dataset_cache['FEG AIR FLOW']
del dataset_cache['NDS-BASE DOSING PUMP ']
del dataset_cache['NDS-ACID SOLENOID']
del dataset_cache['PDS TEMP CONTROL BOX']
del dataset_cache['SUBFLOOR CPO 2']
del dataset_cache['AMS-FEG-FAN AIR LOOP 1']
del dataset_cache['AMS-FEG-FAN AIR LOOP 2']
del dataset_cache['SES TEMP AIR IN 1']
del dataset_cache['FEG HUMIDITY TARGET']
del dataset_cache['NDS-PUMP FW ']
del dataset_cache['FLOW METER TANK 2']
del dataset_cache['SES TEMP AIR IN 2']
del dataset_cache['NDS-REC PUMP TANK 1']
del dataset_cache['NDS-REC PUMP TANK 2']


###############################################################################
# SPLIT DATASET
###############################################################################

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('SPLITTING dataset into \'time_controlled\' and \'environment_controlled\'')

ILS_time_controlled = ['L1-2L BLUE','L1-2R BLUE','L1-4L BLUE','L1-4R BLUE','R4-4R BLUE','R4-4L BLUE','L2-1L BLUE','L2-1R BLUE','L2-2L BLUE','L2-2R BLUE','L2-3L BLUE','L2-3R BLUE','L2-4L BLUE','L2-4R BLUE','L3-1L BLUE','L3-2L BLUE','L3-1R BLUE','L3-2R BLUE','L3-3L BLUE','L3-3R BLUE','L3-4L BLUE','L3-4R BLUE','L4-1L BLUE','L4-1R BLUE','L4-2L BLUE','L4-2R BLUE','L4-3L BLUE','L4-3R BLUE','L4-4L BLUE','L4-4R BLUE','R1-2R BLUE','R1-2L BLUE','R1-4R BLUE','R1-4L BLUE','R2-2R BLUE','R2-2L BLUE','R2-4R BLUE','R2-4L BLUE','R3-2/4R BLUE','R3-2/4L BLUE','R4-2R BLUE','R4-2L BLUE','L1-2L RED','L1-2R RED','L1-4L RED','L1-4R RED','R4-4R RED','R4-4L RED','L2-1L RED','L2-1R RED','L2-2L RED','L2-2R RED','L2-3L RED','L2-3R RED','L2-4L RED','L2-4R RED','L3-1L RED','L3-2L RED','L3-1R RED','L3-2R RED','L3-3L RED','L3-3R RED','L3-4L RED','L3-4R RED','L4-1L RED','L4-1R RED','L4-2L RED','L4-2R RED','L4-3L RED','L4-3R RED','L4-4L RED','L4-4R RED','R1-2R RED','R1-2L RED','R1-4R RED','R1-4L RED','R2-2R RED','R2-2L RED','R2-4R RED','R2-4L RED','R3-2/4R RED','R3-2/4L RED','R4-2R RED','R4-2L RED','L1-2L FAR RED','L1-2R FAR RED','L1-4L FAR RED']

identified_columns = ['SES TEMPERATURE TARGET','FEG CO2 TARGET','FEG TEMPERATURE TARGET','NDS-OZONE GENERATOR','FEG PAR LIGHT 2','FEG PAR LIGHT 1']

time_controlled_columns = ILS_time_controlled + identified_columns

dataset_time_controlled = dataset_cache[time_controlled_columns]
dataset_environment_controlled = dataset_cache.drop(time_controlled_columns, axis=1)

###############################################################################
# WRITE DATASET
###############################################################################
print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print('WRITING dataset_full.csv', end=' ', flush = True)
dataset_cache.to_csv('datasets\dataset_full.csv', sep=';', encoding='utf-8')



print('                           done.')
print('WRITING dataset_time_controlled.csv', end=' ', flush = True)
dataset_time_controlled.to_csv('datasets\dataset_time_controlled.csv', sep=';', encoding='utf-8')
print('                done.')
print('WRITING dataset_environment_controlled.csv', end=' ', flush = True)
dataset_environment_controlled.to_csv('datasets\dataset_environment_controlled.csv', sep=';', encoding='utf-8')
print('         done.')


###############################################################################
# VERIFY
###############################################################################
#print('')
#print('READING dataset.csv', end=' ', flush = True)

#dataset_readin = pd.read_csv('datasets\dataset_full.csv',index_col="Date_Time", sep=';', low_memory=False)
#dataset_readin.index = pd.to_datetime(dataset_readin.index)
#dataset.index.name = 'Date_Time'
#if dataset_readin.equals(dataset):
#    print('     verified.')
#else: print('     mismatch.')


print('end.')
