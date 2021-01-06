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
url_GMTF = 'data_GeneralMTF.csv'

raw_data_GMTF = pd.read_csv(url_GMTF, dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_GMTF.shape, end=' ', flush=True)
dataset_GMTF = raw_data_GMTF.copy()


print('EDIT', end=' ', flush=True)
datetime_GMTF = pd.to_datetime(dataset_GMTF.index)
#datetime_GMTF = pd.DatetimeIndex(((datetime_GMTF.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
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
print('missing')
print('number', end=' ')
print(dataset_GMTF.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_GMTF.isnull().sum().sum()/dataset_GMTF.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_GMTF.isnull().sum().sum()/dataset_GMTF.shape[1])*100) / dataset_GMTF.shape[0]), end=' ')
print('%')

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING AMS", end=' ', flush=True)
url_AMS = 'data_AMS.csv'

raw_data_AMS = pd.read_csv(url_AMS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_AMS.shape, end=' ', flush=True)

dataset_AMS = raw_data_AMS.copy()

print('EDIT', end=' ', flush=True)
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
print('missing')
print('number', end=' ')
print(dataset_AMS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_AMS.isnull().sum().sum()/dataset_AMS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_AMS.isnull().sum().sum()/dataset_AMS.shape[1])*100) / dataset_AMS.shape[0]), end=' ')
print('%')



print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING ILS", end=' ', flush=True)
url_ILS = 'data_ILS.csv'

raw_data_ILS = pd.read_csv(url_ILS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_ILS.shape, end=' ', flush=True)
dataset_ILS = raw_data_ILS.copy()

print('EDIT', end=' ', flush=True)
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
print('missing')
print('number', end=' ')
print(dataset_ILS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_ILS.isnull().sum().sum()/dataset_ILS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_ILS.isnull().sum().sum()/dataset_ILS.shape[1])*100) / dataset_ILS.shape[0]), end=' ')
print('%')


print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING NDS", end=' ', flush=True)
url_NDS = 'data_NDS.csv'

raw_data_NDS = pd.read_csv(url_NDS,dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_NDS.shape, end=' ', flush=True)
dataset_NDS = raw_data_NDS.copy()


print('EDIT', end=' ', flush=True)
datetime_NDS = pd.to_datetime(dataset_NDS.index)
#datetime_NDS = pd.DatetimeIndex(((datetime_NDS.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
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
print('missing')
print('number', end=' ')
print(dataset_NDS.isnull().sum().sum(), end=' ')
print('| lines', end=' ')
print(int(dataset_NDS.isnull().sum().sum()/dataset_NDS.shape[1]), end=' | ')
print("%.2f" % ((int(dataset_NDS.isnull().sum().sum()/dataset_NDS.shape[1])*100) / dataset_NDS.shape[0]), end=' ')
print('%')

print('')
print('───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
print("LOADING TCS", end=' ', flush=True)
url_TCS = 'data_TCS.csv'

raw_data_TCS = pd.read_csv(url_TCS, dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='?',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False)

print(raw_data_TCS.shape, end=' ', flush=True)
dataset_TCS = raw_data_TCS.copy()


print('EDIT', end=' ', flush=True)
datetime_TCS = pd.to_datetime(dataset_TCS.index)
#datetime_GMTF = pd.DatetimeIndex(((datetime_GMTF.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))  # round to flat minutes
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
dataset = pd.concat([timeframe, dataset_GMTF, dataset_AMS, dataset_ILS, dataset_NDS], axis=1, sort=False, join='outer')
dataset.index = pd.to_datetime(dataset.index)
dataset.index.name = 'Date_Time'
del dataset['Placeholder']
print(dataset.shape, end=' ', flush = True)
print(' | order: GTMF AMS ILS NDS   (TCS dropped)')
print(dataset.head(10))
print(dataset.tail(10))
print('')
print('missing numbers:', end=' ')
print(dataset.isnull().sum().sum(), end=' | ')
print("%.2f" % ((dataset.isnull().sum().sum()/(dataset.shape[1]*dataset.shape[0]))*100), end=' ')
print('%')

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
