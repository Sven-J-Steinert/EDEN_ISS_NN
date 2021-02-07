```python
datetime_GMTF = pd.to_datetime(dataset_GMTF.index)
datetime_GMTF = pd.DatetimeIndex(((datetime_GMTF.asi8/(1e10*30)).round()*1e10*30).astype(np.int64))  # 5 min steps
dataset_GMTF.index = datetime_GMTF

dataset_GMTF = dataset_GMTF[~dataset_GMTF.index.duplicated(keep='first')]  # remove duplicate indices
del dataset_GMTF['SampleNumber'] # delete SampleNumber

dataset_GMTF = pd.concat([timeframe, dataset_GMTF], axis=1, sort=False, join='outer')
```
