```python
dataset_GMTF.astype('float64')
dataset_GMTF.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
dataset_GMTF = dataset_GMTF.dropna()
```
