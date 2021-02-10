```python
dataset = pd.concat([dataset_GMTF, dataset_AMS, dataset_ILS, dataset_NDS], axis=1, join='inner')

from_datetime = pd.to_datetime('01.03.2018',dayfirst=True)
to_datetime = pd.to_datetime('13.08.2018',dayfirst=True)

dataset = dataset.loc[from_datetime:to_datetime]
```
