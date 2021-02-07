```python
dataset_GMTF = pd.read_csv(url_GMTF, dayfirst=True,
    parse_dates=[['Date', 'Time']], index_col="Date_Time", na_values='NOTVerified',
    comment='\t', sep=';', skipinitialspace=True, low_memory=False, decimal=',' , thousands='.')
```
