## Knowing Our Data

```python
data.head(10)
```

## df dimensions
```python
data.shape
```
## Data type for each feature

```python
data.dtypes
```

## Main stats for numeric features
```python
data.describe()
```

## Correlations
```python
data.corr(method='method')
```
Usually person

* we are interested in the features with high correlation with the class 

* in case of high correlation between features, that means dependence between them, so we just leave one

## Asymmetry (skew)

```python
data.skew()
```