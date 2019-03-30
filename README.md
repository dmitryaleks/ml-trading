# Machine Learning for Algo Trading

## Time Series Models

A time series is often modelled as a stochastic process consisting of a collection of random variables with one variable for each point in time.

Univariate time series: consist of a single value.
Multivariate time series: consist of several observations that can be represented by a vector.

### Time Series Decomposition

Components:
  - (systematic) trend;
  - (systematic) seasonality and cycles;
  - (unsystematic) noise.

```python
components = tsa.seasonal_decompose(data, model='additive')

ts = (data.to_frame('Original')
      .assign(Trend=components.trend)
      .assign(Seasonality=components.seasonal)
      .assign(Residual=components.resid))
```

[Full code](models/ts-decomposition/ts_decompose.py)

Decomposition result:
![Decomposition](models/ts-decomposition/img/ts-decomposition.png)

[Reference materials](https://otexts.com/fpp2/decomposition.html)

### Exponential Smoothing

Forecasts that rely on exponential smoothing methods use weighted averages of past observations, where the weights decay exponentially as the observations get older.

Exponential smoothing is a popular technique based on weighted averages of past observations, with the weights decaying expoentially as the observations get older.

[Reference materials](https://otexts.com/fpp2/ses.html)

### Autocorellation

Autocorellation (also called serial correlation) measures the extent of linear relationship between time series values separated by a given lag.

Auto-Correlation Function (ACF) computes the correlation coefficients as a fution of the lag.

A correlogram is a plot of Auto-Correlation Function (ACF) for sequential lage (k = 0,1,...,n).
A correlogram can be used to detect any autocorrelation after the removal of the effects of deterministic trend or seasonality.

### Stationarity

The statistical properties, such as the mean, variance, or autocorrelation, of a stationary time series are independent of the period, that is, they don't change over time. Hence, stationarity implies that a time series does not have a trend or seasonal effects and that descriptive statistics such as the mean of the standard deviation, when computed for difference rolling windows, are constand or do not change much over time.

Classical statistical models assume that the time series input data is stationary.
