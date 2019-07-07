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

### Achieve stationarity of Time Series with transformations

  - application of the natural logarithm to convert an exponential growth pattern into a linear trend;
  - deflation (de-trending): dividing a time series by another series that causes trending behavior;
  - differencing: sutracting neighboring data points or values at seasonal lags from each other.

### Random walk as a model for equity prices

Random walk is an autoregressive stochastic process of the following form:

```
p(t) = p(t-1) + E(t) = Sum(E(i)) + p0
```

i.e. price at time t is a sum of all disturbances E(i) and the initial price (E.g. an IPO price).

Such a time series could, over time, assume any variable. On the other hand, taking the first difference:

```
delta(p(t)) = p(t) - p(t-1)
```

leaves:

```
delta(p(t)) = E(t)
```

which is stationary.

#### Diagnosing non-stationary time series

Plot a correlogram and check whether there is a distinct maximum point on it that is greater than 0.5.

Time series for stock prices are autocorrelated with lag one (are Lag-1 autocorrelated).
Lag-1 autocorrelation of stock price tends to be higher than 0.5.


#### Implications and properties

Since it is enough to do a differencing once (the process is said to be "integrated of order one", a random walk of equity prices is therefore a unit-root non-stationary series.

It has a "long memory": since current values are the sum of past disturbances, large innovations persist for much longer than for a mean-reverting, stationary series.

### Unit root tests

The augmented Dickey-Fuller (ADF) test evaluates the null hypothesis that a time series sample has unit root against the alternative of stationarity. It is available in the statsmodels Python package.

## Log returns distribution and investment risk analysis

[Log returns distribution and investmant risk analysis](models/returns-distribution/stock-return-distribution.ipynb)

## Naive Bayes Classifier model

[Naive Bayes Classifier model](models/naive-bayes/naive-bayes.ipynb)
