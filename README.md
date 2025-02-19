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

## Sampling

There are two ways to conduct sampling of a given population:

  - Sampling without replacement: each time we select an item from a population, we remove it from the population (i.e. an item can be selected at most once);
  - Sampling with replacement: each time we select an item from a population, we do not remove it from the population.

In Pandas replacement is controlled by the 'replace' argument of the 'sample' method:

```python
data['price'].sample(42, replace=False)
```

### Sample Variance

When calculating sample variance the degrees of freedom should be set to one to closer match the population variance. This effectively means that we are using (n - 1) as a denominator in the variance formula, where n is the size of the sample.

Degrees of Freedom is the number of values in calculation that are free to variate.

### Sample statistics

Distribution of a Sample Mean drawn from a population that follows the Normal Distribution with mean M and variance V also follows a normal distribution with mean M and variance V/n, where n is the sample size.

## Central Limit Theorem

CLT suggests that even if the sample population is not normal, if sample size is large enough the distribution of sample mean is approximately normal with N(M, (sigma^2)/n), where n is the sample size.

[Central Limit Theorem](models/sample-variance/central-limit-theorem.ipynb)

## Confidence Interval

A Confidence Interval (CI) is a type of interval estimate, computed from the statistics of the observed data, that might contain the true value of an unknown population parameter. The interval has an associated confidence level that, loosely speaking, quantifies the level of confidence that the parameter lies in the interval.

For example, a sample mean can give us an idea about the population mean. Namely, by finding a sample mean we could assert that population mean lies within certain bounds around the sample mean.

It is required to standardize the Sample Mean distribution (we are getting a Z-distribution):

```
Z = (x - Mu)/(sigma/sqrt(n))
```

We then find two central (2nd and 3rd) quantiles of a Z distribution that cover a required level of confidence, E.g. 95% of the distribution. Those quantiles have area of (1 - Alpha), while two other quantiles are of size Alpha/2 each. (1 - Alpha) being 95% of the total distribution gives us a 95% confidence interval.


[Confidence Interval for Log returns of a stock](models/confidence-interval/confidence-interval-for-stock-returns.ipynb)

## Hypothesis testing

In statistics, hypothesis testing uses sample information to test validity of conjectures.

We first set a hypothesis:

  - H_zero: Null Hypothesis;
  - H_a: Alternative Hypothesis.

We can consider a t_distribution, which is of a form:

```
t = (x - Mu)/(s/sqrt(n)), where s is the standard deviation of the sample.
```

t distribution gets close to z distribution as the sample size gets larger, therefore:

```
z_hat = t = (x - Mu)/(s/sqrt(n))
``` 

We set a Rejection Region on z distribution based on the Level of Significance, E.g. alpha = 5%. The test can be Two Tails (with a symmetric rejection region) or One Tail (with a rejection region on only one side of z distribution).

We calculate the value of z_hat statistics, E.g. if our null hypothesis is that the mean of population is zero, we calculate the value of z_hat, setting Mu to zero and obtain a single numberic value for z_hat.

We then check with the rejection reason to see whether a given value of z_hat is expected only under rejection region (E.g. left or right 2.5% of the distribution).

It is possible that null hypothesis is correct and we are making a type-one error (false positive), as we are rejecting a true null hypothesis and ending up claiming that our alternative hypothesis is true while it is not.

[Hypothesis testing for average stock returns](models/hypothesis-testing/stock-return-hypothesis-testing.ipynb)

## Distribution of a sum of multiple normal random variables

Distribution of a sum of multiple normally distributed random variables is normal with Mean = N*Mean(v) and Variance = N*Variance(v), where N is the number of input random variables.

[Distribution of a sum of normally distributed random variables](models/sum-of-random-variables/sum-of-random-variables.ipynb)

## Significance testing: p-value

"Hypothetical frequency called the P-value, also known as the “observed significance level” for the test hypothesis.

The distance between the data and the model prediction is measured using a test statistic (such as a t-statistic or a chi-squared statistic). The P-value is then the probability that the chosen test statistic would have been at least as large
as its observed value if every model assumption were correct, including the test hypothesis. This definition embodies a crucial point lost in traditional definitions: In logical terms, the P-value tests all the assumptions about how the data were generated (the entire model), not just the targeted hypothesis it is supposed to test (such as a null hypothesis).

The best way to build a good statistical model is by calculating confidence intervals and nowadays many journals requires confidence intervals."


[Source: Towards Data Science](https://towardsdatascience.com/what-is-a-p-value-b9e6c207247f)

[Reference: Greenland 2016: "Statistical tests, P values, confidence intervals, and power: a guide
to misinterpretations"](papers/greenland2016-statistical-tests.pdf)

![p-value and significance testng](img/p-value_in_statistical_significance_testing.png)

[Example of p-value significance testing for a stock returns hypothesis](models/hypothesis-testing/stock-return-hypothesis-testing.ipynb)

## Z-distribution

Z-distribution is the Standard Normal Distribution (SND).  The Standard Normal Distribution is a specific instance of the Normal Distribution that has a mean of ‘0’ and a standard deviation of ‘1’: N(0,1).

Standardizing a given distribution to get an SND is a powerful technique that makes it easier to calculate probabilities for decision criteria when testing hypothesis.

Standardization is done as follows:
```
Z = (X - M)/d
```

Z-tables for various p-Levels exist to assist manual to facilite Test of Significance.

## Association between random variables

"In probability theory and statistics, covariance is a measure of the joint variability of two random variables. If the greater values of one variable mainly correspond with the greater values of the other variable, and the same holds for the lesser values, (i.e., the variables tend to show similar behavior), the covariance is positive. In the opposite case, when the greater values of one variable mainly correspond to the lesser values of the other, (i.e., the variables tend to show opposite behavior), the covariance is negative. The sign of the covariance therefore shows the tendency in the linear relationship between the variables. The magnitude of the covariance is not easy to interpret because it is not normalized and hence depends on the magnitudes of the variables."

```
cov(X,Y) = E[(X - E(X))(Y - E(Y))]
```

The normalized version of the covariance, the correlation coefficient, however, shows by its magnitude the strength of the linear relation:

```
corr(X,Y) = cov(X,Y)/(Sigma_X * Sigma_Y)
```

## Linear Regression model

Models a relation between predictors and response:

```
y = f(x_1, x_2, ..., x_n) = beta_0 + beta_1 * x_1 + ... + beta_n * x_n
```

### Simple Linear Regression (single predictor)

Population model:
```
y_i = beta_0 + beta_1 * x_i + e_i,

where:

residual error e_i ~ N(0, sigma^2)
beta_0: y-intercept;
beta_1: coefficient of slope.
```


We use sample to estmate beta_0, beta_1 and sigma.

Mean equasion:
```
Mu(y)|x_i = beta_0 + beta_1 * x_i
```

Assumptions:

  - linearity (the mean of response is linearly determined by predictors);
  - independence (with different x_i responses are independent);
  - normality (residual noise and reponse follow normal distribution).
  - equal variance (variance is equal b/w responses).

#### Model fitting

We use the orinary least square estimation to find the best values beta_0 and beta_1 that would describe the data the best.

[Fitting a Linear Regression Model to data](models/regression-model/linear-regression.ipynb)

##### Ordinary Least Squares (OLS)

Residual: the difference between the model's prediction and the actual outcome for a given data point;
Error: the deviation of the true model from the true output in the population.

Least square estimation method chooses the coefficent vector to minimise the Residual Sum of Squares (RSS):

```
argmin(beta) = RSS
```

Optimal parameters vector that minimises RSS can be found on the training set as follows:

```
beta = (1/(X_transposed * X)) * (X_transposed * y)
```

##### The Gauss-Markov theorem

GMT defines the assumptions required for OLS to produce unbiased estimates of the model parameters:

  - in the population, linearity holds;
  - input data are a random sample from the population;
  - no perfect collinearity - thre are no exact linear relationships among the input variables;
  - the error has a conditional mean of zero given any of the inputs;
  - homaskedasticity: the error term 'e' has constant variance given the inputs.

##### Maximum Likelihood Estimation (MLE)

MLE estimates the parameters of a statistical model. It relies on the likelihood function that computes how likely it is to observe the sample of output values for a given set of input data as a function of the model paramteres. Likelihood differs from probabilities in that it is not normalized to range [0,1].

E.g. for the linear regression we can set up the likeliood function by assuming a distribution for the error term, such as the standard normal distribution:

```
e(i) ~ N(0,1)
```

This allows us to compute the conditional probability of observing a given output y(i) given the corresponding input vector x(i) and the parameters:

```
p(y(i)|x(i), beta) = (1/sigma*sqrt(2*Pi))*exp(-(e(i)^2)/(2*sigma)) = (1/sigma*sqrt(2*Pi))*exp(-((y(i) - x(i)*beta)^2)/(2*sigma))
```

Assuming the output values are conditionally independent given the inputs, the likelihood of the sample is proportional to the product of the conditional probabilities of the individual output data points. Since it is easier to work with sums than with products, we apply the logarithm to obtain the log-likelihood function:

```
log_L(y,x,beta) = sum(log(1/(sigma*sqrt(2*Pi))*exp(-((y(i) - x(i)*beta)^2)/(2*sigma))))
```

The goal of MLE is to maximize the likelihood of the output sample that has in fact been observed by choosing model parameters, taking the observed inputs as given. Hence the MLE parameter estimate results from maximising the log-likelihood function:

```
beta_mle = argmax(beta | log_L)
```

Due to the assumption of normal distribution, maximizing the log-likelihood function produces the same parameter solution as least squares because the only expression that depends on the parameters (beta) is squared residual in the exponent. For other distributional assumptions and models, MLE will produce different results, and in many cases, least squares is not applicable.

#### Model evaluation

Calculate metrics:

  - significance of beta_1 term: calculate a p-value for a null hypothesis of "beta_1 is zero" and decide whether null hypothesis can be rejected;
  - calculate confidence interval for beta_1;
  - calculate R-sqared metric: coeffcicient of determination. Measures the ratio of explained variance with regard to the total variance.

Note: "statsmodels.regression.linear_model.ols" does such evaluation (model.summary()).

#### Model diagnostic

In statistics, a Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other. First, the set of intervals for the quantiles is chosen. A point (x, y) on the plot corresponds to one of the quantiles of the second distribution (y-coordinate) plotted against the same quantile of the first distribution (x-coordinate). Thus the line is a parametric curve with the parameter which is the number of the interval for the quantile.

If the two distributions being compared are similar, the points in the Q–Q plot will approximately lie on the line y = x.

The process of building a Q-Q plot is as follows:

  - assign a quantile to each data point in the sample;
  - divide the reference normal distribution into an equal number of eqally sized quntiles;
  - plot (x, y) points on the chart, where x is the quantile value from the reference normal distribution and y is the alue from the sample.

![Q-Q Plot](img/q-q-plot.png)

### Multiple linear regression model

It is natural to have more than one predictor to predict a response. In such a case the response is determined by multifactors:

```
y = f(x_1, x_2, ..., x_n) = beta_0 + beta_1 * x_1 + ... + beta_n * x_n
```

#### F-test of significance

The F-test of the overall significance is a specific form of the F-test. It compares a model with no predictors to the model that you specify. A regression model that contains no predictors is also known as an intercept-only model.

The hypotheses for the F-test of the overall significance are as follows:

  - Null hypothesis: The fit of the intercept-only model and your model are equal;
  - Alternative hypothesis: The fit of the intercept-only model is significantly reduced compared to your model.

If the P value for the F-test of overall significance test is less than your significance level, you can reject the null-hypothesis and conclude that your model provides a better fit than the intercept-only model.

#### Multicollineatiry

Multicollinearity refers to a situation in which two or more explanatory variables in a multiple regression model are highly linearly related. In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data. Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set; it only affects calculations regarding individual predictors. That is, a multivariate regression model with collinear predictors can indicate how well the entire bundle of predictors predicts the outcome variable, but it may not give valid results about any individual predictor, or about which predictors are redundant with respect to others.

#### Model evaluation

RMSE:

```
RMSE = sqrt(SUM(err^2)/(n - k - 1))
```

where k is the number of predictors.

Adjusted R^2:

```
R^2_adjusted = 1 - (1 - R^2)(n - 1)/(n - k - 1)
```

##### Sharpe Ratio

Measures the excess return (or risk premium) per unit of deviation in an investment asset or a trading strategy, typically referred to as risk.

Excess returns are the return earned by a stock (or portfolio of stocks) and the risk free rate, which is usually estimated using the most recent short-term government treasury bill. For example, if a stock earns 15% in a year when the U.S. treasury bill earned 3%, the excess returns on the stock were 15%-3% = 12%.

Daily Sharpe ration is calculated as follows:

```
SR_daily = Mean(Return_a - Return_b)/Sqrt(Variance(Return_a - Return_b))
```

While the yearly Sharpe ratio is:

```
SR_yearly = Sqrt(252)*SR_daily
```

##### Maximum drawdown

Maximum drawdown is the maximum percentage decline in the strategy from the historical peak profit at each point in time.

At each point in time drawdown is calculated as follows:

```
drawdown = (maximum_wealth_by_this_time - current_wealth)/maximum_wealth_by_this_time
```

We then simply take a maximum of this value during model evaluation.

#### In-sample Goodness of fit measures

These measures help to asses the quality of model specification to select among different model designs.

Below are the in-sample measures.

##### R squared

R^2 measures the share of the variation in the outcome data explained by the model ans is computed as:

```
R^2 = 1 - RSS/TSS,

where:

RSS - the sum of squared residualts,
TSS - the sum of squared deviations of the outcome from its mean.
```

##### Adjusted R squared

The adjusted R^2 penalises R^2 for adding more variables, each additional variable needs to reduce RSS significantly to produce better goodness of fit.

##### Akaike Information Criterion (AIC)

Conceptually, AIC aims at finding the model that best describes an unknown data-generating process. AIC is to be maximized and is based on the maximum-likelihood estimate:

```
AIC = -2*log(L) + 2*k,

where:

L: the value fo the maximized likelihood;
k: number of model parameters.
```

##### Bayesian Information Criterion (BIC)

BIC imposes a higher penalty and might underfit comparatively to AIC, as it has a sample size as part of its terms. BIC tries to find the best model among the set of candidates:

```
BIC = -2*log(L) + log(N)*k,

where:

L: the value fo the maximized likelihood;
k: number of model parameters;
N: sample size.
```

#### Heteroskedasticity

Heteroskedasticity occurs when the residual variance is not constant but differs across observations. If the residual variance is positively correlated with an input variable, that is, when errors are larger for input values that are far from their mean, then Ordinary Least Squares standard error estimates will be too low, and, consequently, te t-statistic will be inflated leading to false discoveries of relationship where none actually exist.

Diagnostic includes visual inspection of the residuals: systematic patterns in the (supposedly random) residuals suggest statistical tests of the null hypothesis that errors are homoscedastic against various alternatives.

Heteroskedasticity tests: Breusch-Pagan and White tests.

Robust standard errors (sometimes called white standard errors) take heteroskedasticity into account when computing the error variance using a so-called 'sandwich estimator'.

#### Serial correlation

Serial correlation means that consecutive residuals produced by linear regression are correlated, which violates one of the Gauss-Markov Theorem's assumptions. Positive serial correlation implies that the standard errors are underestimated and the t-statistics will be inflated, leading to false discoveries if ignored.

The Durbin-Watson statistic diagnoses serial correlation. It tests the hypothesis that the OLS residuals are not autocorrelated against the alternative that they follow an autoregressive precess. The test statistic ranges from 0 to 4, and values near 2 indicate non-autocorrelation, lower values suggest positive, and higher values indicate negative autocorrelation. The exact threshold values depend on the number of parameters and observations.

#### Multicollinearity

Multicollinearity occurs when two or more independent variables are highly correlated.

It makes it difficult to determine which factor influence the dependent variable.

There is no formal or theory-based solution that corrects for multicollinearity. Instead, try to remve one or more of the correlated input variables, or increase the sample size.

### Stationarity and unit roots

Finding unit roots of a characteristic equasion of the process is required to deterime the order of integration of the pricess (I in ARIMA).

If the maximum value of the unit root is one - the first difference of the process will be stationary.

The defining characteristic of a unit-root non-stationary series is long memory: since curent values are the sum of past distrurbances, large innovations persist for much longer than for a mean-reverting, stationary series.

#### Unit root tests

Statistical unit root tests are a common way to determine objectively whether (additional) differencing is necessary.

The Augmented Dickey-Fuller (ADF) test evaluates the null hypothesis that a time series sample has unit root against the alternative of stationarity.

The ADF test statistics uses the sample coefficient Gamma, that, under the null hypothesis of unit-root non-stationarity equals zero, and is negative otherwise.

Python's statsmodels implements ADF as 'adfuller' under TSA package:

[statsmodels ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
