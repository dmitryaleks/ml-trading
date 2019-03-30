import pandas as pd
import pandas_datareader.data as web
import scipy

import statsmodels.tsa.api as tsa
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('ggplot')

industrial_production = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
nasdaq = web.DataReader('NASDAQCOM', 'fred', '1990', '2017-12-31').squeeze().dropna()

components = tsa.seasonal_decompose(industrial_production, model='additive')

ts = (industrial_production.to_frame('Original')
      .assign(Trend=components.trend)
      .assign(Seasonality=components.seasonal)
      .assign(Residual=components.resid))

fig = ts.plot(subplots=True, figsize=(14, 8))

plt.show()
plt.savefig('img/ts-decomposition.png')
