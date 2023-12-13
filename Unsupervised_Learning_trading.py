from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt 
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')


# packages to install
# pip install statsmodels pandas_datareader yfinance pandas_ta

# Read the S&P 500 stocks from wikipedia
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
# # Clean the symbol column data 
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

# # Grab all the stock in symbol column
symbols_list = sp500['Symbol'].unique().tolist()

# # download  data from now and 8 years back
end_date = '2023-12-10'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

# # Download data from yfinance
# # Clean the data cause it seems a lil bit inefficient using stack()
df = yf.download(tickers=symbols_list, 
                 start = start_date,
                 end = end_date).stack()

# # change the index format
df.index.names = ['date', 'ticker']

# # fix the columns to lowercase
df.columns = df.columns.str.lower()

# # calculate the different features and indicator on each stock
# # Garman-Klass volatility
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

# # Calculate RSI
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
 
# # calculate Bollinger Bands
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])

# # calculate atr
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

# #  Compute MACD
def compute_macd(close):
    macd = pandas_ta.macd(close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

# # calculate the dollar volume
df['dollar_volume'] = (df['adj close'] * df['volume'])/1e6

# # Aggregate to monthly level and filter top 150 most liquid stock for each month
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
data = pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
           df.unstack()[last_cols].resample('M').last().stack('ticker')],
            axis=1).dropna()

# # Calculate 5-year rolling average of dollar_volume for each stocks before filtering
data['dollar_volume'] = (data['dollar_volume'].unstack('ticker').rolling(5*12).mean().stack())
# Rank by dollar volume
data['dollar_volume_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
# filter by 150
data = data[data['dollar_volume_rank']<150].drop(['dollar_volume', 'dollar_volume_rank'], axis=1)

# Calculate monthly returns for different time horizons as features.
def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                       upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

# Calculate Fama-French Factors and calculate Rolling factor betas
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                             'famafrench',
                             start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data = factor_data.join(data['return_1m']).sort_index()

# Filter out by the 10 months
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker')].isin(valid_stocks.index)

# Calculate the rolling factor betas
betas = (factor_data.groupby(level=1,
                            group_keys=False)
        .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                    exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                    window=min(24, x.shape[0]),
                                    min_nobs=len(x.columns)+1)
        .fit(params_only=True)
        .params
        .drop('const', axis=1)))

data = (data.join(betas.groupby('ticker').shift()))

factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = (data.join(betas.groupby('ticker').shift()))
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()

# fit a k-means clustering algoritm to group similar assets based on their features
from sklearn import KMeans

data = data.drop('cluster', axis=1)
def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

def plot_clusters(data):
    cluster_0 = data[data['cluster']==0]
    cluster_0 = data[data['cluster']==1]
    cluster_0 = data[data['cluster']==2]
    cluster_0 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6] , color = 'red', label ='cluster 0')
    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6] , color = 'red', label ='cluster 0')
    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6] , color = 'red', label ='cluster 0')
    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6] , color = 'red', label ='cluster 0')
    plt.legend()
    plt.show()
    return
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().tolist():
    g = data.xs(i, level=0)
    plt.title(f'Date {i}')
    plot_clusters(g)

target_rsi_values = [30, 45, 55, 70]
initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 6] = target_rsi_values

# For each month select assets based on the cluster and form a portfolio based on effecient frontier 
# max sharpe ratio optimization
filtered_df = data[data['cluster']==3].copy()
filtered_df.index = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index+pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
fixed_dates

# define the portofilio optimization function
from pypfopt.efficient_frontier import efficient_frontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weight(prices):
    returns = expected_returns.mean_historical_return(price=prices, 
                                                      frequency=252)
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    ef = efficient_frontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(0, .1),
                           solver= 'SCS')
    weights = ef.max_sharpe()
    return ef.clean_weights()

# Download Fresh daily prices data only for short listed stocks.
stocks = data.index.get_level_values('ticker').unique().tolist()
new_df = yf.download(ticker=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])
