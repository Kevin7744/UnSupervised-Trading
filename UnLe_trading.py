from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as pyplot
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

sp500

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
# it will take sometime to download


# # after download
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

print(data)