# Unsupervised Learning Trading

What this program does: 
-Download/load S&P 500 stock price data.

-calculate the different features and indicator on each stock

-Aggregate on monthly level and filter top 150 most liquid stocks

-Calculate Fama-French Factors and calculate Rolling factor betas

-For each month fit a k-means clustering algoritm to group similar assets based on their features

-For each month select assets based on the cluster and from portofolio based on effecient frontier max sharpe ratio optimization.

-Visualize portofolio returns and compare to S&P 500 returns.

# Packages Needed:
-pandas, numpy, matplotlib, statmodels, pandas_datareader, datetime, yfinance, sklearn, PyPortfolioOpt

            -> pip install pandas numpy matplotlib statmodels pandas_datareader datetime yfinance sklearn PyPortfolioOpt


## calculate the different features and indicator on each stock
Garman-Klass Volatility
RSI
Bollinger Bands
ATR
MACD
Dollar Volume

        Formula:
                                        (In(High) - In(Low))^2
            -> Garman-Klass Volatility = ---------------------- (2In(2) - 1)(In(AdjClose - In(Open)))^2
                                                    2
    
