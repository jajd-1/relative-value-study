import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 

# ----------- Define a function to load prices from yfinance --------------- 

def load_prices(tickers, start = '2016-01-01', end = None):
    data = yf.download(tickers, start = start , end = end, auto_adjust = False, progress = False) 

    if "Adj Close" in data: 
        prices = data["Adj Close"].copy()
    else:
        prices = data["Close"].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()      #if there is only one ticker, prices will be a series rather than dataframe, so we convert it for compatability later

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep = 'first')]   #if there are any duplicate rows, keep only the first
    prices = prices.dropna(how = 'any')   #drop rows with any missing values

    return prices 


# ----------- Define some ETF pairs and load their prices -------------

tickers = []

tickers1 = ['SPY', 'IVV'] 
#SPY = State Street SPDR S&P 500 ETF Trust, IVV = iShares Core S&P 500 ETF
tickers += tickers1
prices1 = load_prices(tickers1)

tickers2 = ['GLD', 'IAU']
#GLD = SPDR Gold Shares ETF, IAU = iShares Gold Trust ETF
tickers += tickers2
prices2 = load_prices(tickers2)

tickers3 = ['TLT', 'TLH']
#TLT = iShares 20+ Year Treasury Bond ETF, IEF = iShares 10-20 Year Treasury Bond ETF
tickers += tickers3
prices3 = load_prices(tickers3) 

tickers4 = ['XLF', 'KBE']
#XLF = State Street Financial Select Sector SPDR ETF, KBE = State Street SPDR S&P Bank ETF 
tickers += tickers4
prices4 = load_prices(tickers4) 



# ---------- Visual inspection to determine plausible candidate pairs and catch data issues -------------


# def plot_raw_prices(prices):
#     prices.plot(figsize = (12,6))
#     plt.title('Adjusted Close Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend(loc = 'best')
#     plt.tight_layout
#     plt.show()

# plot_raw_prices(prices1)
# plot_raw_prices(prices2)
# plot_raw_prices(prices3)
# plot_raw_prices(prices4) 

def plot_normalised_prices(prices):
    normalised = prices/prices.iloc[0]
    normalised.plot(figsize = (12,6))
    plt.title('Normalised Adjusted Close Prices (start = 1)')
    plt.xlabel('Date')
    plt.ylabel('Normalised Price')
    plt.legend(loc = 'best')
    plt.tight_layout
    plt.show()

plot_normalised_prices(prices1)
plot_normalised_prices(prices2)
plot_normalised_prices(prices3)
plot_normalised_prices(prices4) 

"""Normalised prices1 for SPY and IVV are essentially identical, so there might not be much to exploit here. 
Normalised prices2 for GLD and IAU are almost identical, perhaps with a very small spread at some points which could be explored. 
Normalised prices3 for TLT and TLH follow very similar trends and have long periods of diversion and convergence, suggesting a pair trade might be suitable (but maybe over a longer horizon).
Normalised prices4 follow the same trend but have diverged in recent years. This could suggest different betas to the same sector, but should look at the ratio to see if there is still an 
opportunity for a pairs trade."""

def plot_normalised_price_ratios(prices):
    normalised = prices/prices.iloc[0]
    ratio = normalised.iloc[:, 0]/normalised.iloc[:, 1]
    ratio.plot(figsize=(12, 6))
    plt.title(f"{prices.columns[0]}/{prices.columns[1]} Price Ratio")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.show()

plot_normalised_price_ratios(prices4)

"""We see that since 2023, the ratio of normalised prices of XLF and KBE fluctuate around a mean of 0.69, so this is certainly worth looking at."""


def plot_normalised_price_scatter(prices):
    normalised = prices/prices.iloc[0]
    plt.figure(figsize = (6,6))
    plt.scatter(normalised.iloc[:, 0], normalised.iloc[:, 1], alpha = 0.5, marker = ".")
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.title(f"{prices.columns[1]} vs {prices.columns[0]} Price Scatter")
    plt.tight_layout()
    plt.show()

plot_normalised_price_scatter(prices1)
plot_normalised_price_scatter(prices2) 
plot_normalised_price_scatter(prices3) 
plot_normalised_price_scatter(prices4) 

"""In the first two instances we have an almost perfectly linear relationship. For the third instance there is certainly a linear relationship although it is more diffused. 
In the final instance, there is a very rough overall linear relationship, but with many clouds (which themselves are rather linear)."""








