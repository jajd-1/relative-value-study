import numpy as np 
import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 
from itertools import combinations
from statsmodels.tsa.stattools import coint 


def load_prices(tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:  # can also enter date as string of the form 'yyyy-mm-dd'
    """Load adjusted prices from yfinance"""
    data = yf.download(tickers, start = start_date , end = end_date, auto_adjust = False, progress = False) 

    if 'Adj Close' in data: 
        prices = data['Adj Close'].copy()
    else:
        prices = data['Close'].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()      #if there is only one ticker, prices will be a series rather than dataframe, so convert it

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep = 'first')]   #if there are any duplicate rows, keep only the first
    prices = prices.dropna(how = 'any')   #drop rows with any missing values

    return prices 


def generate_pairs(tickers: list[str]) -> list[list[str]]:
    """Returns all possible pairs of tickers from the list of tickers"""
    return list(combinations(tickers, 2))


def test_coint(series1: pd.Series, series2: pd.Series) -> float:
    """Uses the Engle-Granger method to test for cointegration between series1 and series2, with a low p-value suggesting cointegration"""
    _, pvalue, _ = coint(series1, series2, trend = 'c')     #change trend to 'ct' to allow constant + linear trend 
    return pvalue 





def find_coint_pairs(tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp, pvalue_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, list[list]]:
    """Tests all pairs for cointegration and returns 3 things: a dataframe consisting of p-values for all pairs, a dataframe consisting of p-values below the threshold, and
     a list of pairs for which the corresponding p-value is below the threshold."""
    results = []
    pairs = generate_pairs(tickers)
    prices = load_prices(tickers, start_date, end_date)

    for ticker1, ticker2 in pairs:
        series1 = prices[ticker1]
        series2 = prices[ticker2]

        try:
            pvalue = test_coint(series1, series2)
            results.append({
                'Ticker 1': ticker1,
                'Ticker 2': ticker2,
                'p-value': pvalue 
            })
        
        except Exception as e:
            print(f'Error testing {ticker1}-{ticker2}: {e}')

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value')

    best = results_df[(0 < results_df['p-value']) & (results_df['p-value'] < pvalue_threshold)].reset_index(drop = True)
    best_pairs = best[['Ticker 1', 'Ticker 2']].values.tolist()

    return results_df, best, best_pairs


# --------------- Visual inspection ----------------

def plot_raw_prices(prices: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize = (12, 6))
    prices.plot(ax = ax)
    plt.title(f'Adjusted Close Prices for {prices.columns[0]} and {prices.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc = 'best')
    plt.tight_layout()
    filename = f'{prices.columns[0]}{prices.columns[1]}1.png'
    plt.savefig(f"/Users/jonahduncan/Desktop/python_work/quant_projects/pair_trading/images/{filename}", dpi = 300, bbox_inches = 'tight')


def plot_normalised_prices(prices: pd.DataFrame) -> None:
    normalised = prices/prices.iloc[0]
    fig, ax = plt.subplots(figsize = (12,6))
    normalised.plot(ax = ax)
    plt.title(f'Normalised Adjusted Close Prices (start = 1) for {prices.columns[0]} and {prices.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Normalised Price')
    plt.legend(loc = 'best')
    plt.tight_layout()
    filename = f'{prices.columns[0]}{prices.columns[1]}2.png'
    plt.savefig(f"/Users/jonahduncan/Desktop/python_work/quant_projects/pair_trading/images/{filename}", dpi = 300, bbox_inches = 'tight')
    

def plot_normalised_price_ratios(prices: pd.DataFrame) -> None:
    normalised = prices/prices.iloc[0]
    ratio = normalised.iloc[:, 0]/normalised.iloc[:, 1]
    fig, ax = plt.subplots(figsize = (12, 6))
    ratio.plot(ax = ax)
    plt.title(f'{prices.columns[0]}/{prices.columns[1]} Normalised Price Ratio')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend(loc = 'best')
    plt.tight_layout()
    filename = f'{prices.columns[0]}{prices.columns[1]}3.png'
    plt.savefig(f"/Users/jonahduncan/Desktop/python_work/quant_projects/pair_trading/images/{filename}", dpi = 300, bbox_inches = 'tight')
    

def plot_normalised_price_scatter(prices: pd.DataFrame) -> None:
    normalised = prices/prices.iloc[0]
    plt.figure(figsize = (6,6))
    plt.scatter(normalised.iloc[:, 0], normalised.iloc[:, 1], alpha = 0.5, marker = '.')
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.title(f'{prices.columns[1]} vs {prices.columns[0]} Normalised Price Scatter')
    plt.tight_layout()
    filename = f'{prices.columns[0]}{prices.columns[1]}4.png'
    plt.savefig(f"/Users/jonahduncan/Desktop/python_work/quant_projects/pair_trading/images/{filename}", dpi = 300, bbox_inches = 'tight')


def run_plots1(pairs: list[list[str]], start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    for pair in pairs: 
        prices = load_prices(pair, start_date, end_date)
        plot_raw_prices(prices)
        plot_normalised_prices(prices)
        plot_normalised_price_scatter(prices)
        plt.show()
