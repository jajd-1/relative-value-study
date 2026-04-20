import numpy as np 
import pandas as pd 
import data 
import signal_construction as sc  # pyright: ignore[reportMissingImports]
import backtesting as bt          # pyright: ignore[reportMissingImports]
import evaluation as ev           # pyright: ignore[reportMissingImports]


#set parameters
                                
tickers = [                     #tickers of assets under consideration; all possible pairs will be assessed for cointegration in data.py if find_pairs = True. Default list consists of 30 ETFs.
    "SPY", "IVV", "VOO", "VTI",
    "QQQ", "VGT", "XLK",
    "IWM", "IJR",
    "EFA", "IEFA", "VEA",
    "EEM", "VWO",
    "VNQ", "XLRE",
    "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU",
    "TLT", "IEF", "SHY", "TLH",
    "GLD", "IAU", "SLV"
]

start_date = pd.Timestamp('2010-01-01')                                        #starting date for price data (also taken to be the start of the formation period)
end_date = pd.Timestamp('2026-04-16')                                          #ending date for price data 
formation_window = 6                                                           #number of previous years used to compute OLS coefficients
zscore_window = 60                                                             #rolling window in calendar days for computing the z-score of the spread
formation_end = start_date + pd.DateOffset(years = formation_window)           #end of initial formation period, i.e. first date we compute the spread using formation_window years of previous price data    
trading_start = formation_end + pd.DateOffset(days = zscore_window)            #start date for making trades (must be at least zscore_window days after start of formation_end)
trading_end = end_date                                                         #end date for making trades


pvalue_threshold = 0.05                             #p-value for Engle-Granger test
pair = ['QQQ', 'XLU']                               #choose a single pair for signal construction and backtesting
entry_threshold = 2.0                               #value of z-score above/below zero to trigger entering a short/long position on the spread
exit_threshold = 0.5                                #value of z-score above/below zero to trigger exiting a short/long position on the spread
cost_bps = 3.0                                      #transaction cost in basis points
trading_days = 252                                  #trading days in a year
risk_free_rate = 0.01                               #risk-free interest rate used in computing Sharpe ratio


#select which parts of the program you want to run 

find_pairs = False       #set to false if you already know the pair you want to look at (i.e. you don't need to search for pairs in data.py)
build_strat = True       #set to false if you only want to search for pairs using data.py
plots1 = False           #set to false if you don't want to see plots of the proposed cointegrated pairs from data.py
plots2 = False           #set to false if you don't want to see plots of the spread, z-score and positions from signal_construction.py
plots3 = False           #set to false if you don't want to see plots of returns and drawdowns from backtesting.py


if find_pairs == True: 
    _, best_pairs, list_of_pairs = data.find_coint_pairs(tickers, start_date, formation_end, pvalue_threshold)
    print(best_pairs)
    if plots1 == True: 
        data.run_plots1(list_of_pairs, start_date, formation_end)


if build_strat == True:
    all_prices = data.load_prices(pair, start_date, end_date)
    formation_mask = (all_prices.index >= start_date) & (all_prices.index <= formation_end)
    formation_prices = all_prices.loc[formation_mask]
    series1 = formation_prices[pair[0]]
    series2 = formation_prices[pair[1]]
    print('p-value is: ', data.test_coint(series1, series2))       
    dfs = sc.build_signal_dataframe(all_prices, trading_start, trading_end, formation_window, zscore_window, entry_threshold, exit_threshold)
    signal_df = dfs[0]
    signal_df.to_csv('/Users/jonahduncan/Desktop/signal_df.csv')

    if plots2 == True:
        sc.run_plots2(signal_df, entry_threshold, exit_threshold) 

    backtest_df = bt.backtest_pair(signal_df, cost_bps)
    backtest_df.to_csv('/Users/jonahduncan/Desktop/backtest_df.csv')

    if plots3 == True:
        bt.run_plots3(df = backtest_df)

    net_return = backtest_df['net_return']
    print(ev.return_stats(net_return, trading_days, risk_free_rate, trading_start, trading_end))
    trades_df = ev.extract_trades(backtest_df, cost_bps)
    print(trades_df)
    trade_stats = ev.trade_stats(trades_df)
    print(trade_stats)


