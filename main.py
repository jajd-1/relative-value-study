import numpy as np 
import pandas as pd 
import data 
import signal_construction as sc    # pyright: ignore[reportMissingImports]
import backtesting as bt            # pyright: ignore[reportMissingImports]
import evaluation as ev             # pyright: ignore[reportMissingImports]
from itertools import combinations


#set parameters

tickers = ["QQQ", "XLK", "GLD"]     #tickers of assets under consideration; all possible pairs will be assessed for cointegration in data.py if find_pairs = True
                                
# tickers = [                     
#     "SPY", "IVV", "VOO", "VTI",
#     "QQQ", "VGT", "XLK",
#     "IWM", "IJR",
#     "EFA", "IEFA", "VEA",
#     "EEM", "VWO",
#     "VNQ", "XLRE",
#     "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU",
#     "TLT", "IEF", "SHY", "TLH",
#     "GLD", "IAU", "SLV"
# ]

start_date = pd.Timestamp('2010-01-01')                                        #starting date for price data (also taken to be the start of the formation period)
end_date = pd.Timestamp('2026-04-16')                                          #ending date for price data 
formation_window = 6                                                           #number of previous years used to compute OLS coefficients
zscore_window = 50                                                             #rolling window in trading days for computing the z-score of the spread
formation_end = start_date + pd.DateOffset(years = formation_window)           #end of initial formation period, i.e. first date we compute the spread using formation_window years of previous price data    


pvalue_threshold = 1.0                            #p-value for Engle-Granger test
pair = ['QQQ', 'XLK']                             #choose a single pair for signal construction and backtesting
benchmark = ['SPY']                               #choose a single benchmark asset to compare our strategy against
entry_threshold = 1.5                             #value of z-score above/below zero to trigger entering a short/long position on the spread
exit_threshold = 0.25                             #value of z-score above/below zero to trigger exiting a short/long position on the spread
cost_bps = 0.0                                    #transaction cost in basis points
trading_days = 252                                #trading days in a year
risk_free_rate = 0.0                              #risk-free interest rate used in computing Sharpe ratio

path = '/Users/jonahduncan/Desktop/'


#select which parts of the program you want to run 

find_pairs = False                  #set to false if you already know the pair you want to look at (i.e. you don't need to search for pairs in data.py)
build_strat = True                 #set to false if you only want to search for pairs using data.py
plots1 = False                      #set to false if you don't want to see plots of the proposed cointegrated pairs from data.py
plots2 = False                      #set to false if you don't want to see plots of the spread, z-score and positions from signal_construction.py
plots3 = False                      #set to false if you don't want to see plots of returns and drawdowns from backtesting.py
show_individual_trades = True


def main() -> None:
    if find_pairs == True: 
        _, best_pairs, list_of_pairs = data.find_coint_pairs(tickers, start_date, formation_end, pvalue_threshold)
        print(best_pairs)
        if plots1 == True: 
            data.run_plots1(list_of_pairs, start_date, formation_end)


    if build_strat == True:
        all_prices = data.load_prices(pair, start_date, end_date)

        formation_mask = (all_prices.index >= start_date) & (all_prices.index <= formation_end)
        formation_prices = all_prices.loc[formation_mask]
        
        spread_dates = all_prices.index[all_prices.index >= formation_end]      #dates for which we can compute the spread using formation_window years of data
        if len(spread_dates) <= zscore_window:
            raise ValueError("Not enough observations after formation_end to compute z-score window.")

        trading_start = spread_dates[zscore_window]     #start date for making trades (must be at least zscore_window trading days after start of formation_end)
        trading_end = all_prices.index[-1]

        series1 = formation_prices[pair[0]]
        series2 = formation_prices[pair[1]]
        print(f'p-value for cointegration between {pair[0]} and {pair[1]} is: {data.test_coint(series1, series2).round(4)}\n')       
        dfs = sc.build_signal_dataframe(all_prices, spread_dates[0], trading_start, trading_end, formation_window, zscore_window, entry_threshold, exit_threshold)
        signal_df = dfs[0]
        signal_df.to_csv(path + 'signal_df.csv')      #uncomment to save signal_df to path

        if plots2 == True:
            sc.run_plots2(signal_df, entry_threshold, exit_threshold) 

        backtest_df = bt.backtest_pair(signal_df, cost_bps)
        backtest_df.to_csv(path + 'backtest_df.csv')    #uncomment to save backtest_df to path

        if plots3 == True:
            bt.run_plots3(df = backtest_df)

        net_return = backtest_df['net_return']
        strategy_stats = ev.return_stats(net_return, trading_days, risk_free_rate, trading_start, trading_end)


        benchmark_prices = data.load_prices(benchmark, trading_start, trading_end).iloc[:, 0]
        benchmark_return = benchmark_prices.pct_change().fillna(0.0)
        benchmark_return = benchmark_return.reindex(backtest_df.index).fillna(0.0)
        benchmark_stats = ev.return_stats(benchmark_return, trading_days, risk_free_rate, trading_start, trading_end)


        stats_df = pd.concat([strategy_stats.rename('Strategy'), benchmark_stats.rename('Benchmark')], axis = 1).round(4)
        print(f'Performance stats:\n\n{stats_df}\n')


        beta_to_benchmark = net_return.cov(benchmark_return) / benchmark_return.var()
        corr_with_benchmark = net_return.cov(benchmark_return) / (benchmark_return.std() * net_return.std())

        print(f'Beta relative to the benchmark: {beta_to_benchmark.round(4)}')
        print(f'Correlation with the benchmark: {corr_with_benchmark.round(4)}\n')

        trades_df = ev.extract_trades(backtest_df, cost_bps)

        if show_individual_trades == True: 
            print(f'List of trades:\n\n{trades_df}\n')

        closed_trade_stats = ev.closed_trade_stats(trades_df).round(4)
        print(f'Trade stats:\n\n{closed_trade_stats}\n')

        # return {
        #     "pair": pair,
        #     "total_return": strategy_stats["total_return"],
        #     "annualised_return": strategy_stats["annualised_return"],
        #     "annualised_sharpe": strategy_stats["annualised_sharpe"],
        #     "max_drawdown": strategy_stats["max_drawdown"],
        #     "beta_to_benchmark": beta_to_benchmark,
        #     "correlation_with_benchmark": corr_with_benchmark,
        # }


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     pairs = list(combinations(tickers, 2))

#     results = []

#     for ticker_pair in pairs:
#         pair = ticker_pair
#         result = main()
#         results.append(result)

#     results_df = pd.DataFrame(results)
#     results_df = results_df.sort_values('total_return', ascending = False)
#     results_df.to_csv(path + 'pair_finder.csv')
#     print(results_df)




