import numpy as np 
import pandas as pd 
import data 
import matplotlib.pyplot as plt
import seaborn as sns
import signal_construction as sc    # pyright: ignore[reportMissingImports]
import backtesting as bt            # pyright: ignore[reportMissingImports]
import evaluation as ev             # pyright: ignore[reportMissingImports]
from itertools import combinations
from pathlib import Path

#set parameters

tickers = ["QQQ", "XLK", "GLD"]     #tickers of assets under consideration; all possible pairs will be assessed for cointegration in data.py if find_pairs = True

start_date = pd.Timestamp('2010-01-01')                                        #starting date for price data (also taken to be the start of the formation period)
end_date = pd.Timestamp('2026-04-16')                                          #ending date for price data 
formation_window = 6                                                           #number of previous years used to compute OLS coefficients
zscore_window = 50                                                             #rolling window in trading days for computing the z-score of the spread
formation_end = start_date + pd.DateOffset(years = formation_window)           #end of initial formation period, i.e. first date we compute the spread using formation_window years of previous price data    

pvalue_threshold = 1.0                            #p-value for Engle-Granger test
pair = ['IEFA', 'EEM']                             #choose a single pair for signal construction and backtesting
benchmark = ['SPY']                               #choose a single benchmark asset to compare our strategy against
entry_threshold = 1.5                             #value of z-score above/below zero to trigger entering a short/long position on the spread
exit_threshold = 0.25                             #value of z-score above/below zero to trigger exiting a short/long position on the spread
cost_bps = 0.0                                    #transaction cost in basis points
trading_days = 252                                #trading days in a year
risk_free_rate = 0.0                              #risk-free interest rate used in computing Sharpe ratio

#select which parts of the program you want to run 

find_pairs = True                  #set to false if you don't want to run cointegration screening 
build_strat = True                  #set to false if you only want to search for pairs using data.py
plots1 = True                      #set to false if you don't want to see plots from the cointegration screening
plots2 = True                      #set to false if you don't want to see plots of the spread, z-score and positions from signal_construction.py
plots3 = True                      #set to false if you don't want to see plots of returns and drawdowns from backtesting.py
sensitivity_plots = True
save_plots1 = True
save_plots2 = True
save_plots3 = True
save_sensitivity_plots = True
show_individual_trades = False
save_csv = True

script_dir = Path(__file__).resolve().parent
images_dir = script_dir/"images"
csv_dir = script_dir/"csv-outputs"


def load() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    all_prices = data.load_prices(pair, start_date, end_date)

    formation_mask = (all_prices.index >= start_date) & (all_prices.index <= formation_end)
    formation_prices = all_prices.loc[formation_mask]

    spread_dates = all_prices.index[all_prices.index >= formation_end]      #dates for which we can compute the spread using formation_window years of data
    if len(spread_dates) <= zscore_window:
        raise ValueError("Not enough observations after formation_end to compute z-score window.")

    trading_start = spread_dates[zscore_window]     #start date for making trades (must be at least zscore_window trading days after start of formation_end)
    trading_end = all_prices.index[-1]

    dfs = sc.build_signal_dataframe(all_prices, spread_dates[0], trading_start, trading_end, formation_window, zscore_window, entry_threshold, exit_threshold)
    signal_df = dfs[0]

    if save_csv == True: 
            filename = f'{pair[0]}{pair[1]}signal_df.csv'
            signal_df.to_csv(csv_dir/filename)

    return formation_prices, trading_start, trading_end, signal_df 

def main() -> None:
    if find_pairs == True: 
        _, best_pairs, list_of_pairs = data.find_coint_pairs(tickers, start_date, formation_end, pvalue_threshold)
        print(best_pairs)
        if plots1 == True: 
            data.run_plots1(list_of_pairs, start_date, formation_end, save_plots1)

    if build_strat == True:
        formation_prices, trading_start, trading_end, signal_df = load()

        series1 = formation_prices[pair[0]]
        series2 = formation_prices[pair[1]]
        print(f'p-value for cointegration between {pair[0]} and {pair[1]} is: {data.test_coint(series1, series2).round(4)}\n')       

        if plots2 == True:
            sc.run_plots2(signal_df, entry_threshold, exit_threshold, save_plots2) 

        backtest_df = bt.backtest_pair(signal_df, cost_bps)
        if save_csv == True: 
            filename = f'{pair[0]}{pair[1]}backtest_df.csv'
            backtest_df.to_csv(csv_dir/filename) 

        if plots3 == True:
            bt.run_plots3(backtest_df, save_plots3)

        net_return = backtest_df['net_return']
        strategy_stats = ev.return_stats(net_return, trading_days, risk_free_rate, trading_start, trading_end)

        benchmark_prices = data.load_prices(benchmark, trading_start, trading_end).iloc[:, 0]
        benchmark_return = benchmark_prices.pct_change().fillna(0.0)
        benchmark_return = benchmark_return.reindex(backtest_df.index).fillna(0.0)
        benchmark_stats = ev.return_stats(benchmark_return, trading_days, risk_free_rate, trading_start, trading_end)

        strategy_stats_formatted = strategy_stats.copy().astype(object)
        benchmark_stats_formatted = benchmark_stats.copy().astype(object)
        for row in ['Total return', 'Annualised return', 'Annualised volatility', 'Maximum drawdown']:
            strategy_stats_formatted.loc[row] = f"{strategy_stats.loc[row]*100:.2f}%"
            benchmark_stats_formatted.loc[row] = f"{benchmark_stats.loc[row]*100:.2f}%"

        strategy_stats_formatted.loc['Annualised Sharpe ratio'] = f"{strategy_stats.loc['Annualised Sharpe ratio']:.2f}"
        benchmark_stats_formatted.loc['Annualised Sharpe ratio'] = f"{benchmark_stats.loc['Annualised Sharpe ratio']:.2f}"

        stats_df = pd.concat([strategy_stats_formatted.rename('Strategy'), benchmark_stats_formatted.rename('Benchmark')], axis = 1)
        print(f'Performance stats:\n\n{stats_df.to_markdown()}\n')

        beta_to_benchmark = net_return.cov(benchmark_return) / benchmark_return.var()
        corr_with_benchmark = net_return.cov(benchmark_return) / (benchmark_return.std() * net_return.std())

        print(f'Beta relative to the benchmark: {beta_to_benchmark:.4f}')
        print(f'Correlation with the benchmark: {corr_with_benchmark:.4f}\n')

        trades_df = ev.extract_trades(backtest_df, cost_bps)

        if show_individual_trades == True: 
            print(f'List of trades:\n\n{trades_df}\n')

        closed_trade_stats = ev.closed_trade_stats(trades_df)
        closed_trade_stats_formatted = closed_trade_stats.copy().astype(object)

        for row_name in closed_trade_stats_formatted.index:
            if row_name in ['Hit rate', 'Avg trade return', 'Median trade return', 'Best trade return', 'Worst trade return', 'Avg win', 'Avg loss']:
                closed_trade_stats_formatted.loc[row_name] = f"{closed_trade_stats.loc[row_name]*100:.2f}%"
            else:
                closed_trade_stats_formatted.loc[row_name] = f"{closed_trade_stats.loc[row_name]:.2f}"

        print(f'Trade stats:\n\n{closed_trade_stats_formatted.to_markdown()}\n')

        return signal_df


def plot_cost_sensitivity(signal_df, pair, trading_days, risk_free_rate, trading_start, trading_end, save_plots = False, cost_grid = None):
    if cost_grid is None:
        cost_grid = np.arange(0.0, 6.05, 0.05)   

    total_returns = []
    hit_rates = []

    for cost in cost_grid:
        backtest_df = bt.backtest_pair(signal_df, cost)
        net_return = backtest_df["net_return"]

        strategy_stats = ev.return_stats(net_return, trading_days, risk_free_rate, trading_start, trading_end)

        total_returns.append(strategy_stats["Total return"])
        trades_df = ev.extract_trades(backtest_df, cost)
        trade_stats = ev.closed_trade_stats(trades_df)

        if isinstance(trade_stats, pd.Series) and "Hit rate" in trade_stats.index:
            hit_rates.append(trade_stats["Hit rate"])
        else:
            hit_rates.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(cost_grid, total_returns, color="tab:blue", label="Total return")
    ax1.set_xlabel("Transaction cost (bps)")
    ax1.set_ylabel("Total return", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    plt.axhline(0, linestyle = '--', color = 'black', alpha = 0.1)

    ax2 = ax1.twinx()
    ax2.plot(cost_grid, hit_rates, color="tab:red", label="Hit rate")
    ax2.set_ylabel("Hit rate", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Total return and hit rate vs transaction cost for {pair[0]}/{pair[1]}")
    fig.tight_layout()

    if save_plots == True:
        filename = f'{pair[0]}{pair[1]}returnhitvscost.png'
        plt.savefig(images_dir/filename, dpi = 300, bbox_inches = 'tight')

    plt.show()


def plot_sharpe_heatmap(signal_df, pair, trading_days, trading_start, trading_end, save_plots = False, cost_grid = None, rf_grid = None):
    if cost_grid is None:
        cost_grid = np.arange(0.0, 5.05, 0.05)   
    if rf_grid is None:
        rf_grid = np.arange(0.0, 0.0505, 0.0005) 

    sharpe_surface = np.empty((len(rf_grid), len(cost_grid)))

    for i, rf in enumerate(rf_grid):
        for j, cost in enumerate(cost_grid):
            backtest_df = bt.backtest_pair(signal_df, cost)
            net_return = backtest_df["net_return"]
            strategy_stats = ev.return_stats(net_return, trading_days, rf, trading_start, trading_end)
            sharpe_surface[i, j] = strategy_stats["Annualised Sharpe ratio"]

    sharpe_df = pd.DataFrame(sharpe_surface, index=[f"{100*x:.1f}%" for x in rf_grid], columns=np.round(cost_grid, 2))

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(sharpe_df, cmap="viridis", center=0, annot=False, cbar_kws={"label": "Annualised Sharpe ratio"}, ax = ax)

    ax.invert_yaxis()
    
    X, Y = np.meshgrid(np.arange(len(cost_grid)) + 0.5, np.arange(len(rf_grid)) + 0.5)
    contour0 = ax.contour(X, Y, sharpe_surface, levels=[0], colors="white", linewidths=1)

    max_sharpe = sharpe_surface[0, 0]
    half_sharpe = max_sharpe / 2
    contour_half = ax.contour(X, Y, sharpe_surface, levels=[half_sharpe], colors="white", linewidths=1)
    
    ax.clabel(contour0, fmt={0: "Sharpe = 0"}, inline=True, fontsize=7)
    ax.clabel(contour_half, fmt={half_sharpe: f"Sharpe = {half_sharpe:.2f}"}, inline=True, fontsize=7)

    x_step = 10
    y_step = 10

    ax.set_xticks(np.arange(0, len(cost_grid), x_step) + 0.5)
    ax.set_xticklabels([f"{cost_grid[i]:.2f}" for i in range(0, len(cost_grid), x_step)], rotation=45)
    ax.set_yticks(np.arange(0, len(rf_grid), y_step) + 0.5)
    ax.set_yticklabels([f"{100*rf_grid[i]:.2f}%" for i in range(0, len(rf_grid), y_step)])

    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Risk-free rate")
    ax.set_title(f"Sharpe ratio heat map for {pair[0]}/{pair[1]}")
    plt.tight_layout()

    if save_plots == True:
        filename = f'{pair[0]}{pair[1]}sharpeheatmap.png'
        plt.savefig(images_dir/filename, dpi = 300, bbox_inches = 'tight')

    plt.show()


if __name__ == "__main__":
    main()

if sensitivity_plots == True:
    _, trading_start, trading_end, signal_df = load()
    plot_cost_sensitivity(signal_df, pair, trading_days, risk_free_rate, trading_start, trading_end, save_sensitivity_plots)
    plot_sharpe_heatmap(signal_df, pair, trading_days, trading_start, trading_end, save_sensitivity_plots)

