import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import data 
import signal_construction as sc # pyright: ignore[reportMissingImports]

def backtest_pair(signal_df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:   
    """Computes (amongst other quantities) cumulative returns and drawdowns, and builds a dataframe with these quantities"""
    y = signal_df.iloc[:, 0]
    x = signal_df.iloc[:, 1]
    holding_position = signal_df['holding_position']  
    trade_made = signal_df['trade_made']
    continuing_position = (holding_position != 0) & (holding_position.shift(-1).fillna(0) != 0) & (trade_made == 0)
    signal_beta = signal_df['betas']
    held_beta = signal_beta.shift(1)

    dy = y.diff().fillna(0.0)                #change in price of y from previous day's close price
    dx = x.diff().fillna(0.0)                #change in price of x from previous day's close price
    pnl_per_unit_spread = dy - held_beta * dx     #pnl of one long-spread unit (i.e. long one share of y and short beta shares of x)

    gross_exposure_per_unit = (y.shift(1).abs() + abs(held_beta * x.shift(1))).fillna(0.0)      #gross exposure of one unit using previous day's close price

    active = (gross_exposure_per_unit > 0) & (holding_position != 0)                       #selects days with an open position

    gross_return_on_long_spread = pd.Series(0.0, index = signal_df.index)                                                     #initialise gross returns to zero
    gross_return_on_long_spread.loc[active] = pnl_per_unit_spread.loc[active] / gross_exposure_per_unit.loc[active]           #gross return on today's position if a long-spread position is held
    gross_return = holding_position.astype(float) * gross_return_on_long_spread                                                             #gross return on today's position 

    cost_rate = cost_bps / 10000.0
    entry_exit_cost = trade_made * cost_rate                    #computes transaction cost on returns if a position has been entered or exited
    beta_turnover = abs(signal_beta - held_beta) * x / gross_exposure_per_unit
    rebalancing_cost = continuing_position.astype(float) * cost_rate * beta_turnover
    total_cost = entry_exit_cost + rebalancing_cost

    net_return = gross_return - total_cost                          #net return equals gross return except on days after a position has been entered or exited, where transaction costs are taken on
    cumulative_net_return = (1.0 + net_return).cumprod()               #computes cumulative net return between start date and end date
    cumulative_gross_return = (1.0 + gross_return).cumprod()           #computes cumulative gross return between start date and end date, i.e. cumulative return if there are no transaction costs

    running_max = cumulative_net_return.cummax()
    drawdown = cumulative_net_return / running_max - 1.0

    backtest_df = signal_df.copy()
    backtest_df['dy'] = dy
    backtest_df['dx'] = dx
    backtest_df['holding_position'] = holding_position
    backtest_df['pnl_per_unit_spread'] = pnl_per_unit_spread
    backtest_df['gross_exposure_per_unit'] = gross_exposure_per_unit
    backtest_df['gross_return_on_long_spread'] = gross_return_on_long_spread
    backtest_df['gross_return'] = gross_return
    backtest_df['trade_made'] = trade_made
    backtest_df['total_cost'] = total_cost
    backtest_df['net_return'] = net_return
    backtest_df['cumulative_net_return'] = cumulative_net_return
    backtest_df['cumulative_gross_return'] = cumulative_gross_return 
    backtest_df['drawdown'] = drawdown

    return backtest_df


# -------------- Visual inspection ---------------


def plot_cumulative_return(backtest_df: pd.DataFrame) -> None:
    backtest_df['cumulative_net_return'].plot(figsize = (12, 5))
    plt.title(f'Cumulative net return (growth of $1) for pairs trading strategy with {backtest_df.columns[0]} and {backtest_df.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.tight_layout()
    plt.show()


def plot_drawdown(backtest_df: pd.DataFrame) -> None:
    backtest_df['drawdown'].plot(figsize = (12, 4))
    plt.title(f'Drawdown for pairs trading strategy with {backtest_df.columns[0]} and {backtest_df.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.tight_layout()
    plt.show()


def run_plots3(df: pd.DataFrame) -> None:
    plot_cumulative_return(df)
    plot_drawdown(df)





