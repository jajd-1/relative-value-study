import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import data


def estimate_hedge_ratio(prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[float,float]:
    """Regress y (first column) on x (second column) with intercept and return the regression coefficients"""
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    prices = prices.loc[mask]

    y = prices.iloc[:, 0]
    x = prices.iloc[:, 1]

    X = sm.add_constant(x)      #add column of 1s to x to include intercept in OLS
    model = sm.OLS(y,X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    return alpha, beta


def construct_spread(prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, formation_window: int) -> tuple[pd.Series, pd.Series]:
    """For each year between first_year and last_year, we return the OLS residual (aka spread) with alpha and beta equal to the regression coefficients found from the previous formation_window years"""
    spread = pd.Series(np.nan, index = prices.index)
    betas = pd.Series(np.nan, index = prices.index)
    y = prices.iloc[:, 0]
    x = prices.iloc[:, 1]

    for day in prices.loc[start_date:end_date].index:
        alpha, beta = estimate_hedge_ratio(prices, day - pd.DateOffset(years = formation_window), day - pd.DateOffset(days = 1))
        mask = (prices.index == day)
        spread.loc[mask] = y.loc[mask] - alpha - beta * x.loc[mask]
        betas.loc[mask] = beta 

    return spread, betas 


def compute_zscore(spread: float, zscore_window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Computes the z-score of the spread on day T using the rolling mean and rolling standard deviation from the previous window number of days"""
    
    rolling_mean = spread.rolling(zscore_window).mean().shift(1)
    rolling_std = spread.rolling(zscore_window).std().shift(1)

    zscore = (spread - rolling_mean) / rolling_std 

    return rolling_mean, rolling_std, zscore 


def generate_positions(zscore: pd.Series, start_date: pd.Timestamp, entry_threshold: float, exit_threshold: float) -> tuple[pd.Series, pd.Series]:
    """
    Holding position convention: +1 = holding long spread, -1 = holding short spread, 0 = flat

    Rules:  if flat and zscore < -entry_threshold on day t: make trade on day t and enter long spread (reflected in holding position from day t+1)
            if flat and zscore >  entry_threshold on day t: make trade on day t and enter short spread (reflected in holding position from day t+1)
            if long spread and zscore > -exit_threshold on day t: make trade on day t and exit long spread (reflected in holding position from day t+1)
            if short spread and zscore < exit_threshold on day t: make trade on day t and exit short spread (reflected in holding position from day t+1)
    """

    holding_position = pd.Series(0, index = zscore.index, dtype = int)
    trade_made = pd.Series(0, index = zscore.index, dtype = int)
    current_holding_position = 0

    mask = (zscore.index >= start_date)

    for t in range(len(zscore)):    
        if mask[t]:
            z = zscore.iloc[t]

            if current_holding_position == 0:
                if z < -entry_threshold:
                    trade_made.iloc[t] = 1 
                    if t != len(zscore) - 1:
                        holding_position.iloc[t+1] = 1
                    current_holding_position = 1    
                elif z > entry_threshold:
                    trade_made.iloc[t] = 1
                    if t != len(zscore) - 1:
                        holding_position.iloc[t+1] = -1
                    current_holding_position = -1   
            
            elif current_holding_position == 1:
                if z > -exit_threshold:
                    if z > entry_threshold:
                        trade_made.iloc[t] = 2
                        if t != len(zscore) - 1:
                            holding_position.iloc[t+1] = -1
                        current_holding_position = -1
                    else:
                        trade_made.iloc[t] = 1
                        current_holding_position = 0    
                elif t != len(zscore) - 1:
                    holding_position.iloc[t+1] = 1
            
            elif current_holding_position == -1:
                if z < exit_threshold:
                    if z < -entry_threshold:
                        trade_made.iloc[t] = 2
                        if t != len(zscore) - 1:
                            holding_position.iloc[t+1] = 1
                        current_holding_position = 1
                    else:
                        trade_made.iloc[t] = 1
                        current_holding_position = 0   
                elif t != len(zscore) - 1:
                    holding_position.iloc[t+1] = -1 
         
    return trade_made, holding_position.fillna(0.0)        

def build_signal_dataframe(prices: pd.DataFrame, trading_start: pd.Timestamp, trading_end: pd.Timestamp, formation_window: int, 
                           zscore_window: int, entry_threshold: float , exit_threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    """Combine the above into one dataframe"""
    spread, betas = construct_spread(prices, trading_start - pd.DateOffset(days = zscore_window), trading_end, formation_window)
    spread_mean, spread_std, zscore = compute_zscore(spread, zscore_window) 
    trade_made, holding_position = generate_positions(zscore, trading_start, entry_threshold, exit_threshold) 

    signal_df = prices.copy()
    signal_df['spread'] = spread
    signal_df['spread_mean'] = spread_mean
    signal_df['spread_std'] = spread_std
    signal_df['zscore'] = zscore
    signal_df['betas'] = betas
    signal_df['holding_position'] = holding_position
    signal_df['trade_made'] = trade_made 

    return signal_df, spread


# ---------------- Visual inspection ------------------


def plot_spread(signal_df: pd.DataFrame) -> None:
    signal_df['spread'].plot(figsize = (12,5))
    plt.title(f'Spread after regressing {signal_df.columns[0]} on {signal_df.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.show()

def plot_zscore(signal_df: pd.DataFrame, entry_threshold: float, exit_threshold: float) -> None:
    signal_df['zscore'].plot(figsize = (12,5), label = 'z-score')
    plt.axhline(entry_threshold, linestyle = '--', color = 'red', label = 'Entry thresholds')
    plt.axhline(-entry_threshold, linestyle = '--', color = 'red')
    plt.axhline(exit_threshold, linestyle = ':', color = 'red', label = 'Exit thresholds')
    plt.axhline(-exit_threshold, linestyle = ':', color = 'red')
    plt.axhline(0, linestyle = '-')
    plt.title(f'Rolling z-score of spread after regressing {signal_df.columns[0]} on {signal_df.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

def plot_position(signal_df: pd.DataFrame) -> None:
    signal_df['holding_position'].plot(figsize=(12, 3))
    plt.title(f'Holding position for {signal_df.columns[0]}/{signal_df.columns[1]} spread')
    plt.xlabel('Date')
    plt.ylabel('Position')
    plt.tight_layout()
    plt.show()

def run_plots2(df: pd.DataFrame, entry_threshold: float, exit_threshold: float) -> None:
    plot_spread(df)
    plot_zscore(df, entry_threshold, exit_threshold)
    plot_position(df)