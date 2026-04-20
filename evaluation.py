import numpy as np 
import pandas as pd 
import data 
import signal_construction as sc # pyright: ignore[reportMissingImports]
import backtesting as bt # pyright: ignore[reportMissingImports]

def compute_drawdown(cumulative_return: pd.Series) -> tuple[pd.Series, float]:
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1.0
    max_drawdown = - drawdown.min()
    return drawdown, max_drawdown


def return_stats(returns: pd.Series, trading_days: int, risk_free_rate: float, trading_start: pd.Timestamp, trading_end: pd.Timestamp) -> pd.Series:
    mask = (returns.index >= trading_start) & (returns.index <= trading_end)
    returns = returns.loc[mask].fillna(0.0)
    cumulative_return = (1 + returns).cumprod()
    _, max_drawdown = compute_drawdown(cumulative_return)

    total_return = cumulative_return.iloc[-1] - 1.0                                                #computes total return for the input series
    annualised_return = (cumulative_return.iloc[-1] ** (trading_days / len(returns))) - 1.0        #trading_days / len(returns) should be number of years in the period considered
    annualised_vol = returns.std(ddof = 1) * np.sqrt(trading_days)                                 #annualised volatility is daily volatility multiplied by square root of number of trading days

    if returns.std(ddof = 1) == 0:
        annualised_sharpe = np.nan
    else:                                                                                          #mean daily excess return multiplied by number of trading days, divided by yearly volatility
        annualised_sharpe =  trading_days * (returns.mean() - ((1 + risk_free_rate)**(1/trading_days) - 1)) / annualised_vol

    return pd.Series({
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "annualised_sharpe": annualised_sharpe,
        "max_drawdown": max_drawdown,
    })


def extract_trades(backtest_df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    trades = []
    entry_date = None 
    direction = 0
    cost_rate = cost_bps / 10000.0

    holding_position = backtest_df['holding_position']
    gross_return = backtest_df['gross_return']
    net_return = backtest_df['net_return']
    position_reversal = False 

    for i in range(len(holding_position)):
        prev_pos = holding_position.iloc[i-1] if i>0 else 0
        curr_pos = holding_position.iloc[i] 

        if prev_pos == 0 and curr_pos != 0:
            entry_date = i-1                                                    #we enter trades at close, and this is reflected in the next day's position
            direction = curr_pos 
        
        elif ((prev_pos != 0 and curr_pos == 0) or (prev_pos == 1 and curr_pos == -1) or (prev_pos == -1 and curr_pos == 1)) and entry_date is not None:
            exit_date = i-1                                                     #we exit trades at close, and this is reflected in the next day's position

            gross_trade_return = gross_return.iloc[entry_date : exit_date + 1].copy()
            net_trade_return = net_return.iloc[entry_date : exit_date + 1].copy()

            if position_reversal == False and curr_pos != 0:
                net_trade_return.iloc[-1] += cost_rate 
                
            elif position_reversal == True and curr_pos == 0:
                gross_trade_return.iloc[0] = 0.0 
                net_trade_return.iloc[0] = -cost_rate

            elif position_reversal == True and curr_pos != 0:
                gross_trade_return.iloc[0] = 0.0
                net_trade_return.iloc[-1] += cost_rate
                net_trade_return.iloc[0] = -cost_rate

            cumulative_gross_trade_return = (1.0 + gross_trade_return).prod() - 1.0
            cumulative_net_trade_return = (1.0 + net_trade_return).prod() - 1.0

            trades.append({
                'entry_date': backtest_df.index[entry_date],
                'exit_date': backtest_df.index[exit_date],
                'direction': direction,
                'duration_days': exit_date - entry_date, 
                'cumulative_gross_trade_return': cumulative_gross_trade_return,
                'cumulative_net_trade_return': cumulative_net_trade_return
            })

            direction = curr_pos
            entry_date = None if curr_pos == 0 else i-1
            position_reversal = True if curr_pos != 0 else False 
            

    if entry_date is not None:
        exit_date = len(holding_position) - 1

        gross_trade_return = gross_return.iloc[entry_date : exit_date + 1].copy()
        net_trade_return = net_return.iloc[entry_date : exit_date + 1].copy()

        if position_reversal == True: 
            gross_trade_return.iloc[0] = 0.0
            net_trade_return.iloc[0] = -cost_rate

        cumulative_gross_trade_return = (1.0 + gross_trade_return).prod() - 1.0
        cumulative_net_trade_return = (1.0 + net_trade_return).prod() - 1.0

        trades.append({
            'entry_date': backtest_df.index[entry_date],
            'exit_date': backtest_df.index[exit_date],
            'direction': direction,
            'duration_days': exit_date - entry_date, 
            'cumulative_gross_trade_return': cumulative_gross_trade_return,
            'cumulative_net_trade_return': cumulative_net_trade_return
        })

    return pd.DataFrame(trades)

def trade_stats(trades_df: pd.Series) -> None:
    if trades_df.empty: 
        return pd.Series({
            "trade_count": 0,
            "avg_holding_period_days": np.nan,
            "hit_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan
        })
    
    trade_return = trades_df['cumulative_net_trade_return']

    wins = trade_return[trade_return > 0]
    losses = trade_return[trade_return < 0]
    avg_win = wins.mean() if len(wins)>0 else np.nan
    avg_loss = losses.mean() if len(losses)>0 else np.nan

    stats = pd.Series({
        'trade_count': len(trades_df),
        'hit_rate': (trade_return > 0).mean(),
        'avg_trade_return': trade_return.mean(),
        'median_trade_return': trade_return.median(),
        'best_trade_return': trade_return.max(),
        'worst_trade_return': trade_return.min(),
        'avg_win': avg_win,
        'avg_loss': avg_loss, 
        'payoff_ratio': avg_win / abs(avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else np.nan,
        'avg_holding_period_days': trades_df['duration_days'].mean(),
        'median_holding_period_days': trades_df['duration_days'].median(),
    })

    return stats 






