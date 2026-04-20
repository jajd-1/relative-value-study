# Pairs trading

DOESN'T REFLECT UPDATED CODE YET!

We build a cointegration-based pairs trading research pipeline. 


## File summaries

`data.py` Loads, cleans and aligns daily price data for a given list of tickers using the yfinance module (an open source tool that uses Yahoo Finance's publicly available APIs). Given a user-specified selection of assets, we run the Engle–Granger cointegration test on all possible pairs and retain those pairs for which the null hypothesis of no cointegration can be rejected with a p-value of 0.05. We also produce plots relating to these pairs to further identify economically plausible candidate pairs.

`signal_construction.py` For a given pair we estimate the hedge ratio by regressing one price series on the other. From this we compute the residual spread and standardise using a rolling z-score. This rolling z-score is used to generate trading signals: when there are large deviations above the equilibrium we go short on the spread, when there are large deviations below the equilibrium we go long on the spread, and when the prices returns to near the equilibrium we close our position.

`backtesting.py` We backtest the trading strategy whilst taking into account trading costs. 


## Assumptions

Beyond using a relatively simple model (e.g. we only use a static hedge ratio), we also make some further simplifying assumptions in computing returns, such as:

- Trades only occur at close
- There are no capital constraints
- Perfect shorting (i.e. no constraints or fees on borrowing, no recalls)
- No financing costs or margin requirements
- No bid-ask spread or slippage (i.e. no market impact from making the proposed trades)
- Perfect liquidity
- No stop-loss or risk controls (i.e. we can hold our positions until exit thresholds are reached)

We will return to incorporate some of these at a later date. 

## Background and methodology 

Rather than trying to predict the price movement of a particular asset, a pairs trading strategy attempts to predict the relative movement of two cointegrated assets. Roughly speaking, two assets are said to be _cointegrated_ if some linear combination of prices exhibits a long-term, stable equilibrium relationship. One may then build a trading strategy based on deviations around this equilibrium. 

More precisely, consider two time series $x_t$ and $y_t$ representing daily price data for two assets $X$ and $Y$. The standard Engle-Granger test assumes that both $x_t$ and $y_t$ are integrated to order one, denoted $I(1)$, meaning that they are non-stationary but their first differences ($\Delta x_t = x_t - x_{t-1}$ and $\Delta y_t = y_t - y_{t-1}$) are stationary. One way to check this is to test for a unit root using e.g. the ADF test, although we won't explain this here. The Engle-Granger method then regresses one asset on the other using ordinary least squares, yielding

$$y_t = \alpha + \beta x_t + \epsilon_t$$

for some $\alpha,\beta\in\mathbb{R}$. We call $\epsilon_t$ the _residual_ (or the _spread_ in our pairs trading context). The time series $x_t$ and $y_t$ are cointegrated if the residual $\epsilon_t$ is itself stationary, which can be tested using a standard t-statistic under the null hypothesis of no cointegration. We use `statsmodels.coint` on a pair of time series, which returns the t-statistic and associated p-value. 

Suppose we have now decided that two assets $X$ and $Y$ are likely cointegrated. We use `statsmodels.OLS` to estimate the coefficients $\alpha$ and $\beta$ in the above, from which we can read of the spread:

$$\epsilon_t = y_t - \alpha - \beta x_t.$$

In our pairs trading context, we refer to $\beta$ as the _hedge ratio_. We can then compute the z-score $Z_N$ of the spread using the rolling mean $\mu_N$ and rolling standard deviation $\sigma_N$ from the previous $N$-day window:

$$Z_N = \frac{\epsilon_t - \mu_N}{\sigma_N}.$$

This provides a normalised measure of how much the spread has deviated from its equilibrium. In what follows we assume $N$ is fixed and denote $Z_N$ by $Z$. 

We are now in a position build our strategy: if the z-score becomes high (e.g. above 2), then this indicates that $Y$ is overpriced relative to $X$, in which case we go short on the spread (meaning we go short on 1 share of $Y$ and long on $\beta$ shares of $X$). If the z-score becomes low (e.g. below -2), then this indicates that $Y$ is underpriced relative to $X$, in which case we go long on the spread (meaning we go long on 1 share of $Y$ and short on $\beta$ shares of $X$). If the z-score returns close to zero (e.g. above -0.5 while we are long on the spread, or below 0.5 while we are short on the spread), then we close our position (sell our long positions and buy back our short ones). 

We label our position as +1 if we're currently long on the spread, -1 if we're currently short on the spread, and 0 if our position is flat (i.e. we have no open positions). Let $z_{\text{entry}}$ denote a fixed entry threshold and $z_{\text{exit}}$ a fixed exit threshold. We use the following rules:

- If on day $t$ our position is 0 and at close it holds that $Z < -z_{\text{entry}}$, then we enter a long spread position using the close price and enter our position as +1 from day $t+1$.
- If on day $t$ our position is 0 and at close it holds that $Z > z_{\text{entry}}$, then we enter a short spread position using the close price and enter our position as -1 from day $t+1$.
- If on day $t$ our position is +1 and at close it holds that $Z > -z_{\text{exit}}$, then we exit our position using the close price and enter our position as 0 from day $t+1$.
- If on day $t$ our position is -1 and at close it holds that $Z < z_{\text{exit}}$, then we exit our position using the cloe price and enter our position as 0 from day $t+1$.

The next step is to backtest the strategy. The return on day $t$ from a long (+) or short (-) position is 

$$\pm\ \frac{\Delta y_t - \beta\Delta x_t}{|y_{t-1}| + |\beta x_{t-1}|},$$

from which we subtract a transaction cost on the days our position label changes, yielding the net return. The transaction cost is specified as a certain number of basis points per unit turnover (i.e. per dollar of trades executed). The net returns for a given period is then the product of the net daily returns over that period. In the backtesting stage we also compute the maximum drawdown (soon to add other quantities). 

## An example 

## To do