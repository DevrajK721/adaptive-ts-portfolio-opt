import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

from modelFitting import modelFitting
from tradingStrategy import ModelDrivenStrategy


def backtest_strategy(
    ticker: str,
    rf_rate: float = 0.02,
    initial_cash: float = 100000.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run a simple walk-forward backtest for ``ticker``.

    Parameters
    ----------
    ticker : str
        Ticker symbol to backtest.
    rf_rate : float, optional
        Annual risk free rate used for the Sharpe ratio, by default 0.02.
    initial_cash : float, optional
        Starting cash for the strategy, by default 100000.0.

    Returns
    -------
    equity_curve : pandas.DataFrame
        DataFrame indexed by date containing equity and drawdown series.
    metrics : Dict[str, float]
        Dictionary with ``max_drawdown``, ``sharpe_ratio``, ``profit`` and
        ``profit_pct`` keys.
    """
    strat = ModelDrivenStrategy(ticker)
    df = strat.load_data()
    df["Date"] = pd.to_datetime(df["Date"])

    equity = [initial_cash]
    portfolio_returns = []
    dates = []

    # require at least 100 observations for indicators
    start_idx = 100

    for i in range(start_idx, len(df)):
        # fit models on data available up to the previous day
        series = df.loc[: i - 1, "LogReturn"]
        mf = modelFitting(ticker)
        arima_model, vol_model = mf.fitModel(series)

        strat.df = df.iloc[:i].copy()
        strat.arima_model = arima_model
        strat.vol_model = vol_model
        weight = strat.compute_weight()

        if i < len(df) - 1:
            r = df.loc[i, "LogReturn"]
            portfolio_returns.append(weight * r)
            new_equity = equity[-1] * np.exp(weight * r)
            equity.append(new_equity)
            dates.append(df.loc[i, "Date"])

    equity_series = pd.Series(equity[1:], index=dates, name="equity")
    drawdown = equity_series / equity_series.cummax() - 1

    rf_daily = (1 + rf_rate) ** (1 / 252) - 1
    excess = np.exp(portfolio_returns) - 1 - rf_daily
    sharpe = np.sqrt(252) * np.mean(excess) / np.std(excess)

    profit = equity_series.iloc[-1] - initial_cash
    profit_pct = profit / initial_cash * 100

    metrics = {
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": float(sharpe),
        "profit": float(profit),
        "profit_pct": float(profit_pct),
    }

    result = pd.DataFrame({
        "equity": equity_series,
        "drawdown": drawdown,
    })

    # Plot equity and drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    equity_series.plot(ax=ax1, title=f"Equity Curve - {ticker}")
    ax1.set_ylabel("Equity ($)")
    drawdown.plot(ax=ax2, color="red", title="Drawdown")
    ax2.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

    return result, metrics