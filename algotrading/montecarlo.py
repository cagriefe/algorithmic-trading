import numpy as np


def trade_returns(trades, initial_capital):
    """Return of every trade relative to the equity just before it."""
    pnl = trades['pnl'].to_numpy(dtype=float)
    equity_before = initial_capital + np.concatenate([[0.0], np.cumsum(pnl)[:-1]])
    return pnl / equity_before


def monte_carlo(trades, initial_capital, simulations=2000, seed=0):
    """Resample the trades with replacement to show how much luck is in a backtest.

    Each simulation draws as many trades as the backtest made, so the spread of outcomes shows
    what the same edge could have produced with a different sequence of wins and losses.
    Drawdowns are measured trade to trade. Returns None when there are no trades.
    """
    returns = trade_returns(trades, initial_capital)
    if len(returns) == 0 or simulations <= 0:
        return None
    rng = np.random.default_rng(seed)
    paths = np.cumprod(1 + rng.choice(returns, size=(simulations, len(returns))), axis=1)
    peaks = np.maximum.accumulate(np.maximum(paths, 1.0), axis=1)
    drawdowns = (paths / peaks - 1).min(axis=1) * 100
    final = (paths[:, -1] - 1) * 100
    return {
        'Simulations': simulations,
        'Median return %': np.median(final),
        'Return 5th percentile %': np.percentile(final, 5),
        'Return 95th percentile %': np.percentile(final, 95),
        'Median max drawdown %': np.median(drawdowns),
        'Max drawdown 5th percentile %': np.percentile(drawdowns, 5),
        'Probability of loss %': (final < 0).mean() * 100,
    }
