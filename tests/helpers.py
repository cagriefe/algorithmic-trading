import numpy as np
import pandas as pd


def make_ohlcv(close, spread=0.2, volume=1000.0, start='2024-01-01', freq='D'):
    """OHLCV frame around a close path: open = previous close, high/low = body +/- spread."""
    close = np.asarray(close, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        'timestamp': pd.date_range(start, periods=len(close), freq=freq),
        'open': open_,
        'high': np.maximum(open_, close) + spread,
        'low': np.minimum(open_, close) - spread,
        'close': close,
        'volume': np.broadcast_to(np.asarray(volume, dtype=float), close.shape).copy(),
    })


def path(*points):
    """Piecewise-linear close prices through (bar, price) waypoints."""
    bars, prices = zip(*points)
    return np.interp(np.arange(bars[-1] + 1), bars, prices)
