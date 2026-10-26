import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest
from helpers import make_ohlcv


@pytest.fixture
def market_data():
    """Deterministic random walk with alternating trends, long enough for every indicator."""
    rng = np.random.default_rng(7)
    n = 600
    returns = rng.normal(0, 0.02, n) + 0.006 * np.sin(np.arange(n) / 30)
    close = 100 * np.exp(np.cumsum(returns))
    data = make_ohlcv(close, spread=0.0, volume=rng.lognormal(10, 0.4, n))
    data['high'] *= 1 + rng.uniform(0, 0.015, n)
    data['low'] *= 1 - rng.uniform(0, 0.015, n)
    return data
