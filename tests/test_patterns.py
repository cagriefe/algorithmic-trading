import numpy as np
import pytest
from helpers import make_ohlcv, path

from algotrading.patterns import detect_chart_patterns
from algotrading.price_action import find_swing_points


def detect(close, scale=1.0):
    data = make_ohlcv(np.asarray(close) * scale, spread=0.2 * scale)
    data = find_swing_points(data, lookback=3)
    data['ATR'] = 2.0 * scale
    return detect_chart_patterns(data, lookback=3)['Pattern']


# close path and the bar where the last pivot is confirmed (pivot + lookback)
REVERSALS = {
    'Double Top': (path((0, 100), (10, 110), (20, 102), (30, 110.3), (45, 95)), 33),
    'Double Bottom': (path((0, 120), (10, 110), (20, 118), (30, 110.3), (45, 125)), 33),
    'Triple Top': (path((0, 100), (10, 110), (18, 102), (26, 110.4), (34, 102), (42, 110.2), (52, 95)), 45),
    'Triple Bottom': (path((0, 120), (10, 110), (18, 118), (26, 109.6), (34, 118), (42, 109.8), (52, 125)), 45),
    'Head and Shoulders': (path((0, 100), (10, 110), (18, 102), (26, 118), (34, 102), (42, 110.3), (52, 95)), 45),
    'Inverse Head and Shoulders': (
        path((0, 120), (10, 110), (18, 118), (26, 102), (34, 118), (42, 109.7), (52, 125)), 45),
}


@pytest.mark.parametrize('scale', [1.0, 0.001])
@pytest.mark.parametrize('name', REVERSALS)
def test_reversal_patterns_are_reported_on_confirmation_at_any_price_scale(name, scale):
    close, confirmation_bar = REVERSALS[name]
    patterns = detect(close, scale)

    assert patterns.iloc[confirmation_bar] == name
    assert name not in set(patterns.iloc[:confirmation_bar])


def test_bull_flag_breakout():
    close = np.r_[np.full(20, 100.0), np.linspace(101.5, 115, 10), [114.5, 114, 113.8, 113.5, 113.2, 117.0]]
    assert detect(close).iloc[-1] == 'Bull Flag'


def test_bear_flag_breakdown():
    close = np.r_[np.full(20, 100.0), np.linspace(98.5, 85, 10), [85.5, 86, 86.2, 86.5, 86.8, 83.0]]
    assert detect(close).iloc[-1] == 'Bear Flag'


def test_cup_and_handle_breakout():
    cup = 112.5 + 7.5 * np.cos(np.linspace(0, 2 * np.pi, 30))
    close = np.r_[np.full(5, 120.0), cup, [119.0, 118.0, 117.5, 118.0, 118.5, 122.0]]
    assert detect(close).iloc[-1] == 'Cup and Handle'


def test_rectangle_is_reported_when_range_starts():
    patterns = detect(101 + 0.5 * np.sin(np.arange(40)))

    assert patterns.iloc[19] == 'Rectangle'
    assert 'Rectangle' not in set(patterns.iloc[:19])


def test_steady_trend_has_no_patterns():
    assert (detect(np.linspace(100, 160, 60)) == 'None').all()
