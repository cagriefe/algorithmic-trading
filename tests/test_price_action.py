import numpy as np
import pytest
from helpers import make_ohlcv, path

from algotrading.price_action import (calculate_trendlines, detect_breakouts, detect_trendline_breakouts,
                                      find_swing_points)


def trending(data, adx=30.0):
    data['ADX'] = adx
    data['volume'] = 2.0
    data['Volume_SMA_20'] = 1.0
    return data


def test_swing_high_is_only_used_after_confirmation():
    data = find_swing_points(make_ohlcv(path((0, 100), (10, 110), (20, 100))), lookback=3)
    peak = data['high'].iloc[10]

    assert data['Swing_High'].iloc[10] == peak
    assert data['Last_Swing_High'].iloc[:13].isna().all()
    assert (data['Last_Swing_High'].iloc[13:] == peak).all()


def test_equal_highs_on_neighbouring_bars_are_one_pivot():
    data = find_swing_points(make_ohlcv(path((0, 100), (10, 110), (20, 100))), lookback=3)

    assert data['high'].iloc[10] == data['high'].iloc[11]
    assert data['Swing_High'].dropna().index.tolist() == [10]


@pytest.mark.parametrize('adx, expected', [(30.0, 1), (20.0, 0)])
def test_breakout_fires_once_when_close_crosses_confirmed_swing_high(adx, expected):
    data = make_ohlcv(path((0, 100), (10, 110), (16, 105), (26, 115)))
    data = detect_breakouts(find_swing_points(trending(data, adx), lookback=3))
    above = np.flatnonzero(data['Breakout'] == 'Above')

    assert len(above) == expected
    if expected:
        first_close_above = np.flatnonzero(data['close'] > data['high'].iloc[10])[0]
        assert above[0] == first_close_above
    assert not (data['Breakout'] == 'Below').any()


def test_trendline_connects_last_two_confirmed_pivots():
    data = make_ohlcv(path((0, 100), (10, 110), (15, 100), (20, 105), (25, 95), (30, 97)))
    data = calculate_trendlines(find_swing_points(data, lookback=3), lookback=3)
    first, second = data['high'].iloc[10], data['high'].iloc[20]
    slope = (second - first) / 10

    assert data['Swing_High'].dropna().index.tolist() == [10, 20]
    assert data['Trendline_High'].iloc[:23].isna().all()
    assert data['Trendline_High'].iloc[23] == pytest.approx(second + 3 * slope)
    assert data['Trendline_High'].iloc[30] == pytest.approx(second + 10 * slope)


def test_trendline_breakout_above_descending_resistance():
    data = make_ohlcv(path((0, 100), (10, 110), (15, 100), (20, 106), (25, 98), (35, 112)))
    data = calculate_trendlines(find_swing_points(trending(data), lookback=3), lookback=3)
    data = detect_trendline_breakouts(data)
    above = np.flatnonzero(data['Trendline_Breakout'] == 'Above')

    assert len(above) == 1
    i = above[0]
    assert data['close'].iloc[i] > data['Trendline_High'].iloc[i]
    assert data['close'].iloc[i - 1] <= data['Trendline_High'].iloc[i - 1]
