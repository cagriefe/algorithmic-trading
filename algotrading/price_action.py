import numpy as np

SWING_LOOKBACK = 3
ADX_TREND_THRESHOLD = 25


def _first_of_plateau(pivots, lookback):
    # Pivots closer than `lookback` bars can only be equal highs/lows of one plateau
    positions = np.flatnonzero(pivots.notna().to_numpy())
    pivots = pivots.copy()
    pivots.iloc[positions[1:][np.diff(positions) <= lookback]] = np.nan
    return pivots


def find_swing_points(data, lookback=SWING_LOOKBACK):
    columns = ['Swing_High', 'Swing_Low', 'Last_Swing_High', 'Last_Swing_Low']
    if len(data) < 2 * lookback + 1:
        print('Insufficient data to detect swing points.')
        for column in columns:
            data[column] = np.nan
        return data

    window = 2 * lookback + 1
    rolling_high = data['high'].rolling(window=window, center=True).max()
    rolling_low = data['low'].rolling(window=window, center=True).min()
    # Pivots are marked on the bar where they happened (used for plotting) ...
    data['Swing_High'] = _first_of_plateau(data['high'].where(data['high'] == rolling_high), lookback)
    data['Swing_Low'] = _first_of_plateau(data['low'].where(data['low'] == rolling_low), lookback)
    # ... but a pivot is only known `lookback` bars later, so signals use confirmed levels.
    data['Last_Swing_High'] = data['Swing_High'].shift(lookback).ffill()
    data['Last_Swing_Low'] = data['Swing_Low'].shift(lookback).ffill()
    return data


def _breakout_labels(data, resistance, support, adx_threshold):
    # Only consider breakouts in trending markets with above-average volume
    confirmed = (data['ADX'] > adx_threshold) & (data['volume'] > data['Volume_SMA_20'])
    close, prev_close = data['close'], data['close'].shift(1)
    above = confirmed & (close > resistance) & (prev_close <= resistance.shift(1))
    below = confirmed & (close < support) & (prev_close >= support.shift(1))
    return np.select([above, below], ['Above', 'Below'], default='No')


def detect_breakouts(data, adx_threshold=ADX_TREND_THRESHOLD):
    data['Breakout'] = _breakout_labels(data, data['Last_Swing_High'], data['Last_Swing_Low'], adx_threshold)
    return data


def _causal_trendline(pivots, lookback, n_points=2):
    """Line through the last `n_points` pivots, using only pivots confirmed at each bar."""
    values = pivots.to_numpy(dtype=float)
    positions = np.flatnonzero(~np.isnan(values))
    line = np.full(len(values), np.nan)
    for k in range(n_points - 1, len(positions)):
        start = positions[k] + lookback
        if start >= len(values):
            break
        end = positions[k + 1] + lookback if k + 1 < len(positions) else len(values)
        xs = positions[k - n_points + 1:k + 1]
        slope, intercept = np.polyfit(xs, values[xs], 1)
        bars = np.arange(start, min(end, len(values)))
        line[bars] = slope * bars + intercept
    return line


def calculate_trendlines(data, lookback=SWING_LOOKBACK):
    data['Trendline_High'] = _causal_trendline(data['Swing_High'], lookback)
    data['Trendline_Low'] = _causal_trendline(data['Swing_Low'], lookback)
    return data


def display_trendline(data, line, level):
    """Trendline for charts: break it where a new pivot is confirmed and hide far extrapolations."""
    new_segment = level.ne(level.shift(1))
    too_far = (line - data['close']).abs() > 3 * data['ATR']
    return line.mask(new_segment | too_far)


def detect_trendline_breakouts(data, adx_threshold=ADX_TREND_THRESHOLD):
    data['Trendline_Breakout'] = _breakout_labels(data, data['Trendline_High'], data['Trendline_Low'], adx_threshold)
    return data
