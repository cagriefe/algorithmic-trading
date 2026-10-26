import numpy as np

from .price_action import SWING_LOOKBACK

BULLISH_PATTERNS = ('Inverse Head and Shoulders', 'Double Bottom', 'Triple Bottom',
                    'Bull Flag', 'Cup and Handle')
BEARISH_PATTERNS = ('Head and Shoulders', 'Double Top', 'Triple Top', 'Bear Flag')


def _peak_formations(pivots, opposite, atr, lookback, tolerance_atr, prominence_atr, max_width):
    """Find double/triple/head-and-shoulders formations among confirmed peaks.

    Yields (confirmation_bar, kind) with kind in {'hs', 'triple', 'double'}.
    Bottoms are handled by passing negated lows/highs.
    """
    positions = np.flatnonzero(~np.isnan(pivots))

    def pullback(a, b):
        # Distance from the lower of two peaks down to the deepest point between them
        if b - a < 2:
            return np.nan
        return min(pivots[a], pivots[b]) - opposite[a + 1:b].min()

    for k in range(1, len(positions)):
        bar = positions[k] + lookback
        if bar >= len(pivots) or np.isnan(atr[bar]):
            continue
        tolerance, prominence = tolerance_atr * atr[bar], prominence_atr * atr[bar]

        if k >= 2 and positions[k] - positions[k - 2] <= max_width:
            p1, p2, p3 = positions[k - 2:k + 1]
            h1, h2, h3 = pivots[p1], pivots[p2], pivots[p3]
            deep = pullback(p1, p2) >= prominence and pullback(p2, p3) >= prominence
            if deep and abs(h1 - h3) <= tolerance and h2 - max(h1, h3) >= prominence:
                yield bar, 'hs'
                continue
            if deep and max(h1, h2, h3) - min(h1, h2, h3) <= tolerance:
                yield bar, 'triple'
                continue

        p1, p2 = positions[k - 1], positions[k]
        if (p2 - p1 <= max_width and abs(pivots[p1] - pivots[p2]) <= tolerance
                and pullback(p1, p2) >= prominence):
            yield bar, 'double'


def _flag_breakouts(data, pole_bars=10, flag_bars=5, pole_atr=3.0, max_retrace=0.5):
    """Bull/Bear flag: strong pole, tight counter-trend consolidation, then a breakout."""
    flag_high = data['high'].shift(1).rolling(flag_bars).max()
    flag_low = data['low'].shift(1).rolling(flag_bars).min()
    pole_end = data['close'].shift(flag_bars + 1)
    pole = pole_end - data['close'].shift(flag_bars + 1 + pole_bars)
    pole_atr_ref = data['ATR'].shift(flag_bars + 1)
    tight = (flag_high - flag_low) <= max_retrace * pole.abs()
    flag_drift = data['close'].shift(1) - data['close'].shift(flag_bars)

    bull = (tight & (pole >= pole_atr * pole_atr_ref) & (flag_drift <= 0)
            & (flag_low >= pole_end - max_retrace * pole) & (data['close'] > flag_high))
    bear = (tight & (-pole >= pole_atr * pole_atr_ref) & (flag_drift >= 0)
            & (flag_high <= pole_end - max_retrace * pole) & (data['close'] < flag_low))
    return bull.to_numpy(), bear.to_numpy()


def _cup_and_handle_breakouts(data, cup_bars=30, handle_bars=5, depth_atr=2.0, tolerance_atr=1.0):
    """Rounded cup with similar rims, a shallow handle, then a close above the rim."""
    high, low, close, atr = (data[c].to_numpy(dtype=float) for c in ['high', 'low', 'close', 'ATR'])
    third = cup_bars // 3
    events = np.zeros(len(data), dtype=bool)
    for i in range(cup_bars + handle_bars, len(data)):
        cup_start, cup_end = i - handle_bars - cup_bars, i - handle_bars
        ref_atr = atr[cup_end - 1]
        if np.isnan(ref_atr):
            continue
        cup_high, cup_low = high[cup_start:cup_end], low[cup_start:cup_end]
        left_rim, right_rim = cup_high[:third].max(), cup_high[-third:].max()
        bottom_pos = cup_low.argmin()
        bottom = cup_low[bottom_pos]
        rim = max(left_rim, right_rim)
        if not third <= bottom_pos < cup_bars - third:
            continue
        if abs(left_rim - right_rim) > tolerance_atr * ref_atr or min(left_rim, right_rim) - bottom < depth_atr * ref_atr:
            continue
        handle_high, handle_low = high[i - handle_bars:i].max(), low[i - handle_bars:i].min()
        if handle_high > rim or handle_low < (rim + bottom) / 2:
            continue
        events[i] = close[i] > rim >= close[i - 1]
    return events


def _rectangle_starts(data, bars=20, range_atr=4.0, drift_atr=1.0):
    """First bar of a sideways range that is narrow relative to volatility."""
    price_range = data['high'].rolling(bars).max() - data['low'].rolling(bars).min()
    drift = (data['close'] - data['close'].shift(bars - 1)).abs()
    in_range = (price_range <= range_atr * data['ATR']) & (drift <= drift_atr * data['ATR'])
    return (in_range & ~in_range.shift(1, fill_value=False)).to_numpy()


def detect_chart_patterns(data, lookback=SWING_LOOKBACK, tolerance_atr=0.5, prominence_atr=1.0, max_width=60):
    """Detect chart patterns using only information available at each bar.

    Reversal patterns are built from confirmed swing pivots and reported on the bar where
    their last pivot is confirmed. Tolerances scale with ATR, so they work at any price level.
    Each bar keeps the first matching pattern in priority order.
    """
    patterns = np.full(len(data), 'None', dtype=object)

    def mark(bar, name):
        if patterns[bar] == 'None':
            patterns[bar] = name

    high, low, atr = (data[c].to_numpy(dtype=float) for c in ['high', 'low', 'ATR'])
    tops = {'hs': 'Head and Shoulders', 'triple': 'Triple Top', 'double': 'Double Top'}
    bottoms = {'hs': 'Inverse Head and Shoulders', 'triple': 'Triple Bottom', 'double': 'Double Bottom'}
    args = (atr, lookback, tolerance_atr, prominence_atr, max_width)
    for bar, kind in _peak_formations(data['Swing_High'].to_numpy(dtype=float), low, *args):
        mark(bar, tops[kind])
    for bar, kind in _peak_formations(-data['Swing_Low'].to_numpy(dtype=float), -high, *args):
        mark(bar, bottoms[kind])

    for bar in np.flatnonzero(_cup_and_handle_breakouts(data)):
        mark(bar, 'Cup and Handle')
    bull_flags, bear_flags = _flag_breakouts(data)
    for bar in np.flatnonzero(bull_flags):
        mark(bar, 'Bull Flag')
    for bar in np.flatnonzero(bear_flags):
        mark(bar, 'Bear Flag')
    for bar in np.flatnonzero(_rectangle_starts(data)):
        mark(bar, 'Rectangle')

    data['Pattern'] = patterns
    return data
