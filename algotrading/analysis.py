from .indicators import calculate_indicators
from .patterns import detect_chart_patterns
from .price_action import (ADX_TREND_THRESHOLD, SWING_LOOKBACK, calculate_trendlines, detect_breakouts,
                           detect_trendline_breakouts, find_swing_points)
from .signals import evaluate_signals


def run_analysis(data, lookback=SWING_LOOKBACK, adx_threshold=ADX_TREND_THRESHOLD):
    """Run the full pipeline on OHLCV data; every column only uses information up to its bar."""
    data = data.reset_index(drop=True)
    data = calculate_indicators(data)
    data = find_swing_points(data, lookback)
    data = detect_breakouts(data, adx_threshold)
    data = calculate_trendlines(data, lookback)
    data = detect_trendline_breakouts(data, adx_threshold)
    data = detect_chart_patterns(data, lookback)
    return evaluate_signals(data, adx_threshold)
