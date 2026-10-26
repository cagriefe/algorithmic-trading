import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from dotenv import load_dotenv
import talib
from matplotlib.dates import DateFormatter, AutoDateLocator
from score import evaluate_signals

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

client = Client(API_KEY, API_SECRET)

SWING_LOOKBACK = 3
ADX_TREND_THRESHOLD = 25


def get_historical_data(symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    # Binance returns the still-forming candle last; using it would make signals repaint.
    now_ms = int(time.time() * 1000)
    data = data[data['close_time'].astype('int64') < now_ms].reset_index(drop=True)
    for column in ['open', 'high', 'low', 'close', 'volume']:
        data[column] = data[column].astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data

def calculate_indicators(data):
    data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['close'], timeperiod=200)
    data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
    data['EMA_200'] = talib.EMA(data['close'], timeperiod=200)
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    data['PLUS_DI'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    data['MINUS_DI'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['Volume_SMA_20'] = talib.SMA(data['volume'], timeperiod=20)
    return data

def find_swing_points(data, lookback=SWING_LOOKBACK):
    columns = ['Swing_High', 'Swing_Low', 'Last_Swing_High', 'Last_Swing_Low']
    if len(data) < 2 * lookback + 1:
        print("Insufficient data to detect swing points.")
        for column in columns:
            data[column] = np.nan
        return data

    window = 2 * lookback + 1
    rolling_high = data['high'].rolling(window=window, center=True).max()
    rolling_low = data['low'].rolling(window=window, center=True).min()
    # Pivots are marked on the bar where they happened (used for plotting) ...
    data['Swing_High'] = data['high'].where(data['high'] == rolling_high)
    data['Swing_Low'] = data['low'].where(data['low'] == rolling_low)
    # ... but a pivot is only known `lookback` bars later, so signals use confirmed levels.
    data['Last_Swing_High'] = data['Swing_High'].shift(lookback).ffill()
    data['Last_Swing_Low'] = data['Swing_Low'].shift(lookback).ffill()
    return data

def _breakout_labels(data, resistance, support):
    # Only consider breakouts in trending markets with above-average volume
    confirmed = (data['ADX'] > ADX_TREND_THRESHOLD) & (data['volume'] > data['Volume_SMA_20'])
    close, prev_close = data['close'], data['close'].shift(1)
    above = confirmed & (close > resistance) & (prev_close <= resistance.shift(1))
    below = confirmed & (close < support) & (prev_close >= support.shift(1))
    return np.select([above, below], ['Above', 'Below'], default='No')

def detect_breakouts(data):
    data['Breakout'] = _breakout_labels(data, data['Last_Swing_High'], data['Last_Swing_Low'])
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

def detect_trendline_breakouts(data):
    data['Trendline_Breakout'] = _breakout_labels(data, data['Trendline_High'], data['Trendline_Low'])
    return data

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

def print_signal_analysis(data):
    print("\n=== Trading Signal Analysis ===")

    # Print latest/current signal first
    latest = data.iloc[-1]
    print("\nCURRENT SIGNAL:")
    print("-" * 80)
    print(f"Date: {latest['timestamp']}")
    print(f"Score: {latest['Score']:.2f}")
    print(f"Confidence: {latest['Confidence']:.1f}%")
    print(f"Decision: {latest['Decision']}")
    print(f"Close Price: {latest['close']}")
    print(f"Reasoning: {latest['Reasoning']}")
    print("-" * 80)

    # Filter for meaningful signals
    significant_signals = data[data['Score'] > 3].copy()
    if len(significant_signals) == 0:
        print("No significant trading signals found")
        return

    print("\nHISTORICAL BEST SIGNALS:")
    print(f"Total signals analyzed: {len(data)}")
    print(f"Significant signals found: {len(significant_signals)}")
    print("-" * 80)
    
    # Sort by score descending for historical best
    sorted_signals = significant_signals.sort_values('Score', ascending=False)
    
    for idx, row in sorted_signals.head(3).iterrows():
        print(f"\nDate: {row['timestamp']}")
        print(f"Score: {row['Score']:.2f}")
        print(f"Confidence: {row['Confidence']:.1f}%")
        print(f"Decision: {row['Decision']}")
        print(f"Close Price: {row['close']}")
        print(f"Reasoning: {row['Reasoning']}")
        print("-" * 80)


def plot_trend(data):
    if data.empty:
        print("No data to plot.")
        return

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    # Main price plot
    ax1.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', color='orange', linestyle='--', alpha=0.7)
    ax1.plot(data.index, data['SMA_200'], label='SMA 200', color='purple', linestyle='--', alpha=0.7)
    
    # Plot Bollinger Bands
    ax1.plot(data.index, data['BB_upper'], 'k--', alpha=0.3)
    ax1.plot(data.index, data['BB_middle'], 'k--', alpha=0.3)
    ax1.plot(data.index, data['BB_lower'], 'k--', alpha=0.3)
    ax1.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1, color='gray', label='Bollinger Bands')

    # Swing points and trendlines
    ax1.scatter(data.index, data['Swing_High'], color='green', label='Swing High', marker='^', alpha=0.9)
    ax1.scatter(data.index, data['Swing_Low'], color='red', label='Swing Low', marker='v', alpha=0.9)
    ax1.plot(data.index, data['Trendline_High'], label='Trendline High', color='darkgreen', linestyle='-.', alpha=0.7)
    ax1.plot(data.index, data['Trendline_Low'], label='Trendline Low', color='darkred', linestyle='-.', alpha=0.7)

    # Breakouts
    breakout_above = data[data['Breakout'] == 'Above']
    breakout_below = data[data['Breakout'] == 'Below']
    ax1.scatter(breakout_above.index, breakout_above['close'], color='lime', label='Breakout Above', marker='o')
    ax1.scatter(breakout_below.index, breakout_below['close'], color='darkred', label='Breakout Below', marker='o')

    # Trendline breakouts
    trendline_breakout_above = data[data['Trendline_Breakout'] == 'Above']
    trendline_breakout_below = data[data['Trendline_Breakout'] == 'Below']
    ax1.scatter(trendline_breakout_above.index, trendline_breakout_above['close'], color='cyan', label='Trendline Breakout Above', marker='x')
    ax1.scatter(trendline_breakout_below.index, trendline_breakout_below['close'], color='magenta', label='Trendline Breakout Below', marker='x')

    # Trading signals
    long_signals = data[data['Decision'] == 'Long']
    strong_long_signals = data[data['Decision'] == 'Strong Long']
    ax1.scatter(long_signals.index, long_signals['close'], color='gold', label='Long Signal', marker='*', s=100)
    ax1.scatter(strong_long_signals.index, strong_long_signals['close'], color='green', label='Strong Long Signal', marker='*', s=150)

    # Volume subplot
    ax2.bar(data.index, data['volume'], color='blue', alpha=0.3, label='Volume')
    
    # Format axes
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Titles and labels
    ax1.set_title('Technical Analysis Overview', pad=20)
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    
    # Legend
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Layout
    plt.tight_layout()
    
    # plt.savefig('technical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    symbol = 'APEUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    start_str = '1 year ago UTC'

    data = get_historical_data(symbol, interval, start_str)
    data = calculate_indicators(data)
    data = find_swing_points(data)
    data = detect_breakouts(data)
    data = calculate_trendlines(data)
    data = detect_trendline_breakouts(data)
    data = detect_chart_patterns(data)
    data = evaluate_signals(data)
    
    # Print analysis with current and historical signals
    print_signal_analysis(data)
        
    # Save the latest data with indicators to a CSV file
    latest_data = data.tail()
    latest_data.to_csv('latest_data_analysis.csv')
    
    # Generate plot
    plot_trend(data)

if __name__ == "__main__":
    main()