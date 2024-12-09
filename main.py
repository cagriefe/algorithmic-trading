import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from dotenv import load_dotenv
import talib
from scipy.stats import linregress
from matplotlib.dates import DateFormatter, AutoDateLocator
from score import *

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


def get_historical_data(symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['close'] = data['close'].astype(float)
    data['high'] = data['high'].astype(float)
    data['low'] = data['low'].astype(float)
    data['volume'] = data['volume'].astype(float)
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
    data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    return data

def find_swing_points(data, lookback=3):
    if len(data) < 2 * lookback + 1:
        print("Insufficient data to detect swing points.")
        data['Swing_High'] = np.nan
        data['Swing_Low'] = np.nan
        return data

    data['Swing_High'] = data['high'].rolling(window=2 * lookback + 1, center=True).apply(
        lambda x: x[lookback] if x[lookback] == max(x) else np.nan, raw=True
    )
    data['Swing_Low'] = data['low'].rolling(window=2 * lookback + 1, center=True).apply(
        lambda x: x[lookback] if x[lookback] == min(x) else np.nan, raw=True
    )
    return data

def detect_breakouts(data):
    data['Breakout'] = 'No'
    for i in range(1, len(data)):
        if data['ADX'][i] > 25:  # Only consider breakouts in trending markets
            if not pd.isna(data['Swing_High'][i-1]) and data['close'][i] > data['Swing_High'][i-1] and data['volume'][i] > data['volume'][i-1]:
                data.loc[i, 'Breakout'] = 'Above'
            elif not pd.isna(data['Swing_Low'][i-1]) and data['close'][i] < data['Swing_Low'][i-1] and data['volume'][i] > data['volume'][i-1]:
                data.loc[i, 'Breakout'] = 'Below'
    return data

def calculate_trendlines(data):
    swing_highs = data.dropna(subset=['Swing_High'])
    swing_lows = data.dropna(subset=['Swing_Low'])
    
    if len(swing_highs) > 1:
        slope_high, intercept_high, _, _, _ = linregress(swing_highs.index, swing_highs['Swing_High'])
        data['Trendline_High'] = slope_high * data.index + intercept_high
    else:
        print("Insufficient swing highs for trendline calculation.")
        data['Trendline_High'] = np.nan
    
    if len(swing_lows) > 1:
        slope_low, intercept_low, _, _, _ = linregress(swing_lows.index, swing_lows['Swing_Low'])
        data['Trendline_Low'] = slope_low * data.index + intercept_low
    else:
        print("Insufficient swing lows for trendline calculation.")
        data['Trendline_Low'] = np.nan
    
    return data

def detect_trendline_breakouts(data):
    data['Trendline_Breakout'] = 'No'
    for i in range(1, len(data)):
        if data['ADX'][i] > 25:  # Only consider trendline breakouts in trending markets
            if not pd.isna(data['Trendline_High'][i-1]) and data['close'][i] > data['Trendline_High'][i-1] and data['volume'][i] > data['volume'][i-1]:
                data.loc[i, 'Trendline_Breakout'] = 'Above'
            elif not pd.isna(data['Trendline_Low'][i-1]) and data['close'][i] < data['Trendline_Low'][i-1] and data['volume'][i] > data['volume'][i-1]:
                data.loc[i, 'Trendline_Breakout'] = 'Below'
    return data

def detect_chart_patterns(data):
    data['Pattern'] = 'None'
    window = 20  # Pattern detection window
    
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i+1]
        
        # Head and Shoulders
        if i > window + 10:
            left_shoulder = max(data['high'][i-window:i-window//2])
            head = max(data['high'][i-window//2:i-5])
            right_shoulder = max(data['high'][i-5:i])
            neckline = min(data['low'][i-window:i])
            
            if (left_shoulder < head and right_shoulder < head and 
                abs(left_shoulder - right_shoulder)/left_shoulder < 0.02):
                data.loc[i, 'Pattern'] = 'Head and Shoulders'
        
        # Inverse Head and Shoulders
        if i > window + 10:
            left_valley = min(data['low'][i-window:i-window//2])
            head_valley = min(data['low'][i-window//2:i-5])
            right_valley = min(data['low'][i-5:i])
            neckline = max(data['high'][i-window:i])
            
            if (left_valley > head_valley and right_valley > head_valley and 
                abs(left_valley - right_valley)/left_valley < 0.02):
                data.loc[i, 'Pattern'] = 'Inverse Head and Shoulders'
        
        # Double Top
        if i > 5:
            if (data['high'][i-1] > data['high'][i] and 
                abs(data['high'][i-1] - data['high'][i-3])/data['high'][i-1] < 0.02 and
                data['high'][i-2] < data['high'][i-1]):
                data.loc[i, 'Pattern'] = 'Double Top'
        
        # Double Bottom
        if i > 5:
            if (data['low'][i-1] < data['low'][i] and 
                abs(data['low'][i-1] - data['low'][i-3])/data['low'][i-1] < 0.02 and
                data['low'][i-2] > data['low'][i-1]):
                data.loc[i, 'Pattern'] = 'Double Bottom'
        
        # Triple Top
        if i > 10:
            tops = [data['high'][i-j] for j in range(1, 7)]
            if (len(set(round(x, 2) for x in tops[:3])) == 1 and
                all(x < max(tops[:3]) for x in tops[3:])):
                data.loc[i, 'Pattern'] = 'Triple Top'
        
        # Triple Bottom
        if i > 10:
            bottoms = [data['low'][i-j] for j in range(1, 7)]
            if (len(set(round(x, 2) for x in bottoms[:3])) == 1 and
                all(x > min(bottoms[:3]) for x in bottoms[3:])):
                data.loc[i, 'Pattern'] = 'Triple Bottom'
        
        # Bull Flag
        if i > window:
            if (data['close'][i-window:i].is_monotonic_increasing and
                data['high'][i-5:i].is_monotonic_decreasing):
                data.loc[i, 'Pattern'] = 'Bull Flag'
        
        # Bear Flag
        if i > window:
            if (data['close'][i-window:i].is_monotonic_decreasing and
                data['low'][i-5:i].is_monotonic_increasing):
                data.loc[i, 'Pattern'] = 'Bear Flag'
        
        # Rectangle (Trading Range)
        if i > window:
            price_range = data['high'][i-window:i] - data['low'][i-window:i]
            if price_range.std() < price_range.mean() * 0.1:
                data.loc[i, 'Pattern'] = 'Rectangle'
        
        # Cup and Handle
        if i > window + 10:
            cup_section = data['close'][i-window:i-5]
            handle_section = data['close'][i-5:i]
            if (cup_section.is_monotonic_decreasing and 
                handle_section.is_monotonic_increasing and
                min(cup_section) > min(handle_section)):
                data.loc[i, 'Pattern'] = 'Cup and Handle'
                
    return data

def print_signal_analysis(data):
    # Filter for meaningful signals
    significant_signals = data[data['Score'] > 3].copy()
    
    if len(significant_signals) == 0:
        print("No significant trading signals found")
        return
        
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


# Modify evaluate_signals to include the print
def evaluate_signals(data):
    scorer = SignalScore()
    data['Score'] = 0.0
    data['Confidence'] = 0.0
    data['Decision'] = 'Neutral'
    data['Reasoning'] = ''
    
    for i in range(1, len(data)):
        score, confidence, reasons = scorer.calculate_score(data, data, i)
        
        data.loc[i, 'Score'] = score
        data.loc[i, 'Confidence'] = confidence
        
        if score > 10 and confidence > 60:
            data.loc[i, 'Decision'] = 'Strong Long'
        elif score > 7 and confidence > 40:
            data.loc[i, 'Decision'] = 'Long'
        elif score < 3:
            data.loc[i, 'Decision'] = 'Neutral'
            
        data.loc[i, 'Reasoning'] = ' | '.join(reasons)
    
    # Print analysis after processing
    print_signal_analysis(data)
    return data

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
    print_signal_analysis(data)

if __name__ == "__main__":
    main()