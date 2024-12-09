import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from dotenv import load_dotenv
import talib
from scipy.stats import linregress
from matplotlib.dates import DateFormatter, AutoDateLocator

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
    for i in range(1, len(data)):
        # Example: Detecting Double Top pattern
        if i > 5 and data['close'][i] < data['close'][i-1] and data['close'][i-1] > data['close'][i-2] and data['close'][i-2] < data['close'][i-3] and data['close'][i-3] > data['close'][i-4] and data['close'][i-4] < data['close'][i-5]:
            data.loc[i, 'Pattern'] = 'Double Top'
        # Example: Detecting Double Bottom pattern
        if i > 5 and data['close'][i] > data['close'][i-1] and data['close'][i-1] < data['close'][i-2] and data['close'][i-2] > data['close'][i-3] and data['close'][i-3] < data['close'][i-4] and data['close'][i-4] > data['close'][i-5]:
            data.loc[i, 'Pattern'] = 'Double Bottom'
    return data

def evaluate_signals(data):
    data['Signals'] = ''
    data['Decision'] = 'Neutral'
    data['Reasoning'] = ''

    for i in range(1, len(data)):
        signals = []
        reasoning = []

        # Price crosses above key moving averages
        if data['close'][i] > data['SMA_50'][i]:
            signals.append('Price above SMA_50')
        if data['close'][i] > data['EMA_50'][i]:
            signals.append('Price above EMA_50')

        # RSI within bullish range
        if 30 < data['RSI'][i] < 70:
            signals.append('RSI in bullish range')

        # MACD bullish crossover
        if data['MACD'][i] > data['MACD_signal'][i]:
            signals.append('MACD bullish crossover')

        # Positive breakouts
        if data['Breakout'][i] == 'Above':
            signals.append('Positive breakout above swing high')
        if data['Trendline_Breakout'][i] == 'Above':
            signals.append('Positive breakout above trendline')

        # ADX indicates strong trend
        if data['ADX'][i] > 25:
            signals.append('ADX indicates strong trend')

        # Bullish patterns
        if data['Pattern'][i] == 'Double Bottom':
            signals.append('Bullish pattern: Double Bottom')

        # Volume and ATR/Bollinger Bands validation
        if data['volume'][i] > data['volume'][i-1] and data['ATR'][i] > data['ATR'][i-1]:
            signals.append('High volume and ATR increase')
        if data['close'][i] > data['BB_upper'][i]:
            signals.append('Price above Bollinger Bands upper band')

        # Aggregate signals into a decision
        if signals:
            data.at[i, 'Signals'] = ', '.join(signals)
            data.at[i, 'Decision'] = 'Long'
            reasoning.append(' and '.join(signals))
            data.at[i, 'Reasoning'] = 'Long signal due to: ' + '; '.join(reasoning)
        else:
            data.at[i, 'Reasoning'] = 'No significant signals detected'

    return data

def plot_trend(data):
    if data.empty:
        print("No data to plot.")
        return

    # Convert timestamps to datetime and set as index for better x-axis handling
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
    plt.plot(data.index, data['SMA_50'], label='SMA 50', color='orange', linestyle='--', alpha=0.7)
    plt.plot(data.index, data['SMA_200'], label='SMA 200', color='purple', linestyle='--', alpha=0.7)
    plt.scatter(data.index, data['Swing_High'], color='green', label='Swing High', marker='^', alpha=0.9)
    plt.scatter(data.index, data['Swing_Low'], color='red', label='Swing Low', marker='v', alpha=0.9)
    plt.plot(data.index, data['Trendline_High'], label='Trendline High', color='darkgreen', linestyle='-.', alpha=0.7)
    plt.plot(data.index, data['Trendline_Low'], label='Trendline Low', color='darkred', linestyle='-.', alpha=0.7)

    # Highlight breakouts
    breakout_above = data[data['Breakout'] == 'Above']
    breakout_below = data[data['Breakout'] == 'Below']
    plt.scatter(breakout_above.index, breakout_above['close'], color='lime', label='Breakout Above', marker='o')
    plt.scatter(breakout_below.index, breakout_below['close'], color='darkred', label='Breakout Below', marker='o')

    # Highlight trendline breakouts
    trendline_breakout_above = data[data['Trendline_Breakout'] == 'Above']
    trendline_breakout_below = data[data['Trendline_Breakout'] == 'Below']
    plt.scatter(trendline_breakout_above.index, trendline_breakout_above['close'], color='cyan', label='Trendline Breakout Above', marker='x')
    plt.scatter(trendline_breakout_below.index, trendline_breakout_below['close'], color='magenta', label='Trendline Breakout Below', marker='x')

    # Highlight long signals
    long_signals = data[data['Decision'] == 'Long']
    plt.scatter(long_signals.index, long_signals['close'], color='gold', label='Long Signal', marker='*', s=100)

    # X-Axis formatting
    plt.gca().xaxis.set_major_locator(AutoDateLocator())  # Automatic date tick spacing
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Format: YYYY-MM-DD
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Chart details
    plt.legend()
    plt.title('Swing Highs, Lows, Breakouts, and Indicators')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(alpha=0.3)
    plt.tight_layout()  # Prevent label cutoff
    plt.show()

def main():
    symbol = 'ETHUSDT'
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
    
    # Save the latest data with indicators to a CSV file
    latest_data = data.tail()
    latest_data.to_csv('latest_data_analysis.csv', index=False)

    # Plot the trend
    plot_trend(data)

if __name__ == "__main__":
    main()