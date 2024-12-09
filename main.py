import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from dotenv import load_dotenv
import talib
from scipy.stats import linregress

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
    return data

def calculate_indicators(data):
    data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['close'], timeperiod=200)
    data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
    data['EMA_200'] = talib.EMA(data['close'], timeperiod=200)
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    return data

def find_swing_points(data, lookback=3):
    data['Swing_High'] = data['high'].rolling(window=2*lookback+1, center=True).apply(
        lambda x: x[lookback] if x[lookback] == max(x) else np.nan, raw=True
    )
    data['Swing_Low'] = data['low'].rolling(window=2*lookback+1, center=True).apply(
        lambda x: x[lookback] if x[lookback] == min(x) else np.nan, raw=True
    )
    return data

def detect_breakouts(data):
    data['Breakout'] = 'No'
    for i in range(1, len(data)):
        if not pd.isna(data['Swing_High'][i-1]) and data['close'][i] > data['Swing_High'][i-1]:
            data.loc[i, 'Breakout'] = 'Above'
        elif not pd.isna(data['Swing_Low'][i-1]) and data['close'][i] < data['Swing_Low'][i-1]:
            data.loc[i, 'Breakout'] = 'Below'
    return data

def calculate_trendlines(data):
    swing_highs = data.dropna(subset=['Swing_High'])
    swing_lows = data.dropna(subset=['Swing_Low'])
    
    if len(swing_highs) > 1:
        slope_high, intercept_high, _, _, _ = linregress(swing_highs.index, swing_highs['Swing_High'])
        data['Trendline_High'] = slope_high * data.index + intercept_high
    else:
        data['Trendline_High'] = np.nan
    
    if len(swing_lows) > 1:
        slope_low, intercept_low, _, _, _ = linregress(swing_lows.index, swing_lows['Swing_Low'])
        data['Trendline_Low'] = slope_low * data.index + intercept_low
    else:
        data['Trendline_Low'] = np.nan
    
    return data

def detect_trendline_breakouts(data):
    data['Trendline_Breakout'] = 'No'
    for i in range(1, len(data)):
        if not pd.isna(data['Trendline_High'][i-1]) and data['close'][i] > data['Trendline_High'][i-1]:
            data.loc[i, 'Trendline_Breakout'] = 'Above'
        elif not pd.isna(data['Trendline_Low'][i-1]) and data['close'][i] < data['Trendline_Low'][i-1]:
            data.loc[i, 'Trendline_Breakout'] = 'Below'
    return data

def plot_trend(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['close'], label='Close Price', color='blue', alpha=0.7)
    plt.plot(data['SMA_50'], label='SMA 50', color='orange', linestyle='--', alpha=0.7)
    plt.plot(data['SMA_200'], label='SMA 200', color='purple', linestyle='--', alpha=0.7)
    plt.scatter(data.index, data['Swing_High'], color='green', label='Swing High', marker='^', alpha=0.9)
    plt.scatter(data.index, data['Swing_Low'], color='red', label='Swing Low', marker='v', alpha=0.9)
    plt.plot(data['Trendline_High'], label='Trendline High', color='darkgreen', linestyle='-.', alpha=0.7)
    plt.plot(data['Trendline_Low'], label='Trendline Low', color='darkred', linestyle='-.', alpha=0.7)
    breakout_above = data[data['Breakout'] == 'Above']
    breakout_below = data[data['Breakout'] == 'Below']
    plt.scatter(breakout_above.index, breakout_above['close'], color='lime', label='Breakout Above', marker='o')
    plt.scatter(breakout_below.index, breakout_below['close'], color='darkred', label='Breakout Below', marker='o')
    trendline_breakout_above = data[data['Trendline_Breakout'] == 'Above']
    trendline_breakout_below = data[data['Trendline_Breakout'] == 'Below']
    plt.scatter(trendline_breakout_above.index, trendline_breakout_above['close'], color='cyan', label='Trendline Breakout Above', marker='x')
    plt.scatter(trendline_breakout_below.index, trendline_breakout_below['close'], color='magenta', label='Trendline Breakout Below', marker='x')
    plt.legend()
    plt.title('Swing Highs, Lows, Breakouts, and Indicators')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(alpha=0.3)
    plt.show()

def main():
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    start_str = '1 year ago UTC'

    data = get_historical_data(symbol, interval, start_str)
    data = calculate_indicators(data)
    data = find_swing_points(data)
    data = detect_breakouts(data)
    data = calculate_trendlines(data)
    data = detect_trendline_breakouts(data)
    
    # Print the latest data with indicators
    latest_data = data.tail()
    print(latest_data[['timestamp', 'close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_signal', 'ADX', 'Swing_High', 'Swing_Low', 'Breakout', 'Trendline_High', 'Trendline_Low', 'Trendline_Breakout']])

    # Plot the trend
    plot_trend(data)

if __name__ == "__main__":
    main()