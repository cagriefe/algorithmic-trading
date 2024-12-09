import os
import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
import talib

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

def determine_trend(data):
    latest_data = data.tail(1).iloc[0]
    if latest_data['SMA_50'] > latest_data['SMA_200']:
        return "Uptrend"
    elif latest_data['SMA_50'] < latest_data['SMA_200']:
        return "Downtrend"
    else:
        return "Sideways"

def main():
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    start_str = '1 year ago UTC'

    data = get_historical_data(symbol, interval, start_str)
    data = calculate_indicators(data)
    
    trend = determine_trend(data)
    print(f"The current trend for {symbol} is: {trend}")

    # Print the latest data with indicators
    latest_data = data.tail()
    print(latest_data[['timestamp', 'close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_signal', 'ADX']])

if __name__ == "__main__":
    main()