import talib


def calculate_indicators(data):
    close, high, low = data['close'], data['high'], data['low']
    data['SMA_50'] = talib.SMA(close, timeperiod=50)
    data['SMA_200'] = talib.SMA(close, timeperiod=200)
    data['EMA_50'] = talib.EMA(close, timeperiod=50)
    data['EMA_200'] = talib.EMA(close, timeperiod=200)
    data['RSI'] = talib.RSI(close, timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    data['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    data['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['Volume_SMA_20'] = talib.SMA(data['volume'], timeperiod=20)
    return data
