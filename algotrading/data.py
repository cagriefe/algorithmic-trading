import os
import time

import pandas as pd

KLINE_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                 'taker_buy_quote_asset_volume', 'ignore']
OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
TIMESTAMP_ALIASES = ('date', 'datetime', 'time', 'open_time')


def create_client(tld='com'):
    from binance.client import Client

    # Market data is public, so API keys are optional.
    return Client(os.getenv('API_KEY') or None, os.getenv('API_SECRET') or None, tld=tld)


def klines_to_frame(klines, now_ms=None):
    data = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    # Binance returns the still-forming candle last; using it would make signals repaint.
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    data = data[data['close_time'].astype('int64') < now_ms]
    return _normalize(data)


def fetch_klines(symbol, interval, start, end=None, client=None):
    client = client or create_client()
    return klines_to_frame(client.get_historical_klines(symbol, interval, start, end))


def load_csv(path):
    data = pd.read_csv(path)
    data.columns = [str(column).strip().lower() for column in data.columns]
    if 'timestamp' not in data.columns:
        alias = next((name for name in TIMESTAMP_ALIASES if name in data.columns), None)
        if alias:
            data = data.rename(columns={alias: 'timestamp'})
    missing = [column for column in OHLCV_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return _normalize(data)


def _normalize(data):
    data = data[OHLCV_COLUMNS].copy()
    for column in OHLCV_COLUMNS[1:]:
        data[column] = data[column].astype(float)
    if pd.api.types.is_numeric_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    else:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data.sort_values('timestamp').reset_index(drop=True)
