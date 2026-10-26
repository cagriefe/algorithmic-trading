import re
import time

import pandas as pd
import requests

KLINE_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                 'taker_buy_quote_asset_volume', 'ignore']
OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
TIMESTAMP_ALIASES = ('date', 'datetime', 'time', 'open_time')
INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

API_HOSTS = {'com': 'https://api.binance.com', 'us': 'https://api.binance.us'}
# Market-data-only mirror without geo-restrictions; api.binance.com answers 451 in some regions (e.g. US clouds)
PUBLIC_DATA_HOST = 'https://data-api.binance.vision'
KLINES_LIMIT = 1000
RELATIVE_DATE = re.compile(r'^(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago(\s+utc)?$', re.IGNORECASE)


class BinanceError(RuntimeError):
    pass


def to_milliseconds(value, now=None):
    """Epoch milliseconds for '2024-01-01', '6 months ago UTC', 'now', a datetime or milliseconds."""
    if isinstance(value, (int, float)):
        return int(value)
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    text = str(value).strip()
    match = RELATIVE_DATE.match(text)
    if text.lower() == 'now':
        timestamp = now
    elif match:
        timestamp = now - pd.DateOffset(**{f'{match.group(2).lower()}s': int(match.group(1))})
    else:
        timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize('UTC')
    return int(timestamp.timestamp() * 1000)


def _error_message(response):
    try:
        return response.json().get('msg', response.text)
    except ValueError:
        return response.text


class BinanceClient:
    """Minimal client for Binance's public market-data endpoints (no API keys needed)."""

    def __init__(self, tld='com', session=None, timeout=15, retries=2):
        self.base_url = API_HOSTS.get(tld, f'https://api.binance.{tld}')
        self.session = session or requests.Session()
        self.timeout = timeout
        self.retries = retries

    def _request(self, path, params):
        return self.session.get(self.base_url + path, params=params, timeout=self.timeout)

    def _get(self, path, params=None):
        for attempt in range(self.retries + 1):
            response = self._request(path, params)
            if response.status_code == 451 and self.base_url == API_HOSTS['com']:
                # Restricted location: use the public market-data mirror from now on
                self.base_url = PUBLIC_DATA_HOST
                response = self._request(path, params)
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < self.retries:
                time.sleep(2 ** attempt)
                continue
            if not response.ok:
                raise BinanceError(f'Binance API error {response.status_code}: {_error_message(response)}')
            return response.json()

    def klines(self, symbol, interval, start, end=None):
        start_ms, end_ms = to_milliseconds(start), to_milliseconds(end or 'now')
        rows = []
        while start_ms < end_ms:
            batch = self._get('/api/v3/klines', {'symbol': symbol.upper(), 'interval': interval,
                                                  'startTime': start_ms, 'endTime': end_ms, 'limit': KLINES_LIMIT})
            rows.extend(batch)
            if len(batch) < KLINES_LIMIT:
                break
            start_ms = batch[-1][0] + 1
        return rows

    def tickers_24h(self):
        return self._get('/api/v3/ticker/24hr')


def create_client(tld='com'):
    return BinanceClient(tld=tld)


def klines_to_frame(klines, now_ms=None):
    data = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    # Binance returns the still-forming candle last; using it would make signals repaint.
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    data = data[data['close_time'].astype('int64') < now_ms]
    return _normalize(data)


def fetch_klines(symbol, interval, start, end=None, client=None):
    client = client or create_client()
    return klines_to_frame(client.klines(symbol, interval, start, end))


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
