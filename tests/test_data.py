import json

import pandas as pd
import pytest

from algotrading.data import (KLINES_LIMIT, PUBLIC_DATA_HOST, BinanceClient, BinanceError, klines_to_frame,
                              load_csv, to_milliseconds)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def kline(open_time_ms, close):
    return [open_time_ms, '1.0', '2.0', '0.5', str(close), '100.0', open_time_ms + 59_999,
            '0', 10, '0', '0', '0']


def test_klines_to_frame_converts_types_and_drops_forming_candle():
    klines = [kline(0, 1.5), kline(60_000, 1.6), kline(120_000, 1.7)]
    data = klines_to_frame(klines, now_ms=150_000)  # the last candle closes at 179_999

    assert list(data.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert data['close'].tolist() == [1.5, 1.6]
    assert data['timestamp'].iloc[1] == pd.Timestamp('1970-01-01 00:01:00')
    assert all(data[column].dtype == float for column in ['open', 'high', 'low', 'close', 'volume'])


@pytest.mark.parametrize('value, expected', [
    ('2024-01-01', '2024-01-01'),
    ('6 months ago UTC', '2024-01-01'),
    ('2 weeks ago', '2024-06-17'),
    ('1 year ago UTC', '2023-07-01'),
    ('now', '2024-07-01'),
])
def test_to_milliseconds_understands_dates_and_relative_periods(value, expected):
    now = pd.Timestamp('2024-07-01', tz='UTC')
    assert to_milliseconds(value, now=now) == int(pd.Timestamp(expected, tz='UTC').timestamp() * 1000)


def test_to_milliseconds_keeps_epoch_milliseconds():
    assert to_milliseconds(1_700_000_000_000) == 1_700_000_000_000


def test_klines_are_paginated():
    first = [kline(i * 60_000, 1.0) for i in range(KLINES_LIMIT)]
    second = [kline((KLINES_LIMIT + i) * 60_000, 1.0) for i in range(5)]
    session = FakeSession(FakeResponse(200, first), FakeResponse(200, second))

    rows = BinanceClient(session=session).klines('btcusdt', '1m', 0, end=10**12)

    assert len(rows) == KLINES_LIMIT + 5
    assert session.calls[0][1]['symbol'] == 'BTCUSDT'
    assert session.calls[1][1]['startTime'] == first[-1][0] + 1


def test_restricted_location_falls_back_to_public_data_mirror():
    session = FakeSession(FakeResponse(451, {'msg': 'restricted'}), FakeResponse(200, [1]), FakeResponse(200, [2]))
    client = BinanceClient(session=session)

    assert client.tickers_24h() == [1]
    assert client.tickers_24h() == [2]
    assert [url for url, _ in session.calls] == [
        'https://api.binance.com/api/v3/ticker/24hr',
        f'{PUBLIC_DATA_HOST}/api/v3/ticker/24hr',
        f'{PUBLIC_DATA_HOST}/api/v3/ticker/24hr',
    ]


def test_rate_limits_are_retried(monkeypatch):
    monkeypatch.setattr('algotrading.data.time.sleep', lambda seconds: None)
    session = FakeSession(FakeResponse(429, {'msg': 'slow down'}), FakeResponse(200, ['ok']))

    assert BinanceClient(session=session).tickers_24h() == ['ok']


def test_api_errors_are_reported():
    session = FakeSession(FakeResponse(400, {'code': -1121, 'msg': 'Invalid symbol.'}))

    with pytest.raises(BinanceError, match='400: Invalid symbol.'):
        BinanceClient(session=session).klines('NOPE', '1d', '2024-01-01', '2024-02-01')


def test_load_csv_accepts_common_column_names(tmp_path):
    csv_path = tmp_path / 'prices.csv'
    csv_path.write_text('Date,Open,High,Low,Close,Volume,Extra\n'
                        '2024-01-02,2,3,1,2.5,10,x\n'
                        '2024-01-01,1,2,0.5,1.5,20,y\n')
    data = load_csv(csv_path)

    assert list(data.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert data['close'].tolist() == [1.5, 2.5]
    assert data['timestamp'].iloc[0] == pd.Timestamp('2024-01-01')


def test_load_csv_reports_missing_columns(tmp_path):
    csv_path = tmp_path / 'prices.csv'
    csv_path.write_text('timestamp,open,close\n2024-01-01,1,2\n')

    with pytest.raises(ValueError, match='high, low, volume'):
        load_csv(csv_path)
