import pandas as pd
import pytest

from algotrading.data import klines_to_frame, load_csv


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
