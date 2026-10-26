import pandas as pd
import pytest

from algotrading.analysis import run_analysis
from algotrading.edge import BASELINE, signal_edge


def bars(close, decisions):
    """Bars where no scoring condition is met, with the given closes and decisions."""
    return pd.DataFrame({
        'close': close, 'Decision': decisions, 'Pattern': 'None', 'Breakout': 'No', 'Trendline_Breakout': 'No',
        'ADX': 10.0, 'PLUS_DI': 1.0, 'MINUS_DI': 2.0, 'MACD': 0.0, 'MACD_signal': 1.0, 'SMA_50': 1e9,
        'EMA_50': 1e9, 'volume': 1.0, 'Volume_SMA_20': 2.0, 'RSI': 80.0,
    })


def test_forward_returns_hit_rate_and_edge():
    data = bars([100, 100, 120, 120, 90, 90], ['Neutral', 'Strong Long', 'Neutral', 'Long', 'Neutral', 'Neutral'])
    table = signal_edge(data, horizons=(1, 2, 3)).set_index('Signal')
    strong = table.loc['Strong Long']

    # 2-bar forward returns: +20%, +20%, -25%, -25%, n/a, n/a -> baseline -2.5%
    assert table.loc[BASELINE, 'Avg 2-bar %'] == pytest.approx(-2.5)
    assert strong['Count'] == 1
    assert [strong['Avg 1-bar %'], strong['Avg 2-bar %'], strong['Avg 3-bar %']] == pytest.approx([20, 20, -10])
    assert strong['Hit rate 2-bar %'] == 100
    assert strong['Edge 2-bar %'] == pytest.approx(22.5)
    assert table.loc['Long or Strong Long', 'Hit rate 2-bar %'] == 50


def test_edge_table_covers_every_signal(market_data):
    data = run_analysis(market_data)
    table = signal_edge(data)

    assert table['Signal'].iloc[0] == BASELINE
    assert table['Count'].iloc[0] == len(data)
    assert {'Strong Long', 'Trendline breakout', 'Bullish pattern', 'Double Bottom', 'Head and Shoulders',
            'Breakdown below support'} <= set(table['Signal'])
    longs = table.loc[table['Signal'] == 'Long or Strong Long', 'Count'].item()
    assert longs == data['Decision'].isin(['Long', 'Strong Long']).sum()
