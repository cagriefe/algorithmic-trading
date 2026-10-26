import pandas as pd
import pytest

from algotrading.analysis import run_analysis

# Pivots are drawn on the bar where they happened, which needs later bars; signals never use them
HINDSIGHT_COLUMNS = ['Swing_High', 'Swing_Low']


def test_pipeline_adds_signal_columns(market_data):
    data = run_analysis(market_data)

    assert len(data) == len(market_data)
    for column in ['RSI', 'ADX', 'Last_Swing_High', 'Trendline_High', 'Breakout', 'Pattern', 'Score', 'Decision']:
        assert column in data
    assert set(data['Decision']) <= {'Neutral', 'Long', 'Strong Long'}
    assert (data['Decision'] != 'Neutral').any()


@pytest.mark.parametrize('cut', [150, 333, 480, 599])
def test_signals_do_not_use_future_data(market_data, cut):
    full = run_analysis(market_data).drop(columns=HINDSIGHT_COLUMNS)
    partial = run_analysis(market_data.iloc[:cut]).drop(columns=HINDSIGHT_COLUMNS)

    pd.testing.assert_frame_equal(full.iloc[:cut], partial, check_dtype=False)
