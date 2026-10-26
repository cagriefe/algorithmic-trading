import pandas as pd
import pytest

from algotrading.signals import SignalScore, evaluate_signals

# A bar where no scoring condition is met, and the overrides that switch each one on
NOTHING = {
    'Trendline_Breakout': 'No', 'Breakout': 'No', 'ADX': 10.0, 'PLUS_DI': 25.0, 'MINUS_DI': 15.0,
    'MACD': 0.0, 'MACD_signal': 0.5, 'close': 90.0, 'SMA_50': 100.0, 'EMA_50': 100.0,
    'volume': 0.5, 'Volume_SMA_20': 1.0, 'RSI': 80.0, 'Pattern': 'None',
}
CONDITIONS = {
    'trendline_breakout': {'Trendline_Breakout': 'Above'},
    'swing_breakout': {'Breakout': 'Above'},
    'adx': {'ADX': 30.0},
    'macd': {'MACD': 1.0},
    'moving_averages': {'close': 110.0},
    'volume': {'volume': 2.0},
    'rsi': {'RSI': 55.0},
    'pattern': {'Pattern': 'Double Bottom'},
}


def bar_with(*components, **overrides):
    row = dict(NOTHING)
    for component in components:
        row.update(CONDITIONS[component])
    row.update(overrides)
    return pd.DataFrame([row])


def test_all_conditions_give_the_maximum_score():
    data = evaluate_signals(bar_with(*CONDITIONS))
    reasoning = data['Reasoning'].iloc[0]

    assert data['Score'].iloc[0] == SignalScore().max_score == 21.0
    assert data['Confidence'].iloc[0] == pytest.approx(100.0)
    assert data['Decision'].iloc[0] == 'Strong Long'
    assert len(reasoning.split(' | ')) == len(CONDITIONS)
    assert 'Bullish pattern: Double Bottom (weight: 2.5)' in reasoning


def test_adx_only_counts_in_uptrends():
    data = evaluate_signals(bar_with('adx', PLUS_DI=10.0, MINUS_DI=30.0))

    assert data['Score'].iloc[0] == 0
    assert data['Reasoning'].iloc[0] == ''


def test_bearish_patterns_and_missing_indicators_score_nothing():
    data = evaluate_signals(bar_with(Pattern='Double Top', ADX=float('nan'), RSI=float('nan')))

    assert data['Score'].iloc[0] == 0
    assert data['Decision'].iloc[0] == 'Neutral'


@pytest.mark.parametrize('components, decision', [
    ((), 'Neutral'),
    (('swing_breakout', 'volume', 'rsi'), 'Neutral'),                            # 6.5
    (('trendline_breakout', 'swing_breakout'), 'Long'),                          # 9.0 -> 42.9%
    (('swing_breakout', 'macd', 'moving_averages', 'volume'), 'Long'),           # 9.5 -> 45.2%
    (('trendline_breakout', 'swing_breakout', 'adx', 'macd'), 'Strong Long'),    # 14.0 -> 66.7%
])
def test_decision_thresholds(components, decision):
    data = evaluate_signals(bar_with(*components))

    assert data['Score'].iloc[0] == sum(SignalScore().weights[c] for c in components)
    assert data['Decision'].iloc[0] == decision
