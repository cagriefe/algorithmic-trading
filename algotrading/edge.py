import numpy as np
import pandas as pd

from .patterns import BEARISH_PATTERNS, BULLISH_PATTERNS
from .signals import SignalScore

BASELINE = 'All bars (baseline)'


def signal_edge(data, horizons=(5, 10, 20), scorer=None):
    """How price moved after each signal compared with an average bar.

    For every decision, scoring condition and chart pattern this reports how often it occurred,
    the average forward return over each horizon and, for the middle horizon, the hit rate
    (share of positive returns) and the edge over the baseline. Forward returns look into the
    future on purpose: this evaluates signals after the fact and is never used to create them.
    """
    scorer = scorer or SignalScore()
    decision = data['Decision']
    events = {
        BASELINE: pd.Series(True, index=data.index),
        'Long or Strong Long': decision.isin(['Long', 'Strong Long']),
        'Strong Long': decision == 'Strong Long',
    }
    for met, label in scorer.conditions(data).values():
        events[label] = met
    for pattern in BULLISH_PATTERNS + BEARISH_PATTERNS:
        events[pattern] = data['Pattern'] == pattern
    events['Breakdown below support'] = (data['Breakout'] == 'Below') | (data['Trendline_Breakout'] == 'Below')

    close = data['close']
    forward = {h: (close.shift(-h) / close - 1) * 100 for h in horizons}
    main = horizons[len(horizons) // 2]
    baseline = forward[main].mean()

    rows = []
    for name, mask in events.items():
        mask = mask.to_numpy(dtype=bool)
        row = {'Signal': name, 'Count': int(mask.sum())}
        for h in horizons:
            row[f'Avg {h}-bar %'] = forward[h][mask].mean()
        outcomes = forward[main][mask].dropna()
        row[f'Hit rate {main}-bar %'] = (outcomes > 0).mean() * 100 if len(outcomes) else np.nan
        row[f'Edge {main}-bar %'] = row[f'Avg {main}-bar %'] - baseline
        rows.append(row)
    return pd.DataFrame(rows)
