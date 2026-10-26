import numpy as np

from .patterns import BULLISH_PATTERNS
from .price_action import ADX_TREND_THRESHOLD

DEFAULT_WEIGHTS = {
    'trendline_breakout': 5.0,
    'swing_breakout': 4.0,
    'adx': 3.0,
    'macd': 2.0,
    'moving_averages': 2.0,
    'volume': 1.5,
    'rsi': 1.0,
    'pattern': 2.5,
}

# (decision, score must exceed, confidence % must exceed); later rules win
DECISION_RULES = (
    ('Long', 7, 40),
    ('Strong Long', 10, 60),
)


class SignalScore:
    def __init__(self, weights=None, adx_threshold=ADX_TREND_THRESHOLD, decision_rules=DECISION_RULES):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.adx_threshold = adx_threshold
        self.decision_rules = decision_rules

    @property
    def max_score(self):
        return sum(self.weights.values())

    def conditions(self, data):
        """Map each weight to (condition met per bar, label)."""
        return {
            'trendline_breakout': (data['Trendline_Breakout'] == 'Above', 'Trendline breakout'),
            'swing_breakout': (data['Breakout'] == 'Above', 'Swing point breakout'),
            # ADX measures trend strength in either direction, so only reward uptrends
            'adx': ((data['ADX'] > self.adx_threshold) & (data['PLUS_DI'] > data['MINUS_DI']),
                    f'Strong uptrend - ADX>{self.adx_threshold:g}, +DI>-DI'),
            'macd': (data['MACD'] > data['MACD_signal'], 'MACD above signal line'),
            'moving_averages': ((data['close'] > data['SMA_50']) & (data['close'] > data['EMA_50']),
                                'Above key moving averages'),
            'volume': (data['volume'] > data['Volume_SMA_20'], 'Above-average volume'),
            'rsi': ((data['RSI'] > 30) & (data['RSI'] < 70), 'RSI in healthy range'),
            'pattern': (data['Pattern'].isin(BULLISH_PATTERNS), 'Bullish pattern'),
        }

    def evaluate(self, data):
        score = np.zeros(len(data))
        reasons = [[] for _ in range(len(data))]
        patterns = data['Pattern'].to_numpy()
        for component, (met, label) in self.conditions(data).items():
            weight = self.weights.get(component, 0.0)
            met = met.to_numpy(dtype=bool)
            score += np.where(met, weight, 0.0)
            for i in np.flatnonzero(met):
                text = f'{label}: {patterns[i]}' if component == 'pattern' else label
                reasons[i].append(f'{text} (weight: {weight})')

        confidence = score / self.max_score * 100
        decision = np.full(len(data), 'Neutral', dtype=object)
        for name, min_score, min_confidence in self.decision_rules:
            decision[(score > min_score) & (confidence > min_confidence)] = name

        data['Score'] = score
        data['Confidence'] = confidence
        data['Decision'] = decision
        data['Reasoning'] = [' | '.join(r) for r in reasons]
        return data


def evaluate_signals(data, adx_threshold=ADX_TREND_THRESHOLD):
    return SignalScore(adx_threshold=adx_threshold).evaluate(data)
