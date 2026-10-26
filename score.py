BULLISH_PATTERNS = ('Inverse Head and Shoulders', 'Double Bottom', 'Triple Bottom',
                    'Bull Flag', 'Cup and Handle')


class SignalScore:
    def __init__(self, adx_threshold=25):
        self.adx_threshold = adx_threshold
        self.weights = {
            'trendline_breakout': 5.0,
            'swing_breakout': 4.0,
            'adx': 3.0,
            'macd': 2.0,
            'moving_averages': 2.0,
            'volume': 1.5,
            'rsi': 1.0,
            'pattern': 2.5
        }

    def calculate_score(self, data, index):
        row = data.iloc[index]
        score = 0
        reasons = []

        def add(component, reason):
            nonlocal score
            score += self.weights[component]
            reasons.append(f'{reason} (weight: {self.weights[component]})')

        # Trendline Breakouts (Highest Priority)
        if row['Trendline_Breakout'] == 'Above':
            add('trendline_breakout', 'Trendline breakout')

        # Swing Point Breakouts
        if row['Breakout'] == 'Above':
            add('swing_breakout', 'Swing point breakout')

        # ADX measures trend strength in either direction, so only reward uptrends
        if row['ADX'] > self.adx_threshold and row['PLUS_DI'] > row['MINUS_DI']:
            add('adx', f'Strong uptrend - ADX>{self.adx_threshold}, +DI>-DI')

        # MACD Signal
        if row['MACD'] > row['MACD_signal']:
            add('macd', 'MACD above signal line')

        # Moving Averages
        if row['close'] > row['SMA_50'] and row['close'] > row['EMA_50']:
            add('moving_averages', 'Above key moving averages')

        # Volume Confirmation
        if row['volume'] > row['Volume_SMA_20']:
            add('volume', 'Above-average volume')

        # RSI
        if 30 < row['RSI'] < 70:
            add('rsi', 'RSI in healthy range')

        # Chart Patterns
        if row['Pattern'] in BULLISH_PATTERNS:
            add('pattern', f'Bullish pattern: {row["Pattern"]}')

        # Calculate confidence level
        max_score = sum(self.weights.values())
        confidence = (score / max_score) * 100

        return score, confidence, reasons

def evaluate_signals(data):
    scorer = SignalScore()
    data['Score'] = 0.0
    data['Confidence'] = 0.0
    data['Decision'] = 'Neutral'
    data['Reasoning'] = ''

    for i in range(1, len(data)):
        score, confidence, reasons = scorer.calculate_score(data, i)

        data.loc[i, 'Score'] = score
        data.loc[i, 'Confidence'] = confidence

        # Decision making based on score and confidence
        if score > 10 and confidence > 60:
            data.loc[i, 'Decision'] = 'Strong Long'
        elif score > 7 and confidence > 40:
            data.loc[i, 'Decision'] = 'Long'
        elif score < 3:
            data.loc[i, 'Decision'] = 'Neutral'

        data.loc[i, 'Reasoning'] = ' | '.join(reasons)

    return data
