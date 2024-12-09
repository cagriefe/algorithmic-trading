class SignalScore:
    def __init__(self):
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
        
    def calculate_score(self, signals, data, index):
        score = 0
        reasons = []
        
        # Trendline Breakouts (Highest Priority)
        if data['Trendline_Breakout'][index] == 'Above':
            score += self.weights['trendline_breakout']
            reasons.append('Trendline breakout (weight: 5.0)')
            
        # Swing Point Breakouts
        if data['Breakout'][index] == 'Above':
            score += self.weights['swing_breakout']
            reasons.append('Swing point breakout (weight: 4.0)')
            
        # ADX Trend Strength
        if data['ADX'][index] > 25:
            score += self.weights['adx']
            reasons.append('Strong trend - ADX>25 (weight: 3.0)')
            
        # MACD Signal
        if data['MACD'][index] > data['MACD_signal'][index]:
            score += self.weights['macd']
            reasons.append('MACD bullish crossover (weight: 2.0)')
            
        # Moving Averages
        if (data['close'][index] > data['SMA_50'][index] and 
            data['close'][index] > data['EMA_50'][index]):
            score += self.weights['moving_averages']
            reasons.append('Above key moving averages (weight: 2.0)')
            
        # Volume Confirmation
        if data['volume'][index] > data['volume'][index-1]:
            score += self.weights['volume']
            reasons.append('Increasing volume (weight: 1.5)')
            
        # RSI
        if 30 < data['RSI'][index] < 70:
            score += self.weights['rsi']
            reasons.append('RSI in healthy range (weight: 1.0)')
            
        # Chart Patterns
        bullish_patterns = ['Inverse Head and Shoulders', 'Double Bottom', 
                          'Triple Bottom', 'Bull Flag', 'Cup and Handle']
        if data['Pattern'][index] in bullish_patterns:
            score += self.weights['pattern']
            reasons.append(f'Bullish pattern: {data["Pattern"][index]} (weight: 2.5)')
            
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
        score, confidence, reasons = scorer.calculate_score(data, data, i)
        
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
