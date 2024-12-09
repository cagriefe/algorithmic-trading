# Algorithmic Trading System

A Python-based algorithmic trading system that implements technical analysis, pattern recognition, and signal generation for cryptocurrency trading on Binance.

## Features

### 1. Technical Analysis Components

#### Price Action Analysis
- Swing point detection for identifying market structure
- Support and resistance level identification 
- Breakout detection with volume confirmation

#### Moving Averages
- SMA (Simple Moving Average) calculation - 50 and 200 periods
- EMA (Exponential Moving Average) calculation - 50 and 200 periods
- Moving average crossover signals

#### Momentum Indicators
- RSI (Relative Strength Index) with overbought/oversold signals
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index) for trend strength

#### Bollinger Bands
- Dynamic support/resistance levels
- Volatility-based trading signals
- Mean reversion opportunities

### 2. Pattern Recognition

#### Chart Patterns
- Head and Shoulders / Inverse Head and Shoulders
- Double/Triple Tops and Bottoms
- Bull/Bear Flags
- Rectangle patterns
- Cup and Handle formations

#### Trendline Analysis
- Dynamic trendline detection
- Trendline breakout signals
- Support/resistance level tracking

### 3. Signal Generation

#### Scoring System
- Multi-factor signal scoring (0-100)
- Confidence level calculation
- Signal categorization:
  - Strong Long (Score > 10, Confidence > 60%)
  - Long (Score > 7, Confidence > 40%)
  - Neutral (Score < 3)

#### Signal Validation
- Volume confirmation
- Trend strength validation (ADX)
- Multiple timeframe confirmation

### 4. Data Management

- Historical data retrieval from Binance API
- Real-time price data processing
- Data normalization and preprocessing
- CSV export functionality

### 5. Visualization

- Interactive price charts with indicators
- Trend visualization
- Signal markers on charts
- Volume analysis subplot

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cagriefe/algorithmic-trading.git
cd algorithmic-trading

3. Install required packages:

pip install -r requirements.txt

## Configuration 

1. Create a .env file with your Binance API credentials:

API_KEY=your_api_key_here
API_SECRET=your_api_secret_here

## Usage

1. Run the main script:

python main.py

2. The script will:
Fetch historical data
Calculate technical indicators
Detect patterns
Generate trading signals
Display analysis results
Save data to CSV
Show visualization plots


### Output
The system generates:

Trading signals with confidence levels
Technical analysis visualization
Signal history in CSV format
Pattern detection results



## Future Improvements
Backtesting functionality
Risk management features
Portfolio optimization
Machine learning integration
Real-time trading execution
Performance analytics
Web interface for monitoring

## License 
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
