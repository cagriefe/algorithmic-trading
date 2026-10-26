import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter


def _readable_trendline(data, line, level):
    # Break the line where a new pivot is confirmed and hide far extrapolations
    new_segment = level.ne(level.shift(1))
    too_far = (line - data['close']).abs() > 3 * data['ATR']
    return line.mask(new_segment | too_far)


def plot_analysis(data, title='Technical Analysis Overview'):
    """Draw price, indicators, breakouts and signals; returns the matplotlib figure."""
    data = data.set_index('timestamp')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Main price plot
    ax1.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', color='orange', linestyle='--', alpha=0.7)
    ax1.plot(data.index, data['SMA_200'], label='SMA 200', color='purple', linestyle='--', alpha=0.7)

    # Bollinger Bands
    ax1.plot(data.index, data['BB_upper'], 'k--', alpha=0.3)
    ax1.plot(data.index, data['BB_middle'], 'k--', alpha=0.3)
    ax1.plot(data.index, data['BB_lower'], 'k--', alpha=0.3)
    ax1.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1, color='gray', label='Bollinger Bands')

    # Swing points and trendlines
    ax1.scatter(data.index, data['Swing_High'], color='green', label='Swing High', marker='^', alpha=0.9)
    ax1.scatter(data.index, data['Swing_Low'], color='red', label='Swing Low', marker='v', alpha=0.9)
    trendline_high = _readable_trendline(data, data['Trendline_High'], data['Last_Swing_High'])
    trendline_low = _readable_trendline(data, data['Trendline_Low'], data['Last_Swing_Low'])
    ax1.plot(data.index, trendline_high, label='Trendline High', color='darkgreen', linestyle='-.', alpha=0.7)
    ax1.plot(data.index, trendline_low, label='Trendline Low', color='darkred', linestyle='-.', alpha=0.7)

    markers = [
        ('Breakout', 'Above', dict(color='lime', label='Breakout Above', marker='o')),
        ('Breakout', 'Below', dict(color='darkred', label='Breakout Below', marker='o')),
        ('Trendline_Breakout', 'Above', dict(color='cyan', label='Trendline Breakout Above', marker='x')),
        ('Trendline_Breakout', 'Below', dict(color='magenta', label='Trendline Breakout Below', marker='x')),
        ('Decision', 'Long', dict(color='gold', label='Long Signal', marker='*', s=100)),
        ('Decision', 'Strong Long', dict(color='green', label='Strong Long Signal', marker='*', s=150)),
    ]
    for column, value, style in markers:
        rows = data[data[column] == value]
        ax1.scatter(rows.index, rows['close'], **style)

    # Volume subplot
    ax2.bar(data.index, data['volume'], color='blue', alpha=0.3, label='Volume')

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.set_title(title, pad=20)
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    return fig
