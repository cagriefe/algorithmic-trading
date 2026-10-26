import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, DateFormatter

from .price_action import display_trendline


def plot_analysis(data, title='Technical Analysis Overview', backtest=None):
    """Draw price, indicators, breakouts, signals and optional backtest results; returns the figure."""
    data = data.set_index('timestamp')
    if backtest is None:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1.3]})
    ax1, ax2 = axes[0], axes[1]

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
    trendline_high = display_trendline(data, data['Trendline_High'], data['Last_Swing_High'])
    trendline_low = display_trendline(data, data['Trendline_Low'], data['Last_Swing_Low'])
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
    ax2.set_ylabel('Volume')

    if backtest is not None:
        trades = backtest.trades
        ax1.scatter(trades['entry_time'], trades['entry_price'], color='lime', edgecolors='black',
                    marker='P', s=90, zorder=5, label='Trade Entry')
        ax1.scatter(trades['exit_time'], trades['exit_price'], color='black', marker='X', s=70,
                    zorder=5, label='Trade Exit')

        ax3 = axes[2]
        buy_and_hold = backtest.metrics['Initial capital'] * data['close'] / data['open'].iloc[0]
        ax3.plot(backtest.equity.index, backtest.equity, color='green', label='Strategy')
        ax3.plot(data.index, buy_and_hold, color='gray', linestyle='--', label='Buy & Hold')
        ax3.set_ylabel('Equity')
        ax3.legend(loc='upper left')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    axes[-1].set_xlabel('Date')

    ax1.set_title(title, pad=20)
    ax1.set_ylabel('Price')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    return fig


def plot_walk_forward(result, title='Walk-forward optimisation'):
    """Out-of-sample equity, per-fold returns and the parameter heatmap of a walk-forward run."""
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    equity_ax, folds_ax, heat_ax = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

    equity_ax.plot(result.oos_equity.index, result.oos_equity, color='green', label='Re-optimised every fold')
    equity_ax.plot(result.baseline_equity.index, result.baseline_equity, color='orange', label='Default parameters')
    equity_ax.plot(result.buy_and_hold_equity.index, result.buy_and_hold_equity, color='gray', linestyle='--',
                   label='Buy & hold')
    starts = [result.oos_equity.index[0]]
    for test in result.folds['Test'].iloc[1:]:
        starts.append(pd.Timestamp(test.split(' -> ')[0]))
    for start in starts:
        equity_ax.axvline(start, color='black', alpha=0.25, linestyle=':')
    equity_ax.set_title('Out-of-sample equity (dotted lines: start of each test period)')
    equity_ax.set_ylabel('Equity')
    equity_ax.grid(True, alpha=0.3)
    equity_ax.legend(loc='upper left')
    equity_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    folds = result.folds
    positions = np.arange(len(folds))
    for offset, column, color in [(-0.27, 'OOS return %', 'green'), (0, 'Default OOS return %', 'orange'),
                                  (0.27, 'Buy & hold %', 'gray')]:
        folds_ax.bar(positions + offset, folds[column], width=0.27, color=color, label=column.replace(' %', ''))
    folds_ax.axhline(0, color='black', linewidth=0.8)
    folds_ax.set_xticks(positions, [f'Fold {k}' for k in folds['Fold']])
    folds_ax.set_ylabel('Return %')
    folds_ax.set_title('Return per test period')
    folds_ax.grid(True, axis='y', alpha=0.3)
    folds_ax.legend(fontsize=8)

    heat = result.grid.replace(-np.inf, np.nan).pivot_table(
        index='stop_loss_atr', columns='take_profit_atr', values='Objective', aggfunc='max')
    image = heat_ax.imshow(heat.to_numpy(), cmap='RdYlGn', aspect='auto', origin='lower')
    heat_ax.set_xticks(range(len(heat.columns)), [f'{v:g}' for v in heat.columns])
    heat_ax.set_yticks(range(len(heat.index)), [f'{v:g}' for v in heat.index])
    for (row, column), value in np.ndenumerate(heat.to_numpy()):
        if np.isfinite(value):
            heat_ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=9)
    heat_ax.set_xlabel('Take profit (ATR)')
    heat_ax.set_ylabel('Stop loss (ATR)')
    heat_ax.set_title(f'Full-history {result.objective} by stop and target (best entry/ADX)')
    fig.colorbar(image, ax=heat_ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
