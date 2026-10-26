import numpy as np


def _print_signal(row):
    print(f"Date: {row['timestamp']}")
    print(f"Score: {row['Score']:.2f}")
    print(f"Confidence: {row['Confidence']:.1f}%")
    print(f"Decision: {row['Decision']}")
    print(f"Close Price: {row['close']}")
    print(f"Reasoning: {row['Reasoning'] or '-'}")
    print('-' * 80)


def print_signal_analysis(data, top=3):
    print('\n=== Trading Signal Analysis ===')

    print('\nCURRENT SIGNAL:')
    print('-' * 80)
    _print_signal(data.iloc[-1])

    significant_signals = data[data['Score'] > 3]
    if significant_signals.empty:
        print('No significant trading signals found')
        return

    print('\nHISTORICAL BEST SIGNALS:')
    print(f'Total signals analyzed: {len(data)}')
    print(f'Significant signals found: {len(significant_signals)}')
    print('-' * 80)
    for _, row in significant_signals.sort_values('Score', ascending=False).head(top).iterrows():
        print()
        _print_signal(row)


def _format_metric(value):
    if isinstance(value, float):
        return f'{value:,.2f}' if np.isfinite(value) else ('inf' if value > 0 else '-')
    return str(value)


def print_backtest_report(result, last_trades=10):
    print('\n=== Backtest Results ===')
    print('-' * 80)
    for name, value in result.metrics.items():
        print(f'{name + ":":<22}{_format_metric(value)}')
    print('-' * 80)

    trades = result.trades
    if trades.empty:
        print('No trades were opened.')
        return
    reasons = trades['exit_reason'].value_counts()
    print('Exit reasons: ' + ', '.join(f'{reason}: {count}' for reason, count in reasons.items()))
    print(f'\nLast {min(last_trades, len(trades))} trades:')
    columns = ['entry_time', 'entry_price', 'exit_time', 'exit_price', 'exit_reason', 'return_pct']
    print(trades[columns].tail(last_trades).to_string(index=False, float_format=lambda x: f'{x:.6g}'))


def print_monte_carlo(stats):
    if stats is None:
        return
    print(f"\n=== Monte Carlo ({stats['Simulations']:,} resampled trade sequences) ===")
    print('-' * 80)
    for name, value in stats.items():
        if name != 'Simulations':
            print(f'{name + ":":<32}{_format_metric(value)}')
    print('-' * 80)


def print_edge_report(table):
    print('\n=== Signal Edge (what happened after each signal) ===')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.2f}', na_rep='-'))
