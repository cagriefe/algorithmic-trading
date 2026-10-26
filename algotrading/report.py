import numpy as np
import pandas as pd


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


def _signed(value):
    return '-' if pd.isna(value) else f'{value:+.2f}'


SCAN_COLUMNS = ['Symbol', 'Decision', 'Score', 'Close', 'Pattern', 'Bars since signal', 'RSI', 'Strategy %',
                'Buy & hold %', 'Max DD %', 'Trades']
SCAN_FORMATTERS = {
    'Score': lambda v: f'{v:.1f}',
    'Close': lambda v: f'{v:,.6g}',
    'Bars since signal': lambda v: f'{int(v)}',
    'RSI': lambda v: f'{v:.0f}',
    'Strategy %': _signed,
    'Buy & hold %': _signed,
    'Max DD %': _signed,
}


def print_scan_report(summary, errors):
    print('\n=== Market Scan (ranked by current signal) ===')
    if summary.empty:
        print('No markets could be analysed.')
    else:
        print(summary[SCAN_COLUMNS].to_string(index=False, formatters=SCAN_FORMATTERS, na_rep='-'))
    for symbol, error in errors.items():
        print(f'Skipped {symbol}: {error}')


def _summary_line(label, metrics):
    print(f'{label:<42}return {metrics["Total return %"]:+7.2f}%   max DD {metrics["Max drawdown %"]:7.2f}%   '
          f'Sharpe {_format_metric(metrics["Sharpe ratio"])}')


def print_walk_forward_report(result):
    print(f'\n=== Walk-Forward Optimisation ({len(result.folds)} folds, objective: {result.objective}) ===')
    print('Each fold picks the best parameters on all earlier data and then trades the next, unseen period.\n')
    print(result.folds.to_string(index=False, float_format=lambda x: f'{x:.2f}', na_rep='-'))
    print()
    _summary_line('Out-of-sample, re-optimised every fold:', result.oos_metrics)
    _summary_line('Default parameters, same periods:', result.baseline_metrics)
    buy_and_hold = (result.buy_and_hold_equity.iloc[-1] / result.buy_and_hold_equity.iloc[0] - 1) * 100
    print(f'{"Buy & hold, same periods:":<42}return {buy_and_hold:+7.2f}%')
    print(f'Out-of-sample trades: {result.oos_metrics["Trades"]}')
    if np.isfinite(result.efficiency):
        print(f'Walk-forward efficiency: {result.efficiency:.0f}% of the in-sample return per bar was kept '
              'out of sample (above ~50% suggests more than curve fitting)')
    else:
        print('Walk-forward efficiency: n/a (the in-sample results were not profitable)')

    params = result.recommended
    print(f'\nBest on the full history (in-sample, validate before trusting): stop {params["stop_loss_atr"]:g} ATR, '
          f'target {params["take_profit_atr"]:g} ATR, entry on {params["entry"]}, ADX > {params["adx_threshold"]:g}')
    print('\nTop 5 parameter sets on the full history:')
    print(result.grid.head(5).to_string(index=False, float_format=lambda x: f'{x:.2f}', na_rep='-'))


def print_edge_report(table):
    print('\n=== Signal Edge (what happened after each signal) ===')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.2f}', na_rep='-'))
