import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .analysis import run_analysis
from .backtest import BacktestConfig, equity_stats, run_backtest
from .price_action import ADX_TREND_THRESHOLD, SWING_LOOKBACK

DEFAULT_GRID = {
    'stop_loss_atr': (1.5, 2.0, 3.0),
    'take_profit_atr': (2.0, 3.0, 4.0, 6.0),
    'entry': ('Long', 'Strong Long'),
    'adx_threshold': (20.0, 25.0, 30.0),
}
ENTRY_DECISIONS = {'Long': ('Long', 'Strong Long'), 'Strong Long': ('Strong Long',)}
OBJECTIVES = ('sharpe', 'return', 'profit_factor', 'calmar')


def objective_value(metrics, objective):
    """Higher is better; undefined values rank last and an infinite profit factor is capped."""
    if objective == 'sharpe':
        value = metrics['Sharpe ratio']
    elif objective == 'return':
        value = metrics['Total return %']
    elif objective == 'profit_factor':
        value = metrics['Profit factor']
    elif objective == 'calmar':
        drawdown = abs(metrics['Max drawdown %'])
        value = metrics['Total return %'] / drawdown if drawdown > 0 else np.nan
    else:
        raise ValueError(f'unknown objective {objective!r}, choose from {", ".join(OBJECTIVES)}')
    return float(np.nan_to_num(value, nan=-np.inf, posinf=1e9))


def parameter_sets(grid):
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def stitch(parts, initial_capital):
    """Chain equity curves that each start at the initial capital into one compounded curve."""
    pieces, level = [], 1.0
    for equity in parts:
        scaled = equity / equity.iloc[0] * level
        pieces.append(scaled)
        level = scaled.iloc[-1]
    return pd.concat(pieces) * initial_capital


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos_equity: pd.Series
    baseline_equity: pd.Series
    buy_and_hold_equity: pd.Series
    oos_metrics: dict
    baseline_metrics: dict
    efficiency: float
    recommended: dict
    grid: pd.DataFrame
    objective: str


def walk_forward(data, base_config=None, grid=None, folds=4, objective='sharpe', min_trades=3,
                 lookback=SWING_LOOKBACK, default_adx=ADX_TREND_THRESHOLD):
    """Optimise parameters on an expanding window and trade the next, unseen period with them.

    The history is split into `folds + 1` equal parts. Fold k searches the grid on parts 1..k and
    trades part k+1 with the winner, so every out-of-sample result comes from parameters chosen
    without seeing that period. Signals are causal, so every window is a slice of one analysis
    per ADX threshold. Parameter sets with fewer than `min_trades` trades are not eligible.
    """
    base_config = base_config or BacktestConfig()
    grid = grid or DEFAULT_GRID
    if folds < 1 or len(data) < (folds + 1) * 30:
        raise ValueError(f'{len(data)} bars are not enough for {folds} folds; use more history or fewer folds')
    objective_value(  # validate the objective name early
        {'Sharpe ratio': 0, 'Total return %': 0, 'Profit factor': 0, 'Max drawdown %': 0}, objective)

    analyses = {}

    def analysis(adx):
        if adx not in analyses:
            analyses[adx] = run_analysis(data, lookback=lookback, adx_threshold=adx)
        return analyses[adx]

    def backtest(params, start, stop):
        config = replace(base_config, stop_loss_atr=params['stop_loss_atr'],
                         take_profit_atr=params['take_profit_atr'], entry_decisions=ENTRY_DECISIONS[params['entry']])
        return run_backtest(analysis(params['adx_threshold']).iloc[start:stop], config)

    def score(result):
        return objective_value(result.metrics, objective) if result.metrics['Trades'] >= min_trades else -np.inf

    def optimise(start, stop):
        best, best_score, rows = None, -np.inf, []
        for params in parameter_sets(grid):
            result = backtest(params, start, stop)
            value = score(result)
            rows.append({**params, **_grid_metrics(result.metrics), 'Objective': value})
            if best is None or value > best_score:
                best, best_score = params, value
        return best, best_score, rows

    default = {'stop_loss_atr': base_config.stop_loss_atr, 'take_profit_atr': base_config.take_profit_atr,
               'entry': 'Long', 'adx_threshold': default_adx}
    times = data['timestamp'].reset_index(drop=True)
    bounds = np.linspace(0, len(data), folds + 2).astype(int)
    rows, oos_parts, baseline_parts = [], [], []
    in_sample_rate = out_of_sample_rate = 0.0
    for k in range(1, folds + 1):
        train, test = (0, bounds[k]), (bounds[k], bounds[k + 1])
        best, best_score, _ = optimise(*train)
        in_sample, out_of_sample, baseline = backtest(best, *train), backtest(best, *test), backtest(default, *test)
        in_sample_rate += in_sample.metrics['Total return %'] / (train[1] - train[0])
        out_of_sample_rate += out_of_sample.metrics['Total return %'] / (test[1] - test[0])
        rows.append({
            'Fold': k,
            'Train': f'{times[train[0]]:%Y-%m-%d} -> {times[train[1] - 1]:%Y-%m-%d}',
            'Test': f'{times[test[0]]:%Y-%m-%d} -> {times[test[1] - 1]:%Y-%m-%d}',
            'Stop ATR': best['stop_loss_atr'],
            'Target ATR': best['take_profit_atr'],
            'Entry': best['entry'],
            'ADX': best['adx_threshold'],
            'IS objective': best_score if np.isfinite(best_score) else np.nan,
            'IS return %': in_sample.metrics['Total return %'],
            'OOS return %': out_of_sample.metrics['Total return %'],
            'OOS trades': out_of_sample.metrics['Trades'],
            'Default OOS return %': baseline.metrics['Total return %'],
            'Buy & hold %': out_of_sample.metrics['Buy & hold return %'],
        })
        oos_parts.append(out_of_sample.equity)
        baseline_parts.append(baseline.equity)

    capital = base_config.initial_capital
    oos_equity, baseline_equity = stitch(oos_parts, capital), stitch(baseline_parts, capital)
    test_data = data.iloc[bounds[1]:]
    buy_and_hold = pd.Series(capital * test_data['close'].to_numpy() / test_data['open'].iloc[0],
                             index=oos_equity.index, name='Buy & hold')
    fold_table = pd.DataFrame(rows)
    oos_metrics = {**equity_stats(oos_equity), 'Trades': int(fold_table['OOS trades'].sum())}
    baseline_metrics = equity_stats(baseline_equity)
    efficiency = out_of_sample_rate / in_sample_rate * 100 if in_sample_rate > 0 else np.nan

    _, _, grid_rows = optimise(0, len(data))
    grid_table = pd.DataFrame(grid_rows).sort_values('Objective', ascending=False, ignore_index=True)
    recommended = {key: grid_table.loc[0, key] for key in grid}
    return WalkForwardResult(fold_table, oos_equity, baseline_equity, buy_and_hold, oos_metrics, baseline_metrics,
                             efficiency, recommended, grid_table, objective)


def _grid_metrics(metrics):
    return {
        'Return %': metrics['Total return %'],
        'Max DD %': metrics['Max drawdown %'],
        'Sharpe': metrics['Sharpe ratio'],
        'Profit factor': metrics['Profit factor'],
        'Win rate %': metrics['Win rate %'],
        'Trades': metrics['Trades'],
    }
