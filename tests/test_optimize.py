import numpy as np
import pandas as pd
import pytest

from algotrading.analysis import run_analysis
from algotrading.backtest import BacktestConfig, run_backtest
from algotrading.cli import main
from algotrading.optimize import (ENTRY_DECISIONS, objective_value, parameter_sets, stitch, walk_forward)

SMALL_GRID = {'stop_loss_atr': (1.5, 3.0), 'take_profit_atr': (2.0, 4.0), 'entry': ('Long',), 'adx_threshold': (25.0,)}


def metrics(**values):
    base = {'Sharpe ratio': 1.0, 'Total return %': 10.0, 'Profit factor': 1.5, 'Max drawdown %': -5.0}
    return {**base, **values}


def test_stitch_compounds_consecutive_periods():
    index = pd.date_range('2024-01-01', periods=4, freq='D')
    parts = [pd.Series([100.0, 110.0], index=index[:2]), pd.Series([100.0, 90.0], index=index[2:])]

    assert stitch(parts, 1000).tolist() == pytest.approx([1000, 1100, 1100, 990])


@pytest.mark.parametrize('objective, values, expected', [
    ('sharpe', {}, 1.0),
    ('return', {}, 10.0),
    ('calmar', {}, 2.0),
    ('profit_factor', {'Profit factor': float('inf')}, 1e9),
    ('sharpe', {'Sharpe ratio': float('nan')}, -np.inf),
])
def test_objective_values(objective, values, expected):
    assert objective_value(metrics(**values), objective) == expected


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match='unknown objective'):
        objective_value(metrics(), 'luck')


def test_parameter_grid_is_the_full_product():
    assert len(parameter_sets(SMALL_GRID)) == 4
    assert {'stop_loss_atr': 3.0, 'take_profit_atr': 4.0, 'entry': 'Long', 'adx_threshold': 25.0} in \
        parameter_sets(SMALL_GRID)


def test_walk_forward_trades_each_period_with_parameters_chosen_before_it(market_data):
    result = walk_forward(market_data, grid=SMALL_GRID, folds=3, min_trades=1)
    folds = result.folds

    assert len(folds) == 3
    assert len(result.oos_equity) == len(result.buy_and_hold_equity) == 450  # everything after the first quarter
    assert result.oos_equity.iloc[0] == pytest.approx(10_000)
    assert result.oos_metrics['Trades'] == folds['OOS trades'].sum()
    assert list(result.grid['Objective']) == sorted(result.grid['Objective'], reverse=True)
    assert result.recommended == {key: result.grid.loc[0, key] for key in SMALL_GRID}

    # fold 1 picked the parameter set with the best in-sample Sharpe ratio on the first 150 bars
    analysed = run_analysis(market_data)

    def in_sample_sharpe(params):
        config = BacktestConfig(stop_loss_atr=params['stop_loss_atr'], take_profit_atr=params['take_profit_atr'],
                                entry_decisions=ENTRY_DECISIONS[params['entry']])
        result = run_backtest(analysed.iloc[:150], config)
        return objective_value(result.metrics, 'sharpe') if result.metrics['Trades'] >= 1 else -np.inf

    best = max(parameter_sets(SMALL_GRID), key=in_sample_sharpe)
    assert (folds.loc[0, 'Stop ATR'], folds.loc[0, 'Target ATR']) == (best['stop_loss_atr'], best['take_profit_atr'])


def test_later_data_cannot_change_earlier_folds(market_data):
    changed = market_data.copy()
    later = changed.index >= 300  # after fold 1's test period (bars 150-299)
    changed.loc[later, ['open', 'high', 'low', 'close']] *= 1.5

    original = walk_forward(market_data, grid=SMALL_GRID, folds=3, min_trades=1).folds.iloc[0]
    rerun = walk_forward(changed, grid=SMALL_GRID, folds=3, min_trades=1).folds.iloc[0]

    pd.testing.assert_series_equal(original, rerun)


def test_too_little_history_is_rejected(market_data):
    with pytest.raises(ValueError, match='not enough'):
        walk_forward(market_data.iloc[:100], grid=SMALL_GRID, folds=4)


def test_optimize_command(market_data, tmp_path, capsys):
    csv_path = tmp_path / 'synthetic.csv'
    market_data.to_csv(csv_path, index=False)
    out = tmp_path / 'out'

    assert main(['optimize', '--csv', str(csv_path), '--folds', '2', '--min-trades', '1',
                 '--output-dir', str(out)]) == 0

    assert 'Walk-Forward Optimisation' in capsys.readouterr().out
    for suffix in ['walkforward.csv', 'grid.csv', 'walkforward.png']:
        assert (out / f'synthetic_{suffix}').exists()
