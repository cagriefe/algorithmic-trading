import pandas as pd
import pytest

from algotrading.backtest import BacktestConfig, run_backtest

NO_COSTS = BacktestConfig(fee_rate=0.0, slippage=0.0)
FLAT = (100, 101, 99, 100, 'Neutral')
SIGNAL = (100, 101, 99, 100, 'Long')


def bars(*rows, atr=5.0):
    """Rows of (open, high, low, close, decision[, breakdown below support])."""
    data = pd.DataFrame([row[:5] for row in rows], columns=['open', 'high', 'low', 'close', 'Decision'])
    data['timestamp'] = pd.date_range('2024-01-01', periods=len(rows), freq='D')
    data['ATR'] = atr
    data['Breakout'] = ['Below' if row[5:] and row[5] else 'No' for row in rows]
    data['Trendline_Breakout'] = 'No'
    return data


def test_signal_fills_at_next_open_and_exits_at_take_profit():
    result = run_backtest(bars(FLAT, SIGNAL, (100, 105, 98, 104, 'Neutral'), (104, 121, 103, 118, 'Neutral'), FLAT),
                          NO_COSTS)
    trade = result.trades.iloc[0]

    assert len(result.trades) == 1
    assert trade['entry_time'] == pd.Timestamp('2024-01-03')
    assert (trade['entry_price'], trade['stop_loss'], trade['take_profit']) == (100, 90, 120)
    assert trade['quantity'] == pytest.approx(20)  # 2% of 10,000 risked over a 10 point stop
    assert (trade['exit_price'], trade['exit_reason']) == (120, 'Take profit')
    assert result.metrics['Final equity'] == pytest.approx(10_400)


def test_gap_through_stop_fills_at_the_open():
    result = run_backtest(bars(SIGNAL, (100, 102, 98, 101, 'Neutral'), (85, 86, 80, 82, 'Neutral')), NO_COSTS)
    trade = result.trades.iloc[0]

    assert (trade['exit_price'], trade['exit_reason']) == (85, 'Stop loss')
    assert trade['pnl'] == pytest.approx(-300)


def test_bar_touching_stop_and_target_counts_as_stop():
    result = run_backtest(bars(SIGNAL, (100, 125, 85, 110, 'Neutral'), FLAT), NO_COSTS)

    assert result.trades['exit_reason'].tolist() == ['Stop loss']
    assert result.trades['exit_price'].iloc[0] == 90


def test_breakdown_exits_at_next_open():
    result = run_backtest(bars(SIGNAL, (100, 103, 97, 99, 'Neutral', True), (98, 99, 96, 97, 'Neutral'), FLAT),
                          NO_COSTS)
    trade = result.trades.iloc[0]

    assert (trade['exit_time'], trade['exit_price'], trade['exit_reason']) == \
        (pd.Timestamp('2024-01-03'), 98, 'Breakdown')


def test_time_exit_and_end_of_data():
    config = BacktestConfig(fee_rate=0.0, slippage=0.0, max_holding_bars=2)
    result = run_backtest(bars(SIGNAL, FLAT, FLAT, SIGNAL, FLAT), config)

    assert result.trades['exit_reason'].tolist() == ['Time exit', 'End of data']
    assert result.trades['exit_time'].tolist() == [pd.Timestamp('2024-01-04'), pd.Timestamp('2024-01-05')]


def test_fees_and_slippage_are_charged_on_both_fills():
    result = run_backtest(bars(SIGNAL, FLAT), BacktestConfig(fee_rate=0.001, slippage=0.001))
    trade = result.trades.iloc[0]
    entry, exit_ = 100 * 1.001, 100 * 0.999
    fees = trade['quantity'] * (entry + exit_) * 0.001
    expected_pnl = trade['quantity'] * (exit_ - entry) - fees

    assert (trade['entry_price'], trade['exit_price']) == (pytest.approx(entry), pytest.approx(exit_))
    assert trade['pnl'] == pytest.approx(expected_pnl)
    assert result.metrics['Fees paid'] == pytest.approx(fees)
    assert result.metrics['Final equity'] == pytest.approx(10_000 + expected_pnl)


def test_position_size_never_exceeds_cash():
    result = run_backtest(bars(SIGNAL, FLAT, atr=0.01), BacktestConfig(fee_rate=0.001, slippage=0.0))
    trade = result.trades.iloc[0]

    assert trade['quantity'] * trade['entry_price'] * 1.001 == pytest.approx(10_000)


def test_metrics_for_a_winning_and_a_losing_trade():
    result = run_backtest(bars(SIGNAL, (100, 121, 99, 110, 'Neutral'), SIGNAL, (100, 101, 89, 95, 'Neutral'), FLAT),
                          NO_COSTS)
    metrics = result.metrics

    assert result.trades['pnl'].tolist() == pytest.approx([400, -208])  # second trade risks 2% of 10,400
    assert metrics['Win rate %'] == 50
    assert metrics['Profit factor'] == pytest.approx(400 / 208)
    assert metrics['Max drawdown %'] == pytest.approx(-2.0)
    assert metrics['Total return %'] == pytest.approx(1.92)


def test_no_signals_means_no_trades():
    result = run_backtest(bars(*[FLAT] * 5), NO_COSTS)

    assert result.trades.empty
    assert result.metrics['Trades'] == 0
    assert result.metrics['Total return %'] == 0
    assert (result.equity == 10_000).all()
