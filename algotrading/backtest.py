from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADE_COLUMNS = ['entry_time', 'entry_price', 'exit_time', 'exit_price', 'quantity', 'stop_loss',
                 'take_profit', 'exit_reason', 'bars_held', 'pnl', 'return_pct']


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001          # per fill, Binance spot taker fee
    slippage: float = 0.0005         # adverse price move per fill
    risk_per_trade: float = 0.02     # share of equity lost when the stop is hit
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0     # 1:2 risk/reward with the default stop
    max_holding_bars: int | None = None
    exit_on_breakdown: bool = True
    entry_decisions: tuple = ('Long', 'Strong Long')


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict


def run_backtest(data, config=None):
    """Simulate long-only trades from the signals produced by `run_analysis`.

    A signal on a bar's close is filled at the next bar's open. Each trade gets an ATR based
    stop loss and take profit, and is sized so that hitting the stop loses `risk_per_trade`
    of equity (never more than the available cash). When a bar touches both levels the stop
    is assumed to be hit first. Breakdowns below support exit at the next open.
    """
    config = config or BacktestConfig()
    data = data.reset_index(drop=True)
    open_, high, low, close, atr = (data[c].to_numpy(dtype=float) for c in ['open', 'high', 'low', 'close', 'ATR'])
    times = data['timestamp']
    entry_signal = data['Decision'].isin(config.entry_decisions).to_numpy()
    breakdown = ((data['Breakout'] == 'Below') | (data['Trendline_Breakout'] == 'Below')).to_numpy()

    cash, fees = config.initial_capital, 0.0
    equity = np.empty(len(data))
    in_market = np.zeros(len(data), dtype=bool)
    trades = []
    position = None
    pending_exit = None

    def close_position(bar, price, reason):
        nonlocal cash, fees, position
        fill = price * (1 - config.slippage)
        proceeds = position['quantity'] * fill
        fee = proceeds * config.fee_rate
        cash += proceeds - fee
        fees += fee
        cost = position['quantity'] * position['entry_price']
        pnl = proceeds - fee - cost - position['entry_fee']
        trades.append({
            'entry_time': times[position['bar']], 'entry_price': position['entry_price'],
            'exit_time': times[bar], 'exit_price': fill, 'quantity': position['quantity'],
            'stop_loss': position['stop'], 'take_profit': position['target'], 'exit_reason': reason,
            'bars_held': bar - position['bar'], 'pnl': pnl, 'return_pct': pnl / cost * 100,
        })
        position = None

    for i in range(len(data)):
        if position is not None and pending_exit:
            close_position(i, open_[i], pending_exit)
            pending_exit = None
        elif position is None and i > 0 and entry_signal[i - 1] and atr[i - 1] > 0:
            entry = open_[i] * (1 + config.slippage)
            stop = entry - config.stop_loss_atr * atr[i - 1]
            target = entry + config.take_profit_atr * atr[i - 1]
            quantity = min(cash * config.risk_per_trade / (entry - stop),
                           cash / (entry * (1 + config.fee_rate)))
            entry_fee = quantity * entry * config.fee_rate
            cash -= quantity * entry + entry_fee
            fees += entry_fee
            position = {'bar': i, 'entry_price': entry, 'quantity': quantity,
                        'stop': stop, 'target': target, 'entry_fee': entry_fee}

        if position is not None:
            gapped = i > position['bar']
            if gapped and open_[i] <= position['stop']:
                close_position(i, open_[i], 'Stop loss')
            elif gapped and open_[i] >= position['target']:
                close_position(i, open_[i], 'Take profit')
            elif low[i] <= position['stop']:
                close_position(i, position['stop'], 'Stop loss')
            elif high[i] >= position['target']:
                close_position(i, position['target'], 'Take profit')

        if position is not None:
            if config.exit_on_breakdown and breakdown[i]:
                pending_exit = 'Breakdown'
            elif config.max_holding_bars and i - position['bar'] + 1 >= config.max_holding_bars:
                pending_exit = 'Time exit'

        in_market[i] = position is not None
        equity[i] = cash + (position['quantity'] * close[i] if position else 0.0)

    if position is not None:
        close_position(len(data) - 1, close[-1], 'End of data')
        equity[-1] = cash

    trades = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity = pd.Series(equity, index=times, name='Equity')
    metrics = compute_metrics(data, equity, trades, in_market, fees, config)
    return BacktestResult(trades, equity, metrics)


def compute_metrics(data, equity, trades, in_market, fees, config):
    returns = equity.pct_change().dropna()
    seconds_per_bar = data['timestamp'].diff().median().total_seconds()
    periods_per_year = 365 * 24 * 3600 / seconds_per_bar if seconds_per_bar > 0 else np.nan
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan

    wins, losses = trades[trades['pnl'] > 0], trades[trades['pnl'] <= 0]
    gross_loss = -losses['pnl'].sum()
    if gross_loss > 0:
        profit_factor = wins['pnl'].sum() / gross_loss
    else:
        profit_factor = np.inf if len(wins) else np.nan
    has_trades = len(trades) > 0

    return {
        'Period': f"{data['timestamp'].iloc[0]:%Y-%m-%d} -> {data['timestamp'].iloc[-1]:%Y-%m-%d}",
        'Initial capital': config.initial_capital,
        'Final equity': equity.iloc[-1],
        'Total return %': (equity.iloc[-1] / config.initial_capital - 1) * 100,
        'Buy & hold return %': (data['close'].iloc[-1] / data['open'].iloc[0] - 1) * 100,
        'Max drawdown %': (equity / equity.cummax() - 1).min() * 100,
        'Sharpe ratio': sharpe,
        'Trades': len(trades),
        'Win rate %': len(wins) / len(trades) * 100 if has_trades else np.nan,
        'Profit factor': profit_factor,
        'Avg trade return %': trades['return_pct'].mean() if has_trades else np.nan,
        'Avg bars held': trades['bars_held'].mean() if has_trades else np.nan,
        'Exposure %': in_market.mean() * 100,
        'Fees paid': fees,
    }
