import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .backtest import BacktestConfig
from .price_action import display_trendline
from .signals import SignalScore

ASSETS = Path(__file__).parent / 'assets'
REPO_URL = 'https://github.com/cagriefe/algorithmic-trading'
CHART_LIBRARY = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js'
CHART_LIBRARY_SRI = 'sha384-KkYqTZlM13Zya6fVUF3IGCBOQ1ehFeJVVgxaVvD/0MebineF9EI8Jkd3NnWNChqN'
DECISION_CLASSES = {'Strong Long': 'strong-long', 'Long': 'long'}
EXIT_LABELS = {'Take profit': 'TP', 'Stop loss': 'SL', 'Breakdown': 'Exit', 'Time exit': 'Time', 'End of data': 'Open'}


def _escape(value):
    return html.escape(str(value))


def _fmt(value, decimals=2, suffix='', signed=False):
    if value is None or pd.isna(value):
        return '&ndash;'
    if np.isinf(value):
        return '&infin;' if value > 0 else '-&infin;'
    sign = '+' if signed and value > 0 else ''
    return f'{sign}{value:,.{decimals}f}{suffix}'


def _tone(value):
    if value is None or pd.isna(value) or value == 0:
        return ''
    return 'pos' if value > 0 else 'neg'


def badge(decision):
    return f'<span class="badge {DECISION_CLASSES.get(decision, "neutral")}">{_escape(decision)}</span>'


def price_precision(prices):
    """Decimals needed to show a price series, e.g. 2 for BTC and 5 for a coin trading at 0.12."""
    median = float(np.nanmedian(prices))
    if not np.isfinite(median) or median <= 0:
        return 2
    return int(min(8, max(2, 4 - np.floor(np.log10(median)))))


def is_intraday(data):
    return bool(data['timestamp'].diff().median() < pd.Timedelta(days=1))


def epoch_seconds(timestamps):
    return ((pd.to_datetime(pd.Series(timestamps)) - pd.Timestamp('1970-01-01')) // pd.Timedelta(seconds=1)).tolist()


def _line(times, values):
    # Missing values become whitespace points so the line breaks instead of interpolating
    return [{'time': t, 'value': float(v)} if np.isfinite(v) else {'time': t}
            for t, v in zip(times, np.asarray(values, dtype=float))]


def _markers(data, result, times):
    markers = []
    decisions = data['Decision'].to_numpy()
    for decision, color, size in [('Long', '#f59e0b', 0.6), ('Strong Long', '#22c55e', 1.0)]:
        markers += [{'time': times[i], 'position': 'belowBar', 'color': color, 'shape': 'circle', 'size': size}
                    for i in np.flatnonzero(decisions == decision)]
    if result is not None and not result.trades.empty:
        trades = result.trades
        for entry, exit_, trade in zip(epoch_seconds(trades['entry_time']), epoch_seconds(trades['exit_time']),
                                       trades.itertuples()):
            markers.append({'time': entry, 'position': 'belowBar', 'color': '#60a5fa', 'shape': 'arrowUp',
                            'text': 'Buy'})
            markers.append({'time': exit_, 'position': 'aboveBar', 'color': '#22c55e' if trade.pnl > 0 else '#ef4444',
                            'shape': 'arrowDown',
                            'text': f'{EXIT_LABELS.get(trade.exit_reason, trade.exit_reason)} {trade.return_pct:+.1f}%'})
    return sorted(markers, key=lambda marker: marker['time'])


def chart_payload(data, result=None):
    times = epoch_seconds(data['timestamp'])
    open_, high, low, close, volume = (data[c].to_numpy(dtype=float) for c in ['open', 'high', 'low', 'close', 'volume'])
    payload = {
        'intraday': is_intraday(data),
        'precision': price_precision(close),
        'candles': [{'time': t, 'open': o, 'high': h, 'low': lo, 'close': c}
                    for t, o, h, lo, c in zip(times, open_, high, low, close)],
        'volume': [{'time': t, 'value': v, 'color': 'rgba(34,197,94,0.35)' if c >= o else 'rgba(239,68,68,0.35)'}
                   for t, v, o, c in zip(times, volume, open_, close)],
        'lines': {
            'SMA 50': _line(times, data['SMA_50']),
            'SMA 200': _line(times, data['SMA_200']),
            'Resistance trendline': _line(times, display_trendline(data, data['Trendline_High'], data['Last_Swing_High'])),
            'Support trendline': _line(times, display_trendline(data, data['Trendline_Low'], data['Last_Swing_Low'])),
        },
        'markers': _markers(data, result, times),
        'bars': {
            'decision': data['Decision'].astype(str).tolist(),
            'score': data['Score'].astype(float).round(2).tolist(),
            'pattern': data['Pattern'].astype(str).tolist(),
            'reasoning': data['Reasoning'].astype(str).tolist(),
        },
        'equity': [], 'buyHold': [], 'drawdown': [],
    }
    if result is not None:
        equity = result.equity.to_numpy(dtype=float)
        payload['equity'] = _line(times, equity)
        payload['buyHold'] = _line(times, result.metrics['Initial capital'] * close / open_[0])
        payload['drawdown'] = _line(times, (equity / np.maximum.accumulate(equity) - 1) * 100)
    return payload


def embed_json(payload):
    # Escaping <, > and & keeps the JSON from closing the <script> element it lives in
    text = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    return text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _signal_card(last, max_score, precision):
    decision = last['Decision']
    return (f'<div class="signal-card"><div class="label">Signal on the last closed bar</div>'
            f'<div class="decision {DECISION_CLASSES.get(decision, "")}">{_escape(decision)}</div>'
            f'<div class="meter"><span style="width:{min(max(last["Confidence"], 0), 100):.0f}%"></span></div>'
            f'<div class="sub">Score {last["Score"]:.1f} / {max_score:g} &middot; {last["Confidence"]:.0f}% confidence'
            f' &middot; close {last["close"]:,.{precision}f}</div></div>')


def _cards(result, monte_carlo_stats):
    m = result.metrics
    cards = [
        ('Strategy return', _fmt(m['Total return %'], signed=True, suffix='%'), _tone(m['Total return %']),
         f'Buy &amp; hold {_fmt(m["Buy & hold return %"], signed=True, suffix="%")}'),
        ('Max drawdown', _fmt(m['Max drawdown %'], suffix='%'), 'neg' if m['Max drawdown %'] < 0 else '',
         f'Exposure {_fmt(m["Exposure %"], 0, "%")}'),
        ('Sharpe ratio', _fmt(m['Sharpe ratio']), _tone(m['Sharpe ratio']), 'annualised'),
        ('Win rate', _fmt(m['Win rate %'], 1, '%'), '', f'{m["Trades"]} trades'),
        ('Profit factor', _fmt(m['Profit factor']), _tone(m['Profit factor'] - 1),
         f'Avg trade {_fmt(m["Avg trade return %"], signed=True, suffix="%")}'),
        ('Fees paid', _fmt(m['Fees paid']), '', f'Capital {_fmt(m["Initial capital"], 0)}'),
    ]
    if monte_carlo_stats:
        cards.append(('Chance of loss', _fmt(monte_carlo_stats['Probability of loss %'], 0, '%'), '',
                      f'{monte_carlo_stats["Simulations"]:,} Monte Carlo runs'))
    return '<section class="cards">' + ''.join(
        f'<div class="card"><div class="label">{label}</div><div class="value {tone}">{value}</div>'
        f'<div class="sub">{sub}</div></div>' for label, value, tone, sub in cards) + '</section>'


def _checklist(data, scorer):
    last = data.iloc[[-1]]
    items = []
    for component, (met, label) in scorer.conditions(last).items():
        ok = bool(met.iloc[0])
        text = f'{label}: {last["Pattern"].iloc[0]}' if component == 'pattern' and ok else label
        items.append(f'<li><span class="{"ok" if ok else "no"}">{_escape(text)}</span>'
                     f'<span class="muted">+{scorer.weights[component]:g}</span></li>')
    return '<ul class="checklist">' + ''.join(items) + '</ul>'


def _trade_plan(last, config, precision):
    atr = last['ATR']
    if last['Decision'] not in config.entry_decisions or not atr > 0:
        return ('<p class="muted">No entry signal on the last closed bar &ndash; the checklist shows which '
                'conditions are missing.</p>')
    entry = last['close']
    stop, target = entry - config.stop_loss_atr * atr, entry + config.take_profit_atr * atr
    risk = config.initial_capital * config.risk_per_trade
    quantity = min(risk / (entry - stop), config.initial_capital / entry)
    cells = [
        ('Entry (next open)', f'&asymp; {entry:,.{precision}f}', ''),
        ('Stop loss', f'{stop:,.{precision}f}', f'{(stop / entry - 1) * 100:+.1f}%'),
        ('Take profit', f'{target:,.{precision}f}', f'{(target / entry - 1) * 100:+.1f}%'),
        ('Position size', f'{quantity:,.6g}', f'&asymp; {quantity * entry:,.0f} notional, risking {risk:,.0f}'),
    ]
    return '<div class="plan">' + ''.join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>' for label, value, sub in cells) + '</div>'


def _key_value_table(stats):
    rows = []
    for name, value in stats.items():
        if name == 'Simulations':
            continue
        is_return = 'return' in name.lower()
        rows.append(f'<tr><td>{_escape(name)}</td><td class="{_tone(value) if is_return else ""}">'
                    f'{_fmt(value, signed=is_return)}</td></tr>')
    return f'<table><tbody>{"".join(rows)}</tbody></table>'


def _edge_table(table):
    head = ''.join(f'<th>{_escape(column)}</th>' for column in table.columns)
    rows = []
    for record in table.itertuples(index=False):
        cells = [f'<td>{_escape(record[0])}</td>', f'<td>{record[1]}</td>']
        for column, value in zip(table.columns[2:], record[2:]):
            signed = not column.startswith('Hit')
            cells.append(f'<td class="{_tone(value) if signed else ""}">{_fmt(value, signed=signed)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _trades_table(trades, precision, intraday):
    if trades.empty:
        return '<p class="muted">No trades were opened.</p>'
    time_format = '%Y-%m-%d %H:%M' if intraday else '%Y-%m-%d'
    rows = ''.join(
        f'<tr><td>{t.entry_time:{time_format}}</td><td>{t.entry_price:,.{precision}f}</td>'
        f'<td>{t.exit_time:{time_format}}</td><td>{t.exit_price:,.{precision}f}</td><td>{_escape(t.exit_reason)}</td>'
        f'<td>{t.bars_held}</td><td class="{_tone(t.return_pct)}">{_fmt(t.return_pct, signed=True, suffix="%")}</td></tr>'
        for t in trades.iloc[::-1].itertuples())
    head = ''.join(f'<th>{name}</th>' for name in ['Entry', 'Entry price', 'Exit', 'Exit price', 'Reason', 'Bars',
                                                    'Return'])
    return f'<div class="scroll tall"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def page(title, body, scripts=''):
    css = (ASSETS / 'style.css').read_text(encoding='utf-8')
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{_escape(title)}</title>\n'
            f'<style>\n{css}</style>\n</head>\n<body>\n<div class="wrap">\n{body}\n'
            f'<footer>Generated by <a href="{REPO_URL}">algotrading</a> v{__version__} &middot; '
            f'charts by TradingView Lightweight Charts&trade; &middot; for education and research only, '
            f'not financial advice.</footer>\n</div>\n{scripts}</body>\n</html>\n')


def render_report(data, result=None, config=None, monte_carlo_stats=None, edge_table=None, title='',
                  back_link=None, generated_at=None):
    """Self-contained interactive HTML report for one market."""
    config = config or BacktestConfig()
    scorer = SignalScore()
    last = data.iloc[-1]
    payload = chart_payload(data, result)
    precision, intraday = payload['precision'], payload['intraday']
    generated_at = generated_at or datetime.now(timezone.utc)
    time_format = '%Y-%m-%d %H:%M' if intraday else '%Y-%m-%d'
    period = f'{data["timestamp"].iloc[0]:{time_format}} &rarr; {data["timestamp"].iloc[-1]:{time_format}}'

    back = f'<a href="{_escape(back_link)}">&larr; All markets</a>' if back_link else ''
    body = [
        f'<header class="top"><div>{back}<h1>{_escape(title)}</h1><p class="muted">{period} &middot; '
        f'{len(data):,} bars &middot; generated {generated_at:%Y-%m-%d %H:%M} UTC</p></div>'
        f'{_signal_card(last, scorer.max_score, precision)}</header>',
    ]
    if result is not None:
        body.append(_cards(result, monte_carlo_stats))
    body.append('<section class="panel"><div class="panel-head"><h2>Price, signals &amp; trades</h2>'
                '<span class="muted">hover a candle to see why it scored</span></div>'
                '<div id="legend" class="legend"></div><div id="price-chart" class="chart"></div></section>')
    if result is not None:
        body.append('<section class="panel"><h2>Equity vs buy &amp; hold</h2>'
                    '<div id="equity-chart" class="chart small"></div></section>')
    body.append(f'<section class="grid-2"><div class="panel"><h2>Signal checklist (last bar)</h2>'
                f'{_checklist(data, scorer)}</div><div class="panel"><h2>Trade plan</h2>'
                f'{_trade_plan(last, config, precision)}</div></section>')
    if monte_carlo_stats:
        body.append(f'<section class="panel"><h2>Monte Carlo &ndash; {monte_carlo_stats["Simulations"]:,} resampled '
                    f'trade sequences</h2>{_key_value_table(monte_carlo_stats)}</section>')
    if edge_table is not None:
        body.append('<section class="panel"><h2>Signal edge &ndash; what happened after each signal</h2>'
                    f'{_edge_table(edge_table)}</section>')
    if result is not None:
        body.append(f'<section class="panel"><h2>Trades</h2>{_trades_table(result.trades, precision, intraday)}'
                    '</section>')

    scripts = (f'<script id="report-data" type="application/json">{embed_json(payload)}</script>\n'
               f'<script src="{CHART_LIBRARY}" integrity="{CHART_LIBRARY_SRI}" crossorigin="anonymous"></script>\n'
               f'<script>\n{(ASSETS / "report.js").read_text(encoding="utf-8")}</script>\n')
    return page(f'{title} · Signal report', '\n'.join(body), scripts)


def write_report(path, *args, **kwargs):
    Path(path).write_text(render_report(*args, **kwargs), encoding='utf-8')
