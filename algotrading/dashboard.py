from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .edge import signal_edge
from .html_report import ASSETS, badge, escape, fmt, page, tone, write_report
from .montecarlo import monte_carlo


def sparkline(values, width=96, height=28):
    """Inline SVG line of the last 90 values, green when they rose and red when they fell."""
    values = np.asarray(values, dtype=float)[-90:]
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return ''
    low, span = values.min(), (values.max() - values.min()) or 1.0
    xs = np.linspace(0, width, len(values))
    ys = height - 2 - (values - low) / span * (height - 4)
    points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs, ys))
    color = '#22c55e' if values[-1] >= values[0] else '#ef4444'
    return (f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/></svg>')


def market_breadth(summary):
    above = summary['Above SMA 200'].dropna().astype(float)
    share_above = above.mean() * 100 if len(above) else np.nan
    if np.isnan(share_above):
        regime = 'unknown'
    else:
        regime = 'bullish' if share_above >= 60 else 'bearish' if share_above <= 40 else 'mixed'
    best = summary.loc[summary['Strategy %'].idxmax()] if len(summary) else None
    return {
        'markets': len(summary),
        'strong_long': summary.loc[summary['Decision'] == 'Strong Long', 'Symbol'].tolist(),
        'long': summary.loc[summary['Decision'] == 'Long', 'Symbol'].tolist(),
        'share_above_sma200': share_above,
        'regime': regime,
        'median_rsi': summary['RSI'].median(),
        'best': best,
    }


def _breadth_cards(breadth):
    best = breadth['best']
    cards = [
        ('Markets scanned', breadth['markets'], ''),
        ('Strong Long now', len(breadth['strong_long']), ', '.join(breadth['strong_long']) or 'none'),
        ('Long now', len(breadth['long']), ', '.join(breadth['long']) or 'none'),
        ('Above 200-bar average', fmt(breadth['share_above_sma200'], 0, '%'), f'market regime: {breadth["regime"]}'),
        ('Median RSI', fmt(breadth['median_rsi'], 0), 'momentum across markets'),
    ]
    if best is not None:
        cards.append(('Best backtest', escape(best['Symbol']), fmt(best['Strategy %'], signed=True, suffix='%')))
    return '<section class="cards">' + ''.join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="sub">{escape(sub) if isinstance(sub, str) else sub}</div></div>'
        for label, value, sub in cards) + '</section>'


def _cell(value, display=None, css=''):
    sort_value = '' if value is None or (isinstance(value, float) and np.isnan(value)) else value
    return f'<td class="{css}" data-value="{escape(sort_value)}">{display if display is not None else escape(value)}</td>'


def _table(summary, closes):
    headers = ['#', 'Market', 'Last 90 bars', 'Close', 'Signal', 'Score', 'Pattern', 'Last signal', 'RSI',
               'Strategy', 'Buy &amp; hold', 'Max DD', 'Win rate']
    head = ''.join(f'<th data-sort>{name}</th>' for name in headers)
    rows = []
    for rank, row in enumerate(summary.to_dict('records'), start=1):
        symbol = row['Symbol']
        bars_since = row['Bars since signal']
        last_signal = 'never' if pd.isna(bars_since) else 'this bar' if bars_since == 0 else f'{int(bars_since)} bars ago'
        meter = (f'<span class="meter mini"><span style="width:{min(row["Score"] / 21 * 100, 100):.0f}%"></span></span>'
                 f' {row["Score"]:.1f}')
        rows.append('<tr>' + ''.join([
            _cell(rank),
            _cell(symbol, f'<a href="reports/{escape(symbol)}.html">{escape(symbol)}</a>'),
            _cell(row['Change %'], sparkline(closes.get(symbol, []))),
            _cell(row['Close'], f'{row["Close"]:,.6g}'),
            _cell(row['Score'], badge(row['Decision'])),
            _cell(row['Score'], meter),
            _cell(row['Pattern'], escape(row['Pattern']) if row['Pattern'] != 'None' else '<span class="muted">&ndash;</span>'),
            _cell(bars_since, last_signal),
            _cell(row['RSI'], fmt(row['RSI'], 0)),
            _cell(row['Strategy %'], fmt(row['Strategy %'], signed=True, suffix='%'), tone(row['Strategy %'])),
            _cell(row['Buy & hold %'], fmt(row['Buy & hold %'], signed=True, suffix='%'), tone(row['Buy & hold %'])),
            _cell(row['Max DD %'], fmt(row['Max DD %'], suffix='%')),
            _cell(row['Win rate %'], fmt(row['Win rate %'], 0, '%')),
        ]) + '</tr>')
    return (f'<div class="scroll"><table class="sortable"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render_dashboard(summary, closes, title='Binance Signal Board', subtitle='', errors=None, generated_at=None):
    generated_at = generated_at or datetime.now(timezone.utc)
    breadth = market_breadth(summary)
    failed = ''
    if errors:
        failed = ('<p class="muted">Skipped: ' + ', '.join(f'{escape(symbol)} ({escape(error)})'
                                                           for symbol, error in errors.items()) + '</p>')
    body = '\n'.join([
        f'<header class="top"><div><h1>{escape(title)}</h1><p class="muted">{escape(subtitle)}'
        f'{" &middot; " if subtitle else ""}updated {generated_at:%Y-%m-%d %H:%M} UTC</p></div></header>',
        _breadth_cards(breadth),
        '<section class="panel"><div class="panel-head"><h2>Markets ranked by current signal</h2>'
        '<span class="muted">click a column to sort &middot; click a market for its full report</span></div>'
        f'{_table(summary, closes)}{failed}</section>',
        '<section class="panel"><h2>How to read this board</h2><ul class="muted">'
        '<li><b>Signal</b> and <b>Score</b> come from the last closed candle: trendline and swing breakouts, '
        'ADX trend strength, MACD, moving averages, volume, RSI and bullish chart patterns, weighted to a '
        'maximum of 21.</li>'
        '<li><b>Strategy</b> is a backtest of these signals over the same history with ATR stops and targets, '
        'risk-based position sizing, fees and slippage. Compare it with <b>Buy &amp; hold</b>.</li>'
        '<li><b>Above 200-bar average</b> is the share of markets trading above their 200-candle moving average, '
        'a quick read of the overall market regime.</li></ul></section>',
    ])
    script = f'<script>\n{(ASSETS / "dashboard.js").read_text(encoding="utf-8")}</script>\n'
    return page(title, body, script)


def write_dashboard(output_dir, summary, details, config, interval, simulations=2000, title='Binance Signal Board',
                    subtitle='', errors=None, generated_at=None):
    """Write index.html with the ranked markets and an interactive report per market."""
    output_dir = Path(output_dir)
    reports = output_dir / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now(timezone.utc)
    for symbol, (data, result) in details.items():
        write_report(reports / f'{symbol}.html', data, result, config,
                     monte_carlo(result.trades, config.initial_capital, simulations), signal_edge(data),
                     title=f'{symbol} ({interval})', back_link='../index.html', generated_at=generated_at)
    closes = {symbol: data['close'].to_numpy() for symbol, (data, _) in details.items()}
    index = output_dir / 'index.html'
    index.write_text(render_dashboard(summary, closes, title, subtitle, errors, generated_at), encoding='utf-8')
    return index
