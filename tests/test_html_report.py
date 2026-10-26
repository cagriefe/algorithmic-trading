import json
import re

import pytest

from algotrading.analysis import run_analysis
from algotrading.backtest import BacktestConfig, run_backtest
from algotrading.edge import signal_edge
from algotrading.html_report import price_precision, render_report
from algotrading.montecarlo import monte_carlo


def embedded_payload(page):
    text = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S).group(1)

    def reject(constant):
        raise ValueError(f'invalid JSON constant {constant}')

    return json.loads(text, parse_constant=reject)


@pytest.fixture
def analysed(market_data):
    return run_analysis(market_data)


def test_full_report(analysed):
    result = run_backtest(analysed)
    page = render_report(analysed, result, BacktestConfig(), monte_carlo(result.trades, 10_000), signal_edge(analysed),
                         title='SYNTH (1d)')
    payload = embedded_payload(page)

    assert page.startswith('<!DOCTYPE html>')
    for section in ['Price, signals &amp; trades', 'Equity vs buy &amp; hold', 'Signal checklist', 'Monte Carlo',
                    'Signal edge', '<h2>Trades</h2>', 'id="equity-chart"']:
        assert section in page
    assert len(payload['candles']) == len(payload['bars']['score']) == len(payload['equity']) == len(analysed)
    times = [marker['time'] for marker in payload['markers']]
    assert times == sorted(times)
    assert sum(marker.get('text') == 'Buy' for marker in payload['markers']) == len(result.trades)
    assert payload['intraday'] is False


def test_report_without_backtest(analysed):
    page = render_report(analysed, title='SYNTH')

    assert 'id="price-chart"' in page
    assert 'id="equity-chart"' not in page
    assert embedded_payload(page)['equity'] == []


def test_untrusted_text_cannot_break_out_of_the_page(analysed):
    analysed.loc[analysed.index[-1], 'Reasoning'] = '</script><script>alert(1)</script>'
    page = render_report(analysed, title='<b>bold</b>')

    assert '<script>alert(1)' not in page
    assert '&lt;b&gt;bold&lt;/b&gt;' in page
    assert embedded_payload(page)['bars']['reasoning'][-1] == '</script><script>alert(1)</script>'


def test_trade_plan_for_a_fresh_signal(analysed):
    analysed.loc[analysed.index[-1], ['Decision', 'ATR', 'close']] = ['Strong Long', 2.0, 100.0]
    decimals = price_precision(analysed['close'])
    page = render_report(analysed, config=BacktestConfig(stop_loss_atr=2, take_profit_atr=4), title='SYNTH')

    assert f'{96:,.{decimals}f}' in page and f'{108:,.{decimals}f}' in page
    assert '-4.0%' in page and '+8.0%' in page


@pytest.mark.parametrize('price, decimals', [(80_000, 2), (2.5, 4), (0.1259, 5), (0.00001234, 8)])
def test_price_precision_follows_the_price_level(price, decimals):
    assert price_precision([price]) == decimals
