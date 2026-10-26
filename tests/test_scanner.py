import pandas as pd
import pytest

from algotrading.backtest import BacktestConfig
from algotrading.cli import main
from algotrading.dashboard import market_breadth, sparkline, write_dashboard
from algotrading.scanner import scan_markets, top_symbols


def ticker(symbol, volume, low=10.0, high=12.0):
    return {'symbol': symbol, 'quoteVolume': str(volume), 'lowPrice': str(low), 'highPrice': str(high)}


class FakeClient:
    def __init__(self, tickers=()):
        self.tickers = list(tickers)
        self.pinged = False

    def tickers_24h(self):
        return self.tickers

    def ping(self):
        self.pinged = True
        return {}

    def copy(self):
        return self


@pytest.fixture
def fetch(market_data):
    def fetch(symbol):
        if symbol == 'BADUSDT':
            raise RuntimeError('Binance API error 400: Invalid symbol.')
        if symbol == 'NEWUSDT':
            return market_data.iloc[:30]
        return market_data if symbol == 'AAAUSDT' else market_data.iloc[:400]
    return fetch


def test_top_symbols_ranks_by_volume_without_stablecoins_and_fiat():
    client = FakeClient([
        ticker('BTCUSDT', 900), ticker('USDCUSDT', 5000, 0.999, 1.001), ticker('EURUSDT', 800, 1.05, 1.07),
        ticker('ETHUSDT', 700), ticker('NEWSTABLEUSDT', 600, 0.998, 1.002), ticker('ETHBTC', 10_000),
        ticker('SOLUSDT', 300), ticker('DOGEUSDT', 100),
    ])
    assert top_symbols(client, limit=3) == ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']


def test_scan_ranks_markets_and_isolates_failures(fetch):
    summary, details, errors = scan_markets(['AAAUSDT', 'BBBUSDT', 'BADUSDT', 'NEWUSDT'], fetch, workers=2)

    assert set(details) == {'AAAUSDT', 'BBBUSDT'}
    assert 'Invalid symbol' in errors['BADUSDT']
    assert 'bars of history' in errors['NEWUSDT']
    assert list(summary['Score']) == sorted(summary['Score'], reverse=True)
    assert {'Symbol', 'Decision', 'Score', 'Strategy %', 'Buy & hold %', 'Above SMA 200'} <= set(summary.columns)


def test_dashboard_links_a_report_for_every_market(fetch, tmp_path):
    summary, details, errors = scan_markets(['AAAUSDT', 'BBBUSDT', 'BADUSDT'], fetch)
    index = write_dashboard(tmp_path, summary, details, BacktestConfig(), '1d', simulations=200, errors=errors)
    page = index.read_text()

    assert '<table class="sortable">' in page
    assert 'href="reports/AAAUSDT.html"' in page and 'href="reports/BBBUSDT.html"' in page
    assert 'Skipped: BADUSDT' in page
    assert 'href="../index.html"' in (tmp_path / 'reports' / 'AAAUSDT.html').read_text()


def test_market_breadth():
    summary = pd.DataFrame({
        'Symbol': ['A', 'B', 'C', 'D'], 'Decision': ['Strong Long', 'Long', 'Neutral', 'Neutral'],
        'Above SMA 200': [True, True, False, None], 'RSI': [70, 50, 40, 30], 'Strategy %': [5, -1, 12, 0],
    })
    breadth = market_breadth(summary)

    assert (breadth['strong_long'], breadth['long']) == (['A'], ['B'])
    assert breadth['share_above_sma200'] == pytest.approx(200 / 3)
    assert breadth['regime'] == 'bullish'
    assert breadth['median_rsi'] == 45
    assert breadth['best']['Symbol'] == 'C'


def test_sparkline_colour_follows_the_direction():
    assert '#22c55e' in sparkline([1, 2, 3])
    assert '#ef4444' in sparkline([3, 2, 1])
    assert sparkline([1]) == ''


def test_scan_command_builds_the_dashboard(market_data, tmp_path, monkeypatch, capsys):
    client = FakeClient([ticker('AAAUSDT', 2), ticker('BBBUSDT', 1)])
    monkeypatch.setattr('algotrading.cli.create_client', lambda tld='com': client)
    monkeypatch.setattr('algotrading.cli.fetch_klines', lambda symbol, *args, **kwargs: market_data)
    out = tmp_path / 'site'

    assert main(['scan', '--top', '2', '--html', '--simulations', '100', '--output-dir', str(out)]) == 0

    assert 'Market Scan' in capsys.readouterr().out
    assert (out / 'scan.csv').exists() and (out / 'index.html').exists()
    assert sorted(p.name for p in (out / 'reports').iterdir()) == ['AAAUSDT.html', 'BBBUSDT.html']


def test_scan_command_with_explicit_symbols(market_data, tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr('algotrading.cli.create_client', lambda tld='com': client)
    monkeypatch.setattr('algotrading.cli.fetch_klines', lambda symbol, *args, **kwargs: market_data)

    assert main(['scan', '--symbols', 'aaausdt, bbbusdt', '--output-dir', str(tmp_path)]) == 0
    assert client.pinged
    assert sorted(pd.read_csv(tmp_path / 'scan.csv')['Symbol']) == ['AAAUSDT', 'BBBUSDT']
