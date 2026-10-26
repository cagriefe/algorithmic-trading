import json

import pandas as pd
import pytest
import requests

from algotrading.alerts import (ConsoleNotifier, DiscordNotifier, TelegramNotifier, broadcast, format_digest,
                                format_signal, fresh_signals, interval_step, notifiers_from_env,
                                seconds_until_next_close)
from algotrading.backtest import BacktestConfig
from algotrading.cli import main

SECRET = '123456:SECRET-TOKEN'


def market(decisions, start='2024-01-01'):
    n = len(decisions)
    return pd.DataFrame({
        'timestamp': pd.date_range(start, periods=n, freq='4h'),
        'close': 100.0, 'ATR': 2.5, 'Decision': decisions, 'Score': 14.0, 'Confidence': 66.7,
        'Reasoning': 'Trendline breakout (weight: 5.0) | Swing point breakout (weight: 4.0)',
    })


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.ok = status_code < 400


class FakeSession:
    def __init__(self, status_code=200, error=None):
        self.status_code, self.error, self.posts = status_code, error, []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if self.error:
            raise self.error
        return FakeResponse(self.status_code)


def test_signal_message_contains_a_trade_plan():
    text = format_signal('BTCUSDT', '4h', market(['Strong Long']).iloc[-1], BacktestConfig())

    assert text.splitlines()[0].endswith('Strong Long signal: BTCUSDT (4h)')
    assert 'stop 95.00 (-5.0%), target 110.00 (+10.0%)' in text
    assert 'Why: Trendline breakout | Swing point breakout' in text
    assert 'weight' not in text


def test_only_new_signals_on_the_last_candle_alert_once():
    markets = {'AAAUSDT': market(['Long', 'Neutral']), 'BBBUSDT': market(['Neutral', 'Long'])}
    state = {}

    first = fresh_signals(markets, '4h', BacktestConfig(), state)
    again = fresh_signals(markets, '4h', BacktestConfig(), state)

    assert len(first) == 1 and 'BBBUSDT' in first[0]
    assert again == []
    assert state == {'BBBUSDT:4h': '2024-01-01 04:00:00'}


def test_strong_long_only_configuration_ignores_long_signals():
    config = BacktestConfig(entry_decisions=('Strong Long',))
    assert fresh_signals({'AAAUSDT': market(['Long'])}, '1d', config) == []


def test_digest_lists_signals_and_the_dashboard():
    text = format_digest(['one', 'two'], '1d', 'https://example.org/board')
    assert text.startswith('\U0001F4CA 2 new signals on 1d candles')
    assert text.endswith('Dashboard: https://example.org/board')


def test_channels_come_from_the_environment():
    assert [n.name for n in notifiers_from_env({})] == ['console']
    environ = {'TELEGRAM_BOT_TOKEN': SECRET, 'TELEGRAM_CHAT_ID': '42', 'DISCORD_WEBHOOK_URL': 'https://discord/x'}
    assert [n.name for n in notifiers_from_env(environ)] == ['console', 'telegram', 'discord']
    assert [n.name for n in notifiers_from_env({'TELEGRAM_BOT_TOKEN': SECRET})] == ['console']


def test_telegram_and_discord_payloads():
    telegram, discord = FakeSession(), FakeSession()
    TelegramNotifier(SECRET, '42', telegram).send('hello')
    DiscordNotifier('https://discord.com/api/webhooks/1/abc', discord).send('x' * 2500)

    assert telegram.posts == [(f'https://api.telegram.org/bot{SECRET}/sendMessage',
                               {'chat_id': '42', 'text': 'hello', 'disable_web_page_preview': True})]
    assert len(discord.posts[0][1]['content']) == 2000


@pytest.mark.parametrize('session', [FakeSession(status_code=401),
                                     FakeSession(error=requests.ConnectionError(f'url: /bot{SECRET}/sendMessage'))])
def test_delivery_errors_never_leak_the_token(session):
    failures = broadcast([TelegramNotifier(SECRET, '42', session)], 'hello')

    assert len(failures) == 1 and failures[0].startswith('telegram: ')
    assert SECRET not in failures[0]


def test_one_broken_channel_does_not_block_the_others(capsys):
    failures = broadcast([TelegramNotifier(SECRET, '42', FakeSession(500)), ConsoleNotifier()], 'hello')

    assert failures == ['telegram: HTTP 500']
    assert capsys.readouterr().out == 'hello\n'


def test_candle_timing():
    assert interval_step('15m') == pd.Timedelta(minutes=15)
    assert interval_step('1w') == pd.Timedelta(weeks=1)
    assert pd.Timestamp('2024-01-31') + interval_step('1M') == pd.Timestamp('2024-02-29')
    # last closed 4h candle opened 00:00, the next one closes 08:00
    assert seconds_until_next_close('2024-01-01 00:00', '4h', now=pd.Timestamp('2024-01-01 05:00'), buffer=10) == \
        3 * 3600 + 10


def test_watch_once_alerts_and_remembers_the_candle(tmp_path, monkeypatch, capsys):
    analysed = market(['Neutral', 'Strong Long'])
    monkeypatch.setattr('algotrading.cli.create_client', lambda tld='com': object())
    monkeypatch.setattr('algotrading.cli.fetch_klines', lambda *args, **kwargs: analysed)
    monkeypatch.setattr('algotrading.cli.run_analysis', lambda data, **kwargs: data)
    for name in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DISCORD_WEBHOOK_URL']:
        monkeypatch.delenv(name, raising=False)
    state_file = tmp_path / 'state.json'
    argv = ['watch', '--symbols', 'BTCUSDT', '--once', '--state-file', str(state_file)]

    assert main(argv) == 0
    assert 'Strong Long signal: BTCUSDT (4h)' in capsys.readouterr().out
    assert json.loads(state_file.read_text()) == {'BTCUSDT:4h': '2024-01-01 04:00:00'}

    assert main(argv) == 0
    assert 'no new signals' in capsys.readouterr().out
