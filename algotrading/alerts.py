import json
import os
import re

import pandas as pd
import requests

from .html_report import price_precision

MAX_HISTORY_BARS = 300
TELEGRAM_LIMIT, DISCORD_LIMIT = 4096, 2000


def interval_step(interval):
    """Length of one candle, e.g. 4 hours for '4h' and one calendar month for '1M'."""
    if interval == '1M':
        return pd.DateOffset(months=1)
    units = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
    return pd.Timedelta(**{units[interval[-1]]: int(interval[:-1])})


def utc_now():
    return pd.Timestamp.now(tz='UTC').tz_localize(None)


def history_start(interval, bars=MAX_HISTORY_BARS, now=None):
    """Start time that gives enough candles for every indicator."""
    return (now or utc_now()) - bars * interval_step(interval)


def seconds_until_next_close(last_open, interval, now=None, buffer=10):
    """Seconds until the candle after the last closed one (opened at `last_open`) has closed."""
    next_close = pd.Timestamp(last_open) + 2 * interval_step(interval)
    return max((next_close - (now or utc_now())).total_seconds(), 0) + buffer


def format_signal(symbol, interval, row, config):
    """Readable alert for an entry signal, with an ATR-based stop and target."""
    close, atr = row['close'], row['ATR']
    stop, target = close - config.stop_loss_atr * atr, close + config.take_profit_atr * atr
    decimals = price_precision([close])
    reasons = re.sub(r' \(weight: [\d.]+\)', '', row['Reasoning']) or '-'
    icon = '\U0001F7E2' if row['Decision'] == 'Strong Long' else '\U0001F7E1'
    return '\n'.join([
        f'{icon} {row["Decision"]} signal: {symbol} ({interval})',
        f'Close {close:,.{decimals}f} | score {row["Score"]:.1f}/21 ({row["Confidence"]:.0f}%)',
        f'Plan: stop {stop:,.{decimals}f} ({(stop / close - 1) * 100:+.1f}%), '
        f'target {target:,.{decimals}f} ({(target / close - 1) * 100:+.1f}%)',
        f'Why: {reasons}',
        f'Candle: {pd.Timestamp(row["timestamp"]):%Y-%m-%d %H:%M} UTC',
    ])


def fresh_signals(markets, interval, config, state=None):
    """Alert texts for markets whose last closed candle is an entry signal not alerted before.

    `markets` maps symbol -> analysed data. `state` maps 'SYMBOL:interval' to the last alerted
    candle and is updated in place, so the same candle never triggers twice.
    """
    state = {} if state is None else state
    messages = []
    for symbol, data in markets.items():
        last = data.iloc[-1]
        key, candle = f'{symbol}:{interval}', str(last['timestamp'])
        if last['Decision'] in config.entry_decisions and state.get(key) != candle:
            messages.append(format_signal(symbol, interval, last, config))
            state[key] = candle
    return messages


def format_digest(messages, interval, dashboard_url=None):
    header = f'\U0001F4CA {len(messages)} new signal{"s" if len(messages) != 1 else ""} on {interval} candles'
    footer = [f'Dashboard: {dashboard_url}'] if dashboard_url else []
    return '\n\n'.join([header, *messages, *footer])


def load_state(path):
    return json.loads(path.read_text(encoding='utf-8')) if path and path.exists() else {}


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding='utf-8')


def _truncate(text, limit):
    return text if len(text) <= limit else text[:limit - 1] + '\u2026'


def _post(session, url, payload):
    # Never include the URL in errors: it contains the bot token or webhook secret
    try:
        response = session.post(url, json=payload, timeout=15)
    except requests.RequestException as error:
        raise RuntimeError(f'request failed ({type(error).__name__})') from None
    if not response.ok:
        raise RuntimeError(f'HTTP {response.status_code}')


class ConsoleNotifier:
    name = 'console'

    def send(self, text):
        print(text)


class TelegramNotifier:
    name = 'telegram'

    def __init__(self, token, chat_id, session=None):
        self.url = f'https://api.telegram.org/bot{token}/sendMessage'
        self.chat_id = chat_id
        self.session = session or requests.Session()

    def send(self, text):
        _post(self.session, self.url, {'chat_id': self.chat_id, 'text': _truncate(text, TELEGRAM_LIMIT),
                                       'disable_web_page_preview': True})


class DiscordNotifier:
    name = 'discord'

    def __init__(self, webhook_url, session=None):
        self.url = webhook_url
        self.session = session or requests.Session()

    def send(self, text):
        _post(self.session, self.url, {'content': _truncate(text, DISCORD_LIMIT)})


def notifiers_from_env(environ=None):
    """Console output, plus Telegram and Discord when their environment variables are set."""
    environ = os.environ if environ is None else environ
    notifiers = [ConsoleNotifier()]
    if environ.get('TELEGRAM_BOT_TOKEN') and environ.get('TELEGRAM_CHAT_ID'):
        notifiers.append(TelegramNotifier(environ['TELEGRAM_BOT_TOKEN'], environ['TELEGRAM_CHAT_ID']))
    if environ.get('DISCORD_WEBHOOK_URL'):
        notifiers.append(DiscordNotifier(environ['DISCORD_WEBHOOK_URL']))
    return notifiers


def broadcast(notifiers, text):
    """Send to every channel; returns the failures so one broken channel does not block the rest."""
    failures = []
    for notifier in notifiers:
        try:
            notifier.send(text)
        except Exception as error:
            failures.append(f'{notifier.name}: {error}')
    return failures
