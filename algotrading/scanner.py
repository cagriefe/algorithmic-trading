from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .analysis import run_analysis
from .backtest import BacktestConfig, run_backtest

# Stablecoins and fiat currencies quoted in USDT move like cash, not like crypto markets
NON_CRYPTO_BASES = {
    'USDC', 'FDUSD', 'TUSD', 'BUSD', 'DAI', 'USDP', 'USDD', 'PYUSD', 'USD1', 'USDE', 'RLUSD', 'BFUSD', 'XUSD',
    'EUR', 'EURI', 'AEUR', 'GBP', 'TRY', 'BRL', 'JPY', 'AUD', 'ARS', 'MXN', 'PLN', 'RON', 'UAH', 'ZAR', 'COP',
    'CZK', 'IDR', 'NGN', 'RUB',
}
MIN_BARS = 60
SIGNALS = ('Long', 'Strong Long')


def _pegged_to_one(ticker):
    # New stablecoins appear all the time, so also skip anything that spent the day within 1% of 1.00
    low, high = float(ticker['lowPrice']), float(ticker['highPrice'])
    return 0.99 <= low and high <= 1.01


def top_symbols(client, limit=20, quote='USDT'):
    """The most traded pairs against `quote` by 24h quote volume, without stablecoins and fiat."""
    ranked = []
    for ticker in client.tickers_24h():
        symbol = ticker['symbol']
        base = symbol[:-len(quote)]
        if (symbol.endswith(quote) and base and base not in NON_CRYPTO_BASES
                and not (quote in {'USDT', 'USDC'} and _pegged_to_one(ticker))):
            ranked.append((float(ticker['quoteVolume']), symbol))
    return [symbol for _, symbol in sorted(ranked, reverse=True)[:limit]]


def analyse_market(symbol, fetch, config, **analysis_options):
    data = fetch(symbol)
    if len(data) < MIN_BARS:
        raise ValueError(f'only {len(data)} bars of history, need at least {MIN_BARS}')
    data = run_analysis(data, **analysis_options)
    return data, run_backtest(data, config)


def _summary_row(symbol, data, result):
    last = data.iloc[-1]
    metrics = result.metrics
    signal_bars = np.flatnonzero(data['Decision'].isin(SIGNALS).to_numpy())
    sma_200 = last['SMA_200']
    return {
        'Symbol': symbol,
        'Close': last['close'],
        'Change %': (last['close'] / data['close'].iloc[-2] - 1) * 100,
        'Decision': last['Decision'],
        'Score': last['Score'],
        'Pattern': last['Pattern'],
        'Reasoning': last['Reasoning'],
        'Bars since signal': len(data) - 1 - signal_bars[-1] if len(signal_bars) else np.nan,
        'RSI': last['RSI'],
        'Above SMA 200': bool(last['close'] > sma_200) if pd.notna(sma_200) else None,
        'Strategy %': metrics['Total return %'],
        'Buy & hold %': metrics['Buy & hold return %'],
        'Max DD %': metrics['Max drawdown %'],
        'Win rate %': metrics['Win rate %'],
        'Profit factor': metrics['Profit factor'],
        'Trades': metrics['Trades'],
    }


def scan_markets(symbols, fetch, config=None, workers=8, **analysis_options):
    """Analyse and backtest every symbol concurrently.

    `fetch(symbol)` must return OHLCV data. Returns a summary ranked by the current signal score
    (then by backtest return), the analysed data and backtest per symbol, and the errors of
    markets that could not be analysed, so one bad market never stops the scan.
    """
    config = config or BacktestConfig()
    details, errors = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {symbol: pool.submit(analyse_market, symbol, fetch, config, **analysis_options)
                   for symbol in symbols}
        for symbol, future in futures.items():
            try:
                details[symbol] = future.result()
            except Exception as error:
                errors[symbol] = str(error)

    summary = pd.DataFrame([_summary_row(symbol, *details[symbol]) for symbol in details])
    if not summary.empty:
        summary = summary.sort_values(['Score', 'Strategy %'], ascending=False, ignore_index=True)
    return summary, details, errors
