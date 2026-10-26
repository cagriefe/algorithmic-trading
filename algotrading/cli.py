import argparse
from pathlib import Path

import matplotlib
from dotenv import load_dotenv

from .analysis import run_analysis
from .backtest import BacktestConfig, run_backtest
from .data import INTERVALS, create_client, fetch_klines, load_csv
from .price_action import ADX_TREND_THRESHOLD, SWING_LOOKBACK
from .report import print_backtest_report, print_signal_analysis


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Technical analysis, pattern recognition and signal scoring for Binance markets.')
    source = parser.add_argument_group('data')
    source.add_argument('--symbol', default='APEUSDT', help='Binance trading pair (default: %(default)s)')
    source.add_argument('--interval', default='1d', choices=INTERVALS, help='candle interval (default: %(default)s)')
    source.add_argument('--start', default='1 year ago UTC',
                        help='start of the history, e.g. "2024-01-01" or "6 months ago UTC" (default: %(default)s)')
    source.add_argument('--end', help='end of the history (default: now)')
    source.add_argument('--csv', type=Path, help='load OHLCV data from a CSV file instead of Binance')
    source.add_argument('--tld', default='com', help="Binance domain, use 'us' for Binance.US (default: %(default)s)")

    strategy = parser.add_argument_group('strategy')
    strategy.add_argument('--swing-lookback', type=int, default=SWING_LOOKBACK,
                          help='bars on each side of a swing pivot (default: %(default)s)')
    strategy.add_argument('--adx-threshold', type=float, default=ADX_TREND_THRESHOLD,
                          help='minimum ADX for a trending market (default: %(default)s)')

    defaults = BacktestConfig()
    backtest = parser.add_argument_group('backtest')
    backtest.add_argument('--no-backtest', action='store_true', help='only analyse signals, do not backtest')
    backtest.add_argument('--capital', type=float, default=defaults.initial_capital,
                          help='initial capital (default: %(default)s)')
    backtest.add_argument('--fee', type=float, default=defaults.fee_rate,
                          help='fee per fill as a fraction, 0.001 = 0.1%% (default: %(default)s)')
    backtest.add_argument('--slippage', type=float, default=defaults.slippage,
                          help='slippage per fill as a fraction (default: %(default)s)')
    backtest.add_argument('--risk', type=float, default=defaults.risk_per_trade,
                          help='share of equity risked per trade (default: %(default)s)')
    backtest.add_argument('--stop-atr', type=float, default=defaults.stop_loss_atr,
                          help='stop loss distance in ATRs (default: %(default)s)')
    backtest.add_argument('--take-profit-atr', type=float, default=defaults.take_profit_atr,
                          help='take profit distance in ATRs (default: %(default)s)')
    backtest.add_argument('--max-hold', type=int, help='close trades after this many bars (default: no limit)')
    backtest.add_argument('--no-breakdown-exit', action='store_true',
                          help='do not exit when price breaks below support')

    output = parser.add_argument_group('output')
    output.add_argument('--output-dir', type=Path, default=Path('output'),
                        help='directory for CSV and chart files (default: %(default)s)')
    output.add_argument('--top', type=int, default=3, help='historical best signals to print (default: %(default)s)')
    output.add_argument('--no-plot', action='store_true', help='do not create the chart')
    output.add_argument('--show', action='store_true', help='also open the chart in a window')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.show:
        matplotlib.use('Agg')
    load_dotenv()

    if args.csv:
        data = load_csv(args.csv)
        name = title = args.csv.stem
    else:
        client = create_client(args.tld)
        data = fetch_klines(args.symbol, args.interval, args.start, args.end, client=client)
        name, title = f'{args.symbol}_{args.interval}', f'{args.symbol} ({args.interval})'
    if data.empty:
        raise SystemExit('No market data returned; check the symbol, interval and date range.')

    data = run_analysis(data, lookback=args.swing_lookback, adx_threshold=args.adx_threshold)
    print_signal_analysis(data, top=args.top)

    result = None
    if not args.no_backtest:
        config = BacktestConfig(
            initial_capital=args.capital, fee_rate=args.fee, slippage=args.slippage, risk_per_trade=args.risk,
            stop_loss_atr=args.stop_atr, take_profit_atr=args.take_profit_atr, max_holding_bars=args.max_hold,
            exit_on_breakdown=not args.no_breakdown_exit)
        result = run_backtest(data, config)
        print_backtest_report(result)
        data['Equity'] = result.equity.to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print()
    analysis_path = args.output_dir / f'{name}_analysis.csv'
    data.to_csv(analysis_path, index=False)
    print(f'Saved analysis to {analysis_path}')
    if result is not None:
        trades_path = args.output_dir / f'{name}_trades.csv'
        result.trades.to_csv(trades_path, index=False)
        print(f'Saved trades to {trades_path}')

    if not args.no_plot:
        import matplotlib.pyplot as plt
        from .plotting import plot_analysis

        fig = plot_analysis(data, title=f'{title} - Technical Analysis Overview', backtest=result)
        chart_path = args.output_dir / f'{name}_chart.png'
        fig.savefig(chart_path, dpi=120, bbox_inches='tight')
        print(f'Saved chart to {chart_path}')
        if args.show:
            plt.show()
        plt.close(fig)
    return 0
