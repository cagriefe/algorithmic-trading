import argparse
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .analysis import run_analysis
from .backtest import BacktestConfig, run_backtest
from .dashboard import write_dashboard
from .data import INTERVALS, create_client, fetch_klines, load_csv
from .edge import signal_edge
from .html_report import write_report
from .montecarlo import monte_carlo
from .optimize import DEFAULT_GRID, OBJECTIVES, parameter_sets, walk_forward
from .price_action import ADX_TREND_THRESHOLD, SWING_LOOKBACK
from .report import (print_backtest_report, print_edge_report, print_monte_carlo, print_scan_report,
                     print_signal_analysis, print_walk_forward_report)
from .scanner import scan_markets, top_symbols


def _market_options():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('market data')
    group.add_argument('--interval', default='1d', choices=INTERVALS, help='candle interval (default: %(default)s)')
    group.add_argument('--start', default='1 year ago UTC',
                       help='start of the history, e.g. "2024-01-01" or "6 months ago UTC" (default: %(default)s)')
    group.add_argument('--end', help='end of the history (default: now)')
    group.add_argument('--tld', default='com', help="Binance domain, use 'us' for Binance.US (default: %(default)s)")
    return parser


def _strategy_options():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('strategy')
    group.add_argument('--swing-lookback', type=int, default=SWING_LOOKBACK,
                       help='bars on each side of a swing pivot (default: %(default)s)')
    group.add_argument('--adx-threshold', type=float, default=ADX_TREND_THRESHOLD,
                       help='minimum ADX for a trending market (default: %(default)s)')
    return parser


def _backtest_options():
    defaults = BacktestConfig()
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('backtest')
    group.add_argument('--capital', type=float, default=defaults.initial_capital,
                       help='initial capital (default: %(default)s)')
    group.add_argument('--fee', type=float, default=defaults.fee_rate,
                       help='fee per fill as a fraction, 0.001 = 0.1%% (default: %(default)s)')
    group.add_argument('--slippage', type=float, default=defaults.slippage,
                       help='slippage per fill as a fraction (default: %(default)s)')
    group.add_argument('--risk', type=float, default=defaults.risk_per_trade,
                       help='share of equity risked per trade (default: %(default)s)')
    group.add_argument('--stop-atr', type=float, default=defaults.stop_loss_atr,
                       help='stop loss distance in ATRs (default: %(default)s)')
    group.add_argument('--take-profit-atr', type=float, default=defaults.take_profit_atr,
                       help='take profit distance in ATRs (default: %(default)s)')
    group.add_argument('--max-hold', type=int, help='close trades after this many bars (default: no limit)')
    group.add_argument('--no-breakdown-exit', action='store_true', help='do not exit when price breaks below support')
    group.add_argument('--simulations', type=int, default=2000,
                       help='Monte Carlo resamples of the trades, 0 to skip (default: %(default)s)')
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Technical analysis, signal scoring, backtesting and market scanning for Binance.',
        epilog='Without a command "analyze" is used, e.g. python main.py --symbol BTCUSDT')
    parser.add_argument('--version', action='version', version=f'algotrading {__version__}')
    commands = parser.add_subparsers(dest='command', metavar='COMMAND')
    shared = [_market_options(), _strategy_options(), _backtest_options()]

    analyze = commands.add_parser('analyze', parents=shared, help='analyse and backtest one market (default)',
                                  description='Analyse and backtest one market.')
    analyze.add_argument('--symbol', default='APEUSDT', help='Binance trading pair (default: %(default)s)')
    analyze.add_argument('--csv', type=Path, help='load OHLCV data from a CSV file instead of Binance')
    analyze.add_argument('--no-backtest', action='store_true', help='only analyse signals, do not backtest')
    output = analyze.add_argument_group('output')
    output.add_argument('--output-dir', type=Path, default=Path('output'),
                        help='directory for CSV, chart and report files (default: %(default)s)')
    output.add_argument('--top', type=int, default=3, help='historical best signals to print (default: %(default)s)')
    output.add_argument('--edge', action='store_true', help='print forward returns after each signal type')
    output.add_argument('--html', action='store_true', help='also write an interactive HTML report')
    output.add_argument('--no-plot', action='store_true', help='do not create the PNG chart')
    output.add_argument('--show', action='store_true', help='also open the chart in a window')

    scan = commands.add_parser('scan', parents=shared, help='rank many markets by their current signal',
                               description='Analyse and backtest many markets and rank them by their current signal.')
    scan.add_argument('--symbols', help='comma-separated pairs, e.g. BTCUSDT,ETHUSDT (default: the most traded)')
    scan.add_argument('--top', type=int, default=20, help='number of most traded pairs to scan (default: %(default)s)')
    scan.add_argument('--quote', default='USDT', help='quote asset of the most traded pairs (default: %(default)s)')
    scan.add_argument('--workers', type=int, default=8, help='markets analysed in parallel (default: %(default)s)')
    scan.add_argument('--output-dir', type=Path, default=Path('output'),
                      help='directory for scan.csv and the dashboard (default: %(default)s)')
    scan.add_argument('--html', action='store_true', help='write a dashboard (index.html) and a report per market')

    optimize = commands.add_parser(
        'optimize', parents=shared, help='walk-forward parameter optimisation',
        description='Search stop loss, take profit, entry level and ADX threshold on expanding windows and test '
                    'the winners on the following, unseen periods.')
    optimize.set_defaults(start='3 years ago UTC')
    optimize.add_argument('--symbol', default='APEUSDT', help='Binance trading pair (default: %(default)s)')
    optimize.add_argument('--csv', type=Path, help='load OHLCV data from a CSV file instead of Binance')
    optimize.add_argument('--folds', type=int, default=4, help='number of out-of-sample test periods (default: %(default)s)')
    optimize.add_argument('--objective', default='sharpe', choices=OBJECTIVES,
                          help='what the optimiser maximises (default: %(default)s)')
    optimize.add_argument('--min-trades', type=int, default=3,
                          help='minimum trades for a parameter set to be eligible (default: %(default)s)')
    optimize.add_argument('--output-dir', type=Path, default=Path('output'),
                          help='directory for the CSV and chart files (default: %(default)s)')
    optimize.add_argument('--no-plot', action='store_true', help='do not create the PNG chart')
    return parser


def backtest_config(args):
    return BacktestConfig(
        initial_capital=args.capital, fee_rate=args.fee, slippage=args.slippage, risk_per_trade=args.risk,
        stop_loss_atr=args.stop_atr, take_profit_atr=args.take_profit_atr, max_holding_bars=args.max_hold,
        exit_on_breakdown=not args.no_breakdown_exit)


def load_market(args):
    """OHLCV data from --csv or Binance, plus a file name stem and a title."""
    if args.csv:
        data = load_csv(args.csv)
        name = title = args.csv.stem
    else:
        data = fetch_klines(args.symbol, args.interval, args.start, args.end, client=create_client(args.tld))
        name, title = f'{args.symbol}_{args.interval}', f'{args.symbol} ({args.interval})'
    if data.empty:
        raise SystemExit('No market data returned; check the symbol, interval and date range.')
    return data, name, title


def run_analyze(args):
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')

    data, name, title = load_market(args)
    data = run_analysis(data, lookback=args.swing_lookback, adx_threshold=args.adx_threshold)
    print_signal_analysis(data, top=args.top)
    edge = signal_edge(data) if args.edge or args.html else None
    if args.edge:
        print_edge_report(edge)

    config = backtest_config(args)
    result = simulations = None
    if not args.no_backtest:
        result = run_backtest(data, config)
        print_backtest_report(result)
        simulations = monte_carlo(result.trades, config.initial_capital, args.simulations)
        print_monte_carlo(simulations)
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
    if args.html:
        report_path = args.output_dir / f'{name}_report.html'
        write_report(report_path, data, result, config, simulations, edge, title=title)
        print(f'Saved interactive report to {report_path}')

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


def run_scan(args):
    client = create_client(args.tld)
    if args.symbols:
        symbols = [symbol.strip().upper() for symbol in args.symbols.split(',') if symbol.strip()]
        client.ping()  # resolve the reachable host once before the threads start
        subtitle = f'{len(symbols)} Binance markets'
    else:
        symbols = top_symbols(client, args.top, args.quote)
        subtitle = f'Top {len(symbols)} Binance {args.quote} markets by 24h volume'
    subtitle += f' · {args.interval} candles since {args.start}'
    print(f'Scanning {len(symbols)} markets ({args.interval} candles since {args.start})...')

    local = threading.local()

    def fetch(symbol):
        if not hasattr(local, 'client'):
            local.client = client.copy()
        return fetch_klines(symbol, args.interval, args.start, args.end, client=local.client)

    config = backtest_config(args)
    summary, details, errors = scan_markets(symbols, fetch, config, workers=args.workers,
                                            lookback=args.swing_lookback, adx_threshold=args.adx_threshold)
    print_scan_report(summary, errors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / 'scan.csv', index=False)
    print(f'\nSaved ranking to {args.output_dir / "scan.csv"}')
    if args.html:
        index = write_dashboard(args.output_dir, summary, details, config, args.interval, args.simulations,
                                subtitle=subtitle, errors=errors)
        print(f'Saved dashboard to {index}')
    return 0 if details else 1


def run_optimize(args):
    data, name, title = load_market(args)
    print(f'Walk-forward optimisation of {title}: {len(parameter_sets(DEFAULT_GRID))} parameter sets, '
          f'{args.folds} folds, {len(data)} bars...')
    try:
        result = walk_forward(data, backtest_config(args), folds=args.folds, objective=args.objective,
                              min_trades=args.min_trades, lookback=args.swing_lookback, default_adx=args.adx_threshold)
    except ValueError as error:
        raise SystemExit(str(error))
    print_walk_forward_report(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print()
    for suffix, table in [('walkforward', result.folds), ('grid', result.grid)]:
        path = args.output_dir / f'{name}_{suffix}.csv'
        table.to_csv(path, index=False)
        print(f'Saved {suffix} results to {path}')
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from .plotting import plot_walk_forward

        fig = plot_walk_forward(result, title=f'{title} - walk-forward optimisation ({args.objective})')
        chart_path = args.output_dir / f'{name}_walkforward.png'
        fig.savefig(chart_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved chart to {chart_path}')
    return 0


HANDLERS = {'analyze': run_analyze, 'scan': run_scan, 'optimize': run_optimize}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in (*HANDLERS, '-h', '--help', '--version'):
        argv.insert(0, 'analyze')
    args = build_parser().parse_args(argv)
    load_dotenv()
    return HANDLERS[args.command](args)
