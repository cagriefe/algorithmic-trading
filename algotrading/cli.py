import argparse
from pathlib import Path

import matplotlib
from dotenv import load_dotenv

from .analysis import run_analysis
from .data import create_client, fetch_klines, load_csv
from .price_action import ADX_TREND_THRESHOLD, SWING_LOOKBACK
from .report import print_signal_analysis

INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']


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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / f'{name}_analysis.csv'
    data.to_csv(analysis_path, index=False)
    print(f'\nSaved analysis to {analysis_path}')

    if not args.no_plot:
        import matplotlib.pyplot as plt
        from .plotting import plot_analysis

        fig = plot_analysis(data, title=f'{title} - Technical Analysis Overview')
        chart_path = args.output_dir / f'{name}_chart.png'
        fig.savefig(chart_path, dpi=120, bbox_inches='tight')
        print(f'Saved chart to {chart_path}')
        if args.show:
            plt.show()
        plt.close(fig)
    return 0
