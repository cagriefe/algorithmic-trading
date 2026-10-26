from algotrading.cli import main


def write_csv(market_data, tmp_path):
    csv_path = tmp_path / 'synthetic.csv'
    market_data.to_csv(csv_path, index=False)
    return csv_path


def test_cli_runs_offline_from_csv(market_data, tmp_path, capsys):
    out = tmp_path / 'out'

    assert main(['--csv', str(write_csv(market_data, tmp_path)), '--output-dir', str(out), '--edge', '--html']) == 0

    printed = capsys.readouterr().out
    for section in ['CURRENT SIGNAL', 'Signal Edge', 'Backtest Results', 'Monte Carlo']:
        assert section in printed
    for suffix in ['analysis.csv', 'trades.csv', 'chart.png', 'report.html']:
        assert (out / f'synthetic_{suffix}').exists()


def test_cli_can_skip_backtest_and_chart(market_data, tmp_path):
    out = tmp_path / 'out'
    argv = ['--csv', str(write_csv(market_data, tmp_path)), '--output-dir', str(out), '--no-backtest', '--no-plot']

    assert main(argv) == 0
    assert [p.name for p in out.iterdir()] == ['synthetic_analysis.csv']
