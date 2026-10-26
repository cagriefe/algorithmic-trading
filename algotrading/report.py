def _print_signal(row):
    print(f"Date: {row['timestamp']}")
    print(f"Score: {row['Score']:.2f}")
    print(f"Confidence: {row['Confidence']:.1f}%")
    print(f"Decision: {row['Decision']}")
    print(f"Close Price: {row['close']}")
    print(f"Reasoning: {row['Reasoning'] or '-'}")
    print('-' * 80)


def print_signal_analysis(data, top=3):
    print('\n=== Trading Signal Analysis ===')

    print('\nCURRENT SIGNAL:')
    print('-' * 80)
    _print_signal(data.iloc[-1])

    significant_signals = data[data['Score'] > 3]
    if significant_signals.empty:
        print('No significant trading signals found')
        return

    print('\nHISTORICAL BEST SIGNALS:')
    print(f'Total signals analyzed: {len(data)}')
    print(f'Significant signals found: {len(significant_signals)}')
    print('-' * 80)
    for _, row in significant_signals.sort_values('Score', ascending=False).head(top).iterrows():
        print()
        _print_signal(row)
