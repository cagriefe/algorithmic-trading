import pandas as pd
import pytest

from algotrading.montecarlo import monte_carlo, trade_returns


def trades(*pnl):
    return pd.DataFrame({'pnl': pd.Series(pnl, dtype=float)})


def test_trade_returns_are_relative_to_equity_before_each_trade():
    assert trade_returns(trades(1000, -550, 1045), 10_000) == pytest.approx([0.1, -0.05, 0.1])


def test_identical_trade_returns_give_a_single_outcome():
    stats = monte_carlo(trades(100, 101, 102.01), 10_000, simulations=500)  # +1% of equity each time

    assert stats['Median return %'] == pytest.approx(3.0301)
    assert stats['Return 5th percentile %'] == pytest.approx(stats['Return 95th percentile %'])
    assert stats['Median max drawdown %'] == 0
    assert stats['Probability of loss %'] == 0


def test_outcome_distribution_matches_the_binomial_model():
    # +10% / -10% of equity, alternating: the number of winners in 4 resampled trades is Binomial(4, 1/2)
    stats = monte_carlo(trades(1000, -1100, 990, -1089), 10_000, simulations=4000, seed=1)

    assert stats['Median return %'] == pytest.approx((1.1 ** 2 * 0.9 ** 2 - 1) * 100)
    assert stats['Return 5th percentile %'] == pytest.approx((0.9 ** 4 - 1) * 100)
    assert stats['Return 95th percentile %'] == pytest.approx((1.1 ** 4 - 1) * 100)
    assert stats['Probability of loss %'] == pytest.approx(100 * 11 / 16, abs=3)
    assert stats['Max drawdown 5th percentile %'] <= stats['Median max drawdown %'] <= 0


def test_no_trades_means_no_simulation():
    assert monte_carlo(trades(), 10_000) is None
