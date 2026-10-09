"""Synthetic two-strategy information-ratio allocation and analytical combination.

Run: python examples/strategy_allocation.py
Every number is generated from a fixed seed for illustration; none is market data
or evidence about any strategy. No downloads, file writes or plots.

Limitations of combining already-costed strategy returns (also in the result
metadata): fixed weights mean the sleeves are rebalanced every period; no
reallocation costs, trade netting, shared collateral or financing offsets are
modeled; and scaling a strategy's returns does not recalculate its market impact
or borrowing, so strategies with fixed dollar targets or nonlinear costs may not
scale proportionally. The combination is not an executable account backtest.
"""
from datetime import date, timedelta
import random

import polars as pl
import research_toolkit as rt


def strategy_report(returns, days, capital, label):
    """A net P&L series from an account of its own size, reported through series_performance."""
    nav, pnl = capital, []
    for r in returns:
        pnl.append(nav*r)
        nav += pnl[-1]
    daily = pl.DataFrame({"period_start": days[:-1], "session": days[1:], "pnl": pnl})
    return rt.series_performance(daily, initial_capital=capital, periods_per_year=252,
                                 risk_free_annual_effective=0.0,
                                 metadata={"source": f"synthetic {label} P&L", "currency": "USD"})


def run_example(seed=21, n_days=500, estimation_days=250, bounds=(0.10, 0.90)):
    rng = random.Random(seed)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days + 1)]
    common = [rng.gauss(0., .006) for _ in range(n_days)]
    trend = [.0006 + c + rng.gauss(0., .008) for c in common]
    carry = [.0003 + .4*c + rng.gauss(0., .004) for c in common]
    # Different original account sizes: only their returns are combined.
    strategies = {"trend": strategy_report(trend, days, 2_000_000., "trend"),
                  "carry": strategy_report(carry, days, 30_000_000., "carry")}

    allocation_kw = dict(benchmark="zero", periods_per_year=252,
                         bounds={name: bounds for name in strategies},
                         estimation_window=(days[0], days[estimation_days]))
    assumed = rt.information_ratio_weights(strategies, covariance="zero_correlation", **allocation_kw)
    empirical = rt.information_ratio_weights(strategies, covariance="estimated", **allocation_kw)

    # Apply each set of fitted weights, without re-estimation, to a separate later window.
    evaluation = (days[estimation_days], days[n_days])
    combinations = {label: rt.combine_strategies(strategies, weights=fit, evaluation_window=evaluation,
                                                 benchmark="zero", initial_capital=1_000_000.)
                    for label, fit in (("zero_correlation", assumed), ("estimated", empirical))}
    reports = {label: rt.series_performance(c.illustrative, initial_capital=c.metadata["initial_capital"],
                                            periods_per_year=252, risk_free_annual_effective=0.0,
                                            benchmark=c.benchmark, benchmark_metadata=c.metadata["benchmark_metadata"],
                                            metadata=c.metadata["series_metadata"], frequency=c.metadata["frequency"])
               for label, c in combinations.items()}
    comparison = rt.compare_performance(reports)
    return assumed, empirical, combinations, comparison


if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    assumed, empirical, combinations, comparison = run_example()
    for label, fit in (("Zero-correlation assumption", assumed), ("Estimated correlation", empirical)):
        print(f"{label}: weights\n", fit.weights)
        print("Assumed objective versus observed combined returns (estimation sample)\n", fit.summary)
    print("Estimated active-return statistics\n", empirical.estimates)
    print("Empirical versus optimization covariance (annualized)\n", assumed.covariance)
    print("Evaluation sample label:", combinations["estimated"].metadata["sample"])
    print("Later-window information ratios\n", comparison.values.filter(pl.col("metric") == "information_ratio"))
    print("Limitations\n", "\n".join(f"- {item}" for item in combinations["estimated"].metadata["limitations"]))
