"""Offline, synthetic notebook-style workflow; no assignment or provider inputs.

Run python examples/research_workflow.py. The main block explicitly exports charts
and tables to artifacts/research-workflow; library calls themselves perform no I/O.
"""
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import polars as pl
import research_toolkit as rt

ROOT = Path(__file__).resolve().parents[1]


def run_example(*, lookback=40, warmup_sessions=60, cap=.23, annualization=252,
                initial_capital=1_000_000., leverages=(1., 1.5)):
    market = rt.load_snapshot(ROOT/"examples/snapshots/v1/equities")
    benchmark_market = rt.load_snapshot(ROOT/"examples/snapshots/v1/benchmark")
    dates = market.sessions["session"].to_list()
    decision, entry, end = dates[warmup_sessions-1], dates[warmup_sessions], dates[-1]
    asset_returns = rt.returns(market, method="simple", basis="total_return", dividend_policy="reinvest_ex_close")
    allocation = rt.inverse_volatility_weights(asset_returns, decision_session=decision,
        lookback=lookback, periods_per_year=annualization, max_asset_weight=cap, cap_policy="redistribute")
    # Explicit synthetic liquidity assumption, not measured Yahoo volume.
    dollar_volume = market.prices.select("session", "asset").join(market.sessions, on="session", how="left").select(
        "session", "asset", pl.lit(20_000_000.).alias("dollar_volume"), pl.col("close_at").alias("available_at"))
    liquidity = rt.estimate_liquidity(asset_returns, dollar_volume, decision_session=decision, lookback=lookback,
        metadata={"source": "synthetic constant dollar volume assumption", "currency": market.metadata["currency"]})
    costs = rt.SquareRootImpactCosts(liquidity=liquidity, commission_bps=1., commission_per_share=.002,
        half_spread_bps=3., impact_coefficient=.5)
    marks = market.prices.filter(pl.col("session") == entry).select("asset", pl.col("close").alias("reference_price"))
    financing = rt.Financing(cash_rate=.01, borrowing_rate=.06, day_count="ACT/365F",
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")
    reinvestment = rt.DividendReinvestment(execution="first_close_on_or_after_payment", funding="before_debt_repayment",
        scheduled_collision="rebalance_only", terminal_action="hold_cash")
    benchmark = rt.returns(benchmark_market, method="simple", basis="total_return").values.filter(
        (pl.col("period_start") >= entry) & (pl.col("session") <= end)).select("period_start", "session", "simple_return")
    bm_meta = dict(source="committed synthetic benchmark", basis="total return; no execution costs",
                   currency=market.metadata["currency"], frequency="1d")
    # Already assigned daily annual nominal rates; every weekend date is included.
    accrual_dates = [entry+timedelta(days=i) for i in range(1, (end-entry).days+1)]
    daily_rates = pl.DataFrame({"date": accrual_dates,
        "annual_rate": [.03 if i < len(accrual_dates)//2 else .04 for i in range(len(accrual_dates))],
        "available_at": [datetime.combine(d-timedelta(days=1), time.min, tzinfo=timezone.utc) for d in accrual_dates]})
    rf = rt.risk_free_returns(daily_rates, intervals=benchmark.select("period_start", "session"), day_count="ACT/360",
        compounding="daily", rate_timing="known_at_accrual_start", timezone_name=market.metadata["timezone"],
        metadata=dict(source="synthetic changing reference rates, not loan rates", basis="calendar accrual",
            currency=market.metadata["currency"], frequency="1d"))
    runs, reports, previews = {}, {}, {}
    for leverage in leverages:
        name = f"{leverage:g}x"
        previews[name] = rt.size_entry_orders(weights=allocation.weights, initial_capital=initial_capital,
            gross_leverage=leverage, prices=marks, costs=costs, execution_session=entry, decision_session=decision)
        runs[name] = rt.buy_and_hold(market, weights=allocation.weights, initial_capital=initial_capital,
            entry_session=entry, end_session=end, costs=costs, financing=financing, dividend_reinvestment=reinvestment,
            policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
                initial_gross_leverage=leverage, terminal_action="mark_only")).require_complete()
        reports[name] = rt.performance(runs[name], periods_per_year=annualization, risk_free_returns=rf,
            sharpe_denominator="excess_returns", minimum_acceptable_return_annual_effective=0.,
            benchmark=benchmark, benchmark_metadata=bm_meta)
    rolling = rt.rolling_risk(next(iter(runs.values())), window=lookback, periods_per_year=annualization,
        risk_free_returns=rf, sharpe_denominator="excess_returns", benchmark=benchmark, benchmark_metadata=bm_meta)
    comparison = rt.compare_performance(reports, benchmark_label="Synthetic benchmark")
    return dict(allocation=allocation, liquidity=liquidity, previews=previews, runs=runs, reports=reports,
                risk_free=rf, rolling=rolling, comparison=comparison)


def main():
    output = run_example()
    destination = ROOT/"artifacts/research-workflow"
    destination.mkdir(parents=True, exist_ok=True)
    output["comparison"].values.write_csv(destination/"comparison.csv")
    for label, run in output["runs"].items():
        run.daily.write_csv(destination/f"daily-{label}.csv")
        run.execution_costs.write_csv(destination/f"execution-costs-{label}.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = next(iter(output["reports"].values()))
    for method in ("sum", "compound", "wealth"):
        fig, _ = rt.plots.cumulative_returns(report, method=method)
        fig.savefig(destination/f"cumulative-{method}.png", dpi=130)
        plt.close(fig)
    for metric in ("beta", "correlation"):
        fig, _ = rt.plots.rolling_risk(output["rolling"], metric=metric)
        fig.savefig(destination/f"rolling-{metric}.png", dpi=130)
        plt.close(fig)
    print(output["allocation"].weights)
    print(output["comparison"].values)
    print(destination)


if __name__ == "__main__":
    main()
