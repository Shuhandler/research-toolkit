"""Offline long-portfolio/hedge comparison using the repository's synthetic data.

Run: python examples/long_short_hedge.py
All rates and positions below are example assumptions, not historical estimates.
No data downloads, file exports, or automatic plotting.
"""
from dataclasses import replace
from pathlib import Path

import polars as pl
import research_toolkit as rt


def run_example():
    market = rt.load_snapshot(Path(__file__).resolve().parent / "snapshots/v1/equities")
    start, end = market.sessions["session"][0], market.sessions["session"][-1]
    capital = 100_000.
    costs = rt.TradeCosts(commission_bps=2., half_spread_bps=3., impact_bps=1.)
    finance = rt.Financing(cash_rate=0., borrowing_rate=.06, day_count="ACT/365F",
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")
    entry_policy = rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
        fractional_shares=True, initial_gross_leverage=1.5, terminal_action="mark_only")
    common = dict(initial_capital=capital, entry_session=start, end_session=end, costs=costs)
    unhedged = rt.buy_and_hold(market, weights=rt.equal_weights(["SYNTH_A", "SYNTH_B", "SYNTH_C", "SYNTH_D"]),
        policy=entry_policy, financing=finance, **common).require_complete()

    # Keep the exact long quantities. The caller supplies the hedge; no optimizer.
    quantities = dict(unhedged.positions.filter(pl.col("session") == start)
                      .select("asset", "quantity").iter_rows())
    hedge_price = market.prices.filter((pl.col("session") == start) & (pl.col("asset") == "SYNTH_E"))["close"].item()
    quantities["SYNTH_E"] = -.30 * capital / hedge_price
    hedged = rt.buy_and_hold(market, quantities=quantities,
        policy=replace(entry_policy, initial_gross_leverage=1.),
        financing=replace(finance, maintenance_equity_ratio=None),
        long_short=rt.LongShortPolicy(collateral_multiple=1.02, long_margin=.25,
            short_margin=.30, rebate_rate=0., rebate_day_count="ACT/360"),
        stock_borrow=rt.StockBorrow(rates={"SYNTH_E": .02}, day_count="ACT/360",
            metadata={"source": "explicit example fee assumption", "basis": "modeled"}),
        **common).require_complete()

    # Supply any exactly aligned benchmark here; zero returns are useful for IR.
    zero_benchmark = unhedged.daily.select("period_start", "session", pl.lit(0.).alias("simple_return"))
    reports = {name: rt.performance(run, periods_per_year=252, risk_free_annual_effective=0.,
        minimum_acceptable_return_annual_effective=0., benchmark=zero_benchmark,
        benchmark_metadata={"source": "explicit zero-return comparator", "basis": "zero return",
            "currency": "USD", "frequency": "1d"}) for name, run in {"unhedged":unhedged, "hedged":hedged}.items()}
    comparison = rt.compare_performance(reports)
    cost_breakdown = hedged.costs.group_by("component").agg(pl.col("amount").sum()).sort("component")
    return {"unhedged":unhedged, "hedged":hedged}, reports, comparison, cost_breakdown


if __name__ == "__main__":
    runs, reports, comparison, cost_breakdown = run_example()
    print(cost_breakdown)
    print(comparison.values.filter(pl.col("metric").is_in([
        "ending_equity", "compounded_return", "information_ratio", "annualized_tracking_error"])))
