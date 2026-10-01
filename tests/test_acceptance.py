"""Whole-year saved-input checks, complementary to the small hand oracles."""
from pathlib import Path
import math

import polars as pl
import pytest
import research_toolkit as rt

SNAPSHOTS = Path(__file__).resolve().parents[1]/"examples"/"snapshots"/"v1"


@pytest.mark.parametrize("leverage", [0., 1., 1.5, 2.])
def test_offline_year(leverage):
    market = rt.load_snapshot(SNAPSHOTS/"equities")
    bm = rt.load_snapshot(SNAPSHOTS/"benchmark")
    dates = market.sessions["session"].to_list()
    run = rt.buy_and_hold(market, weights=rt.equal_weights(market.prices["asset"].unique()),
        initial_capital=100_000_000., entry_session=dates[0], end_session=dates[-1],
        policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
            fractional_shares=True, initial_gross_leverage=leverage, terminal_action="mark_only"),
        costs=rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=3.),
        financing=rt.Financing(cash_rate=.02, borrowing_rate=.06, day_count="ACT/365F",
            maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")).require_complete()
    benchmark = rt.returns(bm, method="simple", basis="total_return").values.filter(
        pl.col("period_start").is_not_null()).select("period_start", "session", "simple_return")
    report = rt.performance(run, benchmark=benchmark, benchmark_metadata={"source": bm.metadata["source"],
        "basis": "total_return_index", "currency": "USD", "frequency": "1d"},
        periods_per_year=252, risk_free_annual_effective=.03, minimum_acceptable_return_annual_effective=0.)
    assert (dates[-1]-dates[0]).days == 366
    assert run.daily.height == len(dates)-1 == 262
    assert report.metadata["n_obs"] == 262
    assert run.trades.height == (5 if leverage else 0)
    assert run.positions["asset"].n_unique() == 5
    assert run.daily["pnl"].sum() == pytest.approx(run.daily["equity"][-1]-100_000_000., abs=.001)
    # Separate reconstruction using returned balance-sheet/position records.
    for row in run.valuations.iter_rows(named=True):
        if row["phase"] == "pre_entry": continue
        marks = run.positions.filter(pl.col("session") == row["session"])
        risky = math.fsum(q*p for q, p in marks.select("quantity", "raw_mark").iter_rows())
        assert row["equity"] == pytest.approx(risky+row["cash"]+row["dividend_receivable"]-row["debt"], abs=.001)
    for row in run.daily.iter_rows(named=True):
        contributions = run.attribution.filter(pl.col("session") == row["session"])["pnl"].sum()
        assert contributions == pytest.approx(row["pnl"], abs=.001)
    weights = report.allocation.group_by("session").agg(pl.col("weight").sum())["weight"]
    assert weights.to_list() == pytest.approx([1.]*len(dates))
    assert (run.diagnostics["residual"].abs() <= run.diagnostics["tolerance"]).all()
    # Dividend paid after the final mark stays a receivable, not a lost asset.
    if leverage:
        assert run.daily["dividend_receivable"][-1] > 0
        assert run.costs.filter(pl.col("trade_id").is_not_null())["date"].unique().to_list() == [dates[0]]
