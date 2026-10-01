"""Offline SOFR timing, rate validation, and independent interest oracles."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit._sofr import RATE_SCHEMA, PUBLICATION_SCHEMA
from test_rebalancing import reconcile


def sofr_config(*, rows=None, **overrides):
    # Synthetic observation/publication calendar: Jan 4 prints Jan 5; Jan 5 prints Jan 8.
    rows = rows if rows is not None else [(date(2024, 1, 4), datetime(2024, 1, 5, 13, tzinfo=timezone.utc), .036),
        (date(2024, 1, 5), datetime(2024, 1, 8, 13, tzinfo=timezone.utc), .072),
        (date(2024, 1, 8), datetime(2024, 1, 9, 13, tzinfo=timezone.utc), .108)]
    kwargs = dict(rates=pl.DataFrame([(d, r) for d, _, r in rows], schema=RATE_SCHEMA, orient="row"),
        publication_calendar=pl.DataFrame([(d, t) for d, t, _ in rows], schema=PUBLICATION_SCHEMA, orient="row"),
        metadata=dict(source="synthetic SOFR test fixture", calendar="supplied synthetic publication calendar",
            calendar_version="1", currency="USD", rate_units="decimal", timezone="America/New_York",
            vintage="point_in_time", calendar_complete=True, coverage_start="2024-01-05", coverage_end="2024-01-10",
            retrieved_at="2024-02-01T00:00:00Z"), borrowing_spread_bps=0., cash_rate=0., day_count="ACT/360",
        cash_day_count="ACT/365F", rate_timing="known_at_accrual_start", max_rate_age_days=7,
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")
    return rt.SOFRFinancing(**(kwargs | overrides))


def simulate(inputs, run, policy, *, finance=None, values=None, dates=None, leverage=2., **kwargs):
    dates = dates if dates is not None else [date(2024, 1, d) for d in (5, 8, 9, 10)]
    values = values if values is not None else [100.]*len(dates)
    market = rt.prepare_market_data(**inputs({"A": values}, dates=dates))
    return run(market, policy=policy(leverage), financing=finance or sofr_config(),
               cash_rate=None, cash_day_count=None, **kwargs)


def check_audit(result):
    reconcile(result)
    for row in result.financing_accruals.iter_rows(named=True):
        assert row["available_at"] <= row["cutoff_at"]
        assert row["borrowing_rate"] == pytest.approx(row["sofr"]+row["borrowing_spread_bps"]/10_000)
        assert row["borrowing_interest"] == pytest.approx(row["opening_debt"]*row["borrowing_rate"]*row["borrowing_day_fraction"])
        assert row["cash_interest"] == pytest.approx(row["opening_cash"]*row["cash_rate"]*row["cash_day_fraction"])
        costs = result.costs.filter((pl.col("date") == row["date"]) & (pl.col("component") == "borrowing_interest"))
        assert costs["amount"].sum() == pytest.approx(row["borrowing_interest"])
        events = result.events.filter((pl.col("date") == row["date"]) & (pl.col("type") == "cash_interest"))
        assert events["cash_delta"].sum() == pytest.approx(row["cash_interest"])
    json.dumps(result.metadata, allow_nan=False)


def test_calendar_days_publication_lag_spread_and_compounding(inputs, run, policy):
    result = simulate(inputs, run, policy, finance=sofr_config(borrowing_spread_bps=36.))
    audit = result.financing_accruals
    assert audit["date"].to_list() == [date(2024, 1, d) for d in range(6, 11)]
    assert audit["sofr"].to_list() == [.036, .036, .036, .072, .108]
    assert audit["observation_date"].to_list() == [date(2024, 1, d) for d in (4, 4, 4, 5, 8)]
    # 36bps adds .0036, so daily factors are 1.00011, 1.00021, 1.00031.
    debt = Decimal(100)*Decimal('1.00011')**3*Decimal('1.00021')*Decimal('1.00031')
    assert result.daily["debt"][-1] == pytest.approx(float(debt))
    assert result.daily["equity"][-1] == pytest.approx(200-float(debt))
    assert result.daily.height == 3 and result.trades.height == 1
    assert result.metadata["borrowing_rate"] is None  # Do not mislabel a variable curve as one rate.
    assert result.metadata["financing"]["model"] == "historical_sofr_plus_spread"
    check_audit(result)


def test_future_publication_cannot_change_earlier_financing(inputs, run, policy):
    base = sofr_config()
    changed = replace(base, rates=base.rates.with_columns(pl.when(pl.col("observation_date") >= date(2024, 1, 5))
        .then(pl.lit(.5)).otherwise(pl.col("sofr")).alias("sofr")))
    a = simulate(inputs, run, policy, finance=base)
    b = simulate(inputs, run, policy, finance=changed)
    assert a.daily.head(1).equals(b.daily.head(1))
    assert a.financing_accruals.head(3).equals(b.financing_accruals.head(3))
    assert a.daily["debt"][-1] != b.daily["debt"][-1]


def test_act365_constant_curve_matches_fixed_financing(inputs, run, policy):
    config = sofr_config()
    config = replace(config, rates=config.rates.with_columns(pl.lit(.05).alias("sofr")),
                     borrowing_spread_bps=100., day_count="ACT/365F")
    fixed = rt.Financing(cash_rate=0., borrowing_rate=.05+100/10_000, day_count="ACT/365F",
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")
    a = simulate(inputs, run, policy, finance=config)
    b = simulate(inputs, run, policy, finance=fixed)
    for name in ("daily", "events", "trades", "positions", "costs", "valuations", "attribution"):
        assert getattr(a, name).equals(getattr(b, name))
    assert b.financing_accruals.is_empty()


def test_cash_day_count_is_separate_and_zero_debt_is_audited(inputs, run, policy):
    result = simulate(inputs, run, policy, finance=sofr_config(cash_rate=.36, cash_day_count="ACT/360"), leverage=0.)
    assert result.daily["equity"][-1] == pytest.approx(100*1.001**5)
    assert result.financing_accruals["borrowing_interest"].sum() == 0.
    assert result.financing_accruals.height == 5 and result.costs.is_empty()
    check_audit(result)


def test_holiday_carry_and_dst_midnight(inputs, run, policy):
    rows = [(date(2024, 3, 7), datetime(2024, 3, 8, 13, tzinfo=timezone.utc), .036),
            (date(2024, 3, 8), datetime(2024, 3, 11, 12, tzinfo=timezone.utc), .072)]
    base = sofr_config()
    meta = dict(base.metadata, coverage_start="2024-03-08", coverage_end="2024-03-12", retrieved_at="2024-04-01T00:00:00Z")
    result = simulate(inputs, run, policy, dates=[date(2024, 3, d) for d in (8, 11, 12)],
        finance=sofr_config(rows=rows, metadata=meta))
    assert result.financing_accruals["cutoff_at"].dt.hour().to_list() == [5, 5, 4, 4]
    assert result.financing_accruals["sofr"].to_list() == [.036, .036, .036, .072]
    assert result.financing_accruals["borrowing_day_fraction"].to_list() == [1/360]*4
    check_audit(result)


def test_explicit_long_weekend_calendar(inputs, run, policy):
    # Synthetic calendar explicitly has no Jan 8 publication (do not infer holidays).
    rows = [(date(2024, 1, 4), datetime(2024, 1, 5, 13, tzinfo=timezone.utc), .036),
            (date(2024, 1, 5), datetime(2024, 1, 9, 13, tzinfo=timezone.utc), .072)]
    result = simulate(inputs, run, policy, finance=sofr_config(rows=rows))
    assert result.financing_accruals["sofr"].to_list() == [.036]*4+[.072]
    check_audit(result)


def test_rate_boundary_includes_exact_cutoff(inputs, run, policy):
    # Artificial delayed publication at precisely New York midnight.
    rows = [(date(2024, 1, 4), datetime(2024, 1, 6, 5, tzinfo=timezone.utc), .036)]
    config = sofr_config(rows=rows)
    result = simulate(inputs, run, policy, finance=config)
    assert result.financing_accruals["available_at"][0] == result.financing_accruals["cutoff_at"][0]


@pytest.mark.parametrize("change,match", [
    (dict(max_rate_age_days=1), "max_rate_age_days"),
    (dict(metadata=dict(sofr_config().metadata, coverage_end="2024-01-09")), "coverage"),
])
def test_insufficient_rate_coverage_rejected(inputs, run, policy, change, match):
    with pytest.raises(ValueError, match=match):
        simulate(inputs, run, policy, finance=sofr_config(**change))


def test_no_seed_rate_rejected(inputs, run, policy):
    base = sofr_config()
    config = replace(base, rates=base.rates.tail(2), publication_calendar=base.publication_calendar.tail(2))
    with pytest.raises(ValueError, match="no SOFR rate available"):
        simulate(inputs, run, policy, finance=config)


def test_missing_publication_not_silently_carried():
    base = sofr_config()
    with pytest.raises(ValueError, match="every expected"):
        replace(base, rates=base.rates.filter(pl.col("observation_date") != date(2024, 1, 5)))


@pytest.mark.parametrize("field,value,match", [
    ("day_count", "ACT/366", "day_count"), ("cash_day_count", "infer", "cash_day_count"),
    ("rate_timing", "same_observation_date", "rate_timing"),
    ("max_rate_age_days", True, "max_rate_age_days"),
    ("borrowing_spread_bps", -1., "borrowing_spread_bps"),
    ("borrowing_spread_bps", float('nan'), "borrowing_spread_bps"),
])
def test_invalid_policies(field, value, match):
    with pytest.raises(ValueError, match=match):
        sofr_config(**{field: value})


@pytest.mark.parametrize("bad", [-.01, float('nan'), float('inf'), None])
def test_invalid_rates_rejected(bad):
    base = sofr_config()
    with pytest.raises(ValueError):
        replace(base, rates=base.rates.with_columns(pl.lit(bad, dtype=pl.Float64).alias("sofr")))


def test_duplicate_wrong_schema_and_early_publication():
    base = sofr_config()
    with pytest.raises(ValueError, match="duplicate"):
        replace(base, rates=pl.concat([base.rates, base.rates.head(1)]))
    with pytest.raises(ValueError, match="schema"):
        replace(base, rates=base.rates.with_columns(pl.col("sofr").cast(pl.Float32)))
    with pytest.raises(ValueError, match="follow its observation"):
        replace(base, publication_calendar=base.publication_calendar.with_columns(
            pl.col("available_at")-pl.duration(days=5)))


@pytest.mark.parametrize("key,value", [("vintage", "latest_revised"), ("rate_units", "percent"),
    ("calendar_complete", False), ("timezone", "UTC"), ("currency", "EUR"),
    ("retrieved_at", "2024-01-01T00:00:00Z")])
def test_source_metadata_rejected(key, value):
    with pytest.raises(ValueError):
        sofr_config(metadata=dict(sofr_config().metadata, **{key: value}))


def test_owned_copies_and_mutation_detection(inputs, run, policy):
    config = sofr_config()
    owned = config.rates.clone()
    another = replace(config)
    assert another.snapshot_id == config.snapshot_id
    config.metadata["source"] = "changed"
    with pytest.raises(ValueError, match="changed after construction"):
        simulate(inputs, run, policy, finance=config)
    assert another.rates.equals(owned)
    assert another.metadata["source"] == "synthetic SOFR test fixture"


def test_usd_and_mixed_cash_arguments(inputs, run, policy):
    dates = [date(2024, 1, d) for d in (5, 8)]
    market = rt.prepare_market_data(**inputs({"A": [100., 100.]}, dates=dates,
        currency="EUR", asset_currencies={"A": "EUR"}))
    with pytest.raises(ValueError, match="USD"):
        run(market, financing=sofr_config(), policy=policy(2.), cash_rate=None, cash_day_count=None)
    market = rt.prepare_market_data(**inputs({"A": [100., 100.]}, dates=dates))
    with pytest.raises(ValueError, match="do not mix"):
        run(market, financing=sofr_config())


def test_stopped_run_has_no_later_accruals(inputs, run, policy):
    result = simulate(inputs, run, policy, values=[100., 60., 100., 110.])
    assert result.status == "stopped" and result.stop_session == date(2024, 1, 8)
    assert result.financing_accruals["date"].max() == result.stop_session
    assert result.financing_accruals.height == 3
    check_audit(result)
    with pytest.raises(ValueError, match="stopped"):
        rt.performance(result, periods_per_year=252, risk_free_annual_effective=0., minimum_acceptable_return_annual_effective=0.)
    rt.performance(result, periods_per_year=252, risk_free_annual_effective=0., minimum_acceptable_return_annual_effective=0., allow_partial=True)


@pytest.mark.parametrize("priority", ["before_debt_repayment", "after_debt_repayment"])
def test_dividend_payment_and_reinvestment_use_same_financing_ledger(inputs, run, policy, priority):
    dates = [date(2024, 1, d) for d in (5, 8, 9, 10)]
    market = rt.prepare_market_data(**inputs({"A": [100.]*4}, dates=dates,
        dividends=[("d", "A", dates[1], dates[2], 10.)]))
    result = run(market, policy=policy(2.), financing=sofr_config(), cash_rate=None, cash_day_count=None,
        dividend_reinvestment=rt.DividendReinvestment(execution="first_close_on_or_after_payment",
            funding=priority, scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    before_payment = Decimal(100)*Decimal('1.0001')**3*Decimal('1.0002')
    debt = (before_payment - (Decimal(20) if priority == "after_debt_repayment" else 0))*Decimal('1.0003')
    assert result.daily["debt"][-1] == pytest.approx(float(debt))
    assert result.positions["quantity"][-1] == pytest.approx(2. if priority == "after_debt_repayment" else 2.2)
    check_audit(result)


def test_scheduled_deleveraging_stops_interest_after_execution(inputs):
    from research_toolkit._rebalancing import TARGET_SCHEMA
    dates = [date(2024, 1, d) for d in (4, 5, 8, 9, 10)]
    market = rt.prepare_market_data(**inputs({"A": [100.]*5}, dates=dates))
    targets = pl.DataFrame([(dates[0], dates[1], "A", 1., 2.), (dates[1], dates[2], "A", 1., 0.)],
        schema=TARGET_SCHEMA, orient="row")
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=100.,
        entry_session=dates[1], end_session=dates[-1], financing=sofr_config(),
        costs=rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.),
        policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity",
            fractional_shares=True, terminal_action="mark_only", non_session="raise",
            receivable_policy="reserve", max_asset_weight=1.))
    assert result.financing_accruals["borrowing_interest"].tail(2).to_list() == [0., 0.]
    assert result.daily["debt"].to_list() == [0., 0., 0.]
    assert result.daily["equity"][-1] == pytest.approx(200-100*1.0001**3)
    check_audit(result)


def test_saved_metadata_reconstructs_same_rate_identity(inputs, run, policy):
    result = simulate(inputs, run, policy)
    payload = json.loads(json.dumps(result.metadata["financing"]))
    reconstructed = sofr_config(
        rates=pl.DataFrame([(date.fromisoformat(r["observation_date"]), r["sofr"]) for r in payload["rates"]],
            schema=RATE_SCHEMA, orient="row"),
        publication_calendar=pl.DataFrame([(date.fromisoformat(r["observation_date"]), datetime.fromisoformat(r["available_at"]))
            for r in payload["publication_calendar"]], schema=PUBLICATION_SCHEMA, orient="row"), metadata=payload["metadata"])
    assert reconstructed.snapshot_id == payload["snapshot_id"]
    rerun = simulate(inputs, run, policy, finance=reconstructed)
    assert result.financing_accruals.equals(rerun.financing_accruals)
    assert result.daily.equals(rerun.daily)
