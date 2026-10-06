from datetime import date
import math

import polars as pl
import pytest

import research_toolkit as rt
from test_metrics import metric

D1, D2, D3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
NEXT = {D1: (D2, D3), D2: (D3, date(2024, 1, 5))}  # Interval starts after the signal session.
META = {"signal_source": "hand-built scores", "return_source": "hand-built returns", "return_basis": "total_return"}
OPTIONS = dict(timing="after_signal_session", quantiles=2, grouping="equal_count", ties="asset_order", metadata=META)


def tables(rows, intervals=NEXT):
    """rows: (signal_date, asset, signal, forward_return); forward_return None means no outcome."""
    signals = pl.DataFrame([(d, a, s) for d, a, s, _ in rows if s is not None],
                           schema={"signal_date": pl.Date, "asset": pl.String, "signal": pl.Float64}, orient="row")
    outcomes = pl.DataFrame([(d, a, *intervals[d], r) for d, a, _, r in rows if r is not None],
                            schema={"signal_date": pl.Date, "asset": pl.String, "period_start": pl.Date,
                                    "period_end": pl.Date, "forward_return": pl.Float64}, orient="row")
    return signals, outcomes


def run(rows, intervals=NEXT, **kw):
    return rt.signal_diagnostics(*tables(rows, intervals), **{**OPTIONS, **kw})


def ic(result, day):
    return result.ic.filter(pl.col("signal_date") == day).row(0, named=True)


def test_positive_and_negative_rank_ic_and_daily_mean():
    rows = [(D1, a, s, r) for a, s, r in zip("ABCD", [1., 2., 3., 4.], [.01, .02, .03, .04])]
    rows += [(D2, a, s, r) for a, s, r in zip("ABCD", [1., 2., 3., 4.], [.04, .03, .02, .01])]
    result = run(rows)
    assert ic(result, D1)["rank_ic"] == pytest.approx(1) and ic(result, D2)["rank_ic"] == pytest.approx(-1)
    assert metric(result.summary, "mean_rank_ic")["value"] == pytest.approx(0)
    assert metric(result.summary, "positive_rank_ic_share")["value"] == .5
    assert metric(result.summary, "rank_ic_std")["value"] == pytest.approx(math.sqrt(2))
    assert metric(result.summary, "mean_rank_ic")["n_obs"] == 2  # Dates, not asset-date pairs.
    assert metric(result.summary, "asset_date_observations")["value"] == 8


def test_average_ranks_for_ties_by_hand():
    # Signal ranks 1.5, 1.5, 3, 4; return ranks 1, 2, 4, 3: Spearman 3.5/sqrt(4.5*5).
    rows = [(D1, a, s, r) for a, s, r in zip("ABCD", [1., 1., 2., 3.], [-.01, 0., .02, .01])]
    assert ic(run(rows), D1)["rank_ic"] == pytest.approx(3.5/math.sqrt(22.5))


def test_mean_ic_is_not_a_pooled_correlation():
    rows = [(D1, a, s, r) for a, s, r in zip("ABC", [1., 2., 3.], [.01, .02, .03])]
    rows += [(D2, a, s, r) for a, s, r in zip("ABC", [10., 20., 30.], [.03, .01, .02])]
    result = run(rows)
    per_date = [ic(result, D1)["rank_ic"], ic(result, D2)["rank_ic"]]
    assert per_date == pytest.approx([1., -.5])
    assert metric(result.summary, "mean_rank_ic")["value"] == pytest.approx(.25)


def test_constant_and_insufficient_cross_sections():
    rows = [(D1, a, 1., r) for a, r in zip("ABC", [.01, .02, .03])]
    rows += [(D2, a, s, .01) for a, s in zip("ABC", [1., 2., 3.])]
    result = run(rows)
    assert ic(result, D1)["rank_ic"] is None and ic(result, D1)["status"] == "constant_signal"
    assert ic(result, D2)["rank_ic"] is None and ic(result, D2)["status"] == "constant_outcome"
    assert metric(result.summary, "mean_rank_ic")["status"] == "no_evaluated_dates"
    assert result.spreads.filter(pl.col("signal_date") == D1)["status"].item() == "constant_signal"
    single = run([(D1, "A", 1., .01)])
    assert ic(single, D1)["status"] == "insufficient_assets"
    assert single.spreads["status"].item() == "insufficient_assets_for_quantiles"


def test_uneven_equal_count_and_extreme_groups():
    rows = [(D1, a, float(i), .01*i) for i, a in enumerate("ABCDEFG")]
    equal = run(rows, quantiles=3)
    assert equal.quantiles["n_assets"].to_list() == [3, 2, 2]
    extremes = run(rows, quantiles=3, grouping="extremes")
    # floor(7/3) = 2 lowest and 2 highest assets; the middle 3 form the interior group.
    assert extremes.quantiles["n_assets"].to_list() == [2, 3, 2]
    assert extremes.quantiles["mean_forward_return"].to_list() == pytest.approx([.005, .03, .055])
    assert extremes.spreads["high_minus_low"].item() == pytest.approx(.05)
    halves = run(rows[:5], grouping="extremes")
    assert halves.quantiles["n_assets"].to_list() == [2, 2] and halves.coverage["n_unassigned"].item() == 1
    assert "unassigned when Q=2" in halves.metadata["quantile_assignment"]


def test_group_tie_handling_is_explicit():
    rows = [(D1, a, s, r) for a, s, r in zip("ABCD", [1., 2., 2., 3.], [.01, .02, .03, .04])]
    ordered = run(rows)
    # B and C tie on the boundary; asset order puts B low and C high.
    assert ordered.quantiles["mean_forward_return"].to_list() == pytest.approx([.015, .035])
    rejected = run(rows, ties="reject_boundary_ties")
    assert set(rejected.quantiles["status"]) == {"tie_at_group_boundary"}
    assert rejected.spreads["high_minus_low"].item() is None
    # The rank IC still uses average ranks regardless of the grouping tie rule.
    assert ic(rejected, D1)["rank_ic"] == ic(ordered, D1)["rank_ic"]


def test_changing_universe_and_explained_exclusions():
    rows = [(D1, a, s, r) for a, s, r in zip("ABC", [1., 2., 3.], [.01, .02, .03])]
    rows += [(D2, "A", 1., .02), (D2, "B", 2., .01), (D2, "D", 3., .03), (D2, "E", 4., None)]
    signals, outcomes = tables(rows)
    with pytest.raises(ValueError, match="without an exclusion"):
        rt.signal_diagnostics(signals, outcomes, **OPTIONS)
    exclusions = pl.DataFrame({"signal_date": [D2], "asset": ["E"], "reason": ["delisted; no forward return"]})
    result = rt.signal_diagnostics(signals, outcomes, exclusions=exclusions, **OPTIONS)
    assert result.coverage.rows() == [(D1, 3, 3, 3, 0, 0), (D2, 4, 3, 3, 1, 0)]
    assert result.exclusions.rows() == [(D2, "E", "signal_without_outcome", "delisted; no forward return")]
    stray = pl.DataFrame({"signal_date": [D1], "asset": ["A"], "reason": ["not actually missing"]})
    with pytest.raises(ValueError, match="matched"):
        rt.signal_diagnostics(signals, outcomes, exclusions=pl.concat([exclusions, stray]), **OPTIONS)
    extra = pl.concat([outcomes, outcomes.slice(0, 1).with_columns(pl.lit("Z").alias("asset"))])
    with pytest.raises(ValueError, match="outcome_without_signal"):
        rt.signal_diagnostics(signals, extra, exclusions=exclusions, **OPTIONS)


def test_forward_return_timing():
    rows = [(D1, a, s, r) for a, s, r in zip("AB", [1., 2.], [.01, .02])]
    same = {D1: (D1, D2)}
    with pytest.raises(ValueError, match="same_session_close_assumed"):
        run(rows, same)
    assert run(rows, same, timing="same_session_close_assumed").ic["status"].item() == "ok"
    with pytest.raises(ValueError, match="backward-looking"):
        run(rows, {D1: (date(2024, 1, 1), D2)}, timing="same_session_close_assumed")
    with pytest.raises(ValueError, match="after period_start"):
        run(rows, {D1: (D2, D2)})
    # A later execution start is allowed; the skipped return is simply not part of the outcome.
    later = run(rows, {D1: (D3, date(2024, 1, 5))})
    assert later.ic.select("period_start", "period_end").row(0) == (D3, date(2024, 1, 5))
    signals, outcomes = tables(rows)
    mixed = outcomes.with_columns(pl.Series("period_end", [D3, date(2024, 1, 5)]))
    with pytest.raises(ValueError, match="share one forward interval"):
        rt.signal_diagnostics(signals, mixed, **OPTIONS)


def test_overlapping_horizons_and_validation():
    rows = [(d, a, s, r) for d in (D1, D2) for a, s, r in zip("AB", [1., 2.], [.01, .02])]
    weekly = run(rows, {D1: (D2, date(2024, 1, 9)), D2: (D3, date(2024, 1, 10))})
    assert weekly.metadata["overlapping_forward_intervals"] is True
    assert "serially dependent" in weekly.metadata["independence"]
    assert run(rows).metadata["overlapping_forward_intervals"] is False
    for bad in (dict(quantiles=1), dict(grouping="ranked"), dict(ties="random"), dict(weighting="signal"),
                dict(timing="next_day"), dict(metadata={"signal_source": "x"})):
        with pytest.raises(ValueError):
            run(rows, **bad)
    signals, outcomes = tables(rows)
    with pytest.raises(ValueError, match="duplicate"):
        rt.signal_diagnostics(pl.concat([signals, signals.slice(0, 1)]), outcomes, **OPTIONS)
    with pytest.raises(ValueError, match="finite"):
        rt.signal_diagnostics(signals.with_columns(pl.lit(float("inf")).alias("signal")), outcomes, **OPTIONS)


def test_synthetic_example_runs_offline():
    import runpy
    from pathlib import Path
    example = runpy.run_path(str(Path(__file__).resolve().parents[1]/"examples"/"diagnostics.py"))
    report, calendar, trading, joint, market_only, signal = example["run_example"]()
    assert set(calendar.values["status"]) == {"ok"} and joint.metadata["status"] == "ok"
    assert joint.metadata["transformations"] == {"dependent": "minus_risk_free", "MKT": "minus_risk_free", "SMB": "none"}
    assert market_only.standalone["beta"].item() == pytest.approx(
        market_only.coefficients.filter(pl.col("term") == "MKT")["estimate"].item())
    assert set(signal.quantiles["n_assets"].unique()) == {4, 5}  # floor(23/5) at each extreme, 15 in the middle.
