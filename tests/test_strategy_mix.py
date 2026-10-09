"""Two-strategy information-ratio allocation and analytical combination, checked by hand."""
from datetime import date, timedelta
import math
import random
from statistics import mean, stdev

import polars as pl
import pytest

import research_toolkit as rt

A = 252
META = {"source": "hand-built strategy", "currency": "USD", "frequency": "1d",
        "return_basis": "net_equity", "return_method": "simple"}
BMETA = {"source": "hand-built benchmark", "basis": "total_return", "currency": "USD", "frequency": "1d"}
# Zero-sum and mutually orthogonal: sample correlation is exactly zero and each
# pattern's ddof=1 variance is 4/3 of its squared scale.
X = [1., -1., 1., -1.]
Y = [1., 1., -1., -1.]


def days(n, start=date(2024, 1, 1)):
    return [start + timedelta(days=i) for i in range(n + 1)]


def table(values, start=date(2024, 1, 1)):
    d = days(len(values), start)
    return pl.DataFrame({"period_start": d[:-1], "session": d[1:], "simple_return": values},
                        schema={"period_start": pl.Date, "session": pl.Date, "simple_return": pl.Float64})


def window(values, start=date(2024, 1, 1)):
    d = days(len(values), start)
    return d[0], d[-1]


def allocate(a, b, *, bounds=((0., 1.), (0., 1.)), covariance="estimated", benchmark="zero", **kw):
    strategies = {"A": table(a), "B": table(b)}
    return rt.information_ratio_weights(
        strategies, strategy_metadata={"A": META, "B": META}, benchmark=benchmark, periods_per_year=A,
        bounds={"A": bounds[0], "B": bounds[1]}, estimation_window=kw.pop("estimation_window", window(a)),
        covariance=covariance, **kw)


def weight(result, name="A"):
    return result.weights.filter(pl.col("strategy") == name)["weight"].item()


def summary(result, basis, name="information_ratio"):
    return result.summary.filter((pl.col("basis") == basis) & (pl.col("metric") == name)).row(0, named=True)


def ir(w, a, b):
    """Independent oracle: annualized IR of the observed combined series."""
    c = [w*x + (1 - w)*y for x, y in zip(a, b)]
    return math.sqrt(A)*mean(c)/stdev(c)


def grid_argmax(a, b, lo=0., hi=1., steps=2000):
    """Brute-force grid plus golden-section reference, independent of the closed-form candidates.

    IR(w) has at most one interior stationary point, so the best grid cell brackets the maximum.
    """
    best = max((lo + (hi - lo)*i/steps for i in range(steps + 1)), key=lambda w: ir(w, a, b))
    left, right = max(lo, best - (hi - lo)/steps), min(hi, best + (hi - lo)/steps)
    for _ in range(200):  # golden-section refinement of the bracketing cell
        m1, m2 = left + (right - left)*.382, left + (right - left)*.618
        left, right = (m1, right) if ir(m1, a, b) < ir(m2, a, b) else (left, m2)
    return (left + right)/2


def test_zero_correlation_positive_means_match_the_analytical_interior_solution():
    a = [.002 + .01*x for x in X]
    b = [.001 + .005*y for y in Y]
    result = allocate(a, b, covariance="zero_correlation")
    # w_i proportional to mean_i/var_i: (.002/(1e-4*4/3), .001/(2.5e-5*4/3)) = (15, 30).
    assert weight(result) == pytest.approx(1/3) and weight(result, "B") == pytest.approx(2/3)
    assert result.candidates.filter(pl.col("selected"))["kind"].item() == "stationary"
    expected = math.sqrt(A)*math.sqrt(.002**2/(1e-4*4/3) + .001**2/(2.5e-5*4/3))
    assert summary(result, "optimization_covariance")["value"] == pytest.approx(expected)
    # The sample correlation really is zero, so the estimated mode agrees exactly.
    estimated = allocate(a, b, covariance="estimated")
    assert weight(estimated) == pytest.approx(1/3)
    assert summary(estimated, "observed_combined_returns")["value"] == pytest.approx(expected)
    vols = dict(result.estimates.select("strategy", "annualized_tracking_error").iter_rows())
    assert vols == pytest.approx({"A": .01*math.sqrt(4/3*A), "B": .005*math.sqrt(4/3*A)})
    assert result.metadata["optimum"] == "unique" and result.metadata["status"] == "ok"
    assert not any(result.weights["at_lower_bound"]) and not any(result.weights["at_upper_bound"])


def test_correlated_case_matches_an_independent_numerical_reference():
    rng = random.Random(4)
    common = [rng.gauss(0, .01) for _ in range(120)]
    a = [.002 + c + rng.gauss(0, .006) for c in common]
    b = [.001 + .5*c + rng.gauss(0, .004) for c in common]
    result = allocate(a, b)
    reference = grid_argmax(a, b)
    assert .2 < reference < .8 and mean(a) > 0 and mean(b) > 0
    assert weight(result) == pytest.approx(reference, abs=1e-6)
    assert summary(result, "optimization_covariance")["value"] == pytest.approx(ir(reference, a, b), rel=1e-9)
    # Tangency weights Sigma^-1 mu, normalized, coincide here because they are feasible.
    ma, mb = mean(a), mean(b)
    da, db = [v - ma for v in a], [v - mb for v in b]
    saa = sum(v*v for v in da)/119
    sbb = sum(v*v for v in db)/119
    sab = sum(x*y for x, y in zip(da, db))/119
    ta, tb = sbb*ma - sab*mb, saa*mb - sab*ma
    assert weight(result) == pytest.approx(ta/(ta + tb), rel=1e-9)
    corr = result.covariance.filter((pl.col("strategy") == "A") & (pl.col("other_strategy") == "B"))
    assert corr["empirical_correlation"].item() == pytest.approx(sab/math.sqrt(saa*sbb))


def test_binding_bounds_and_infeasible_constraints():
    a = [.002 + .01*x for x in X]
    b = [.001 + .005*y for y in Y]
    result = allocate(a, b, bounds=((.5, 1.), (0., 1.)), covariance="zero_correlation")
    assert weight(result) == .5 and weight(result, "B") == .5
    row = result.weights.filter(pl.col("strategy") == "A").row(0, named=True)
    assert row["at_lower_bound"] and not row["at_upper_bound"]
    # The bound implied by the other strategy is binding for that strategy.
    implied = allocate(a, b, bounds=((0., 1.), (0., .25)), covariance="zero_correlation")
    assert weight(implied, "B") == .25 and weight(implied) == .75
    assert implied.weights.filter(pl.col("strategy") == "B")["at_upper_bound"].item()
    fixed = allocate(a, b, bounds=((.3, .3), (.7, .7)))
    assert (weight(fixed), weight(fixed, "B")) == (.3, .7)
    with pytest.raises(ValueError, match="infeasible bounds"):
        allocate(a, b, bounds=((.6, 1.), (.5, 1.)))
    for bad in [((-.1, 1.), (0., 1.)), ((0., 1.1), (0., 1.)), ((.6, .5), (0., 1.)), ((0., math.nan), (0., 1.))]:
        with pytest.raises(ValueError):
            allocate(a, b, bounds=bad)
    with pytest.raises(ValueError, match="exactly the strategies"):
        rt.information_ratio_weights({"A": table(a), "B": table(b)}, strategy_metadata={"A": META, "B": META},
                                     benchmark="zero", periods_per_year=A, bounds={"A": (0, 1)},
                                     estimation_window=window(a), covariance="estimated")


def test_mixed_means_reject_naive_normalized_weights():
    a = [.002 + .01*x for x in X]
    b = [-.001 + .005*y for y in Y]
    # mean/var gives (15, -30): normalizing would give (-1, 2), a short and levered pair.
    result = allocate(a, b, bounds=((.1, .9), (.1, .9)), covariance="zero_correlation")
    assert weight(result) == .9 and weight(result, "B") == pytest.approx(.1)
    assert result.weights.filter(pl.col("strategy") == "A")["at_upper_bound"].item()


def test_mixed_means_with_a_negatively_correlated_hedge_have_an_interior_optimum():
    rng = random.Random(11)
    common = [rng.gauss(0, .01) for _ in range(150)]
    a = [.001 + c + rng.gauss(0, .003) for c in common]
    b = [-.0002 - .8*c + rng.gauss(0, .003) for c in common]
    result = allocate(a, b)
    reference = grid_argmax(a, b)
    assert .5 < weight(result) < 1 and weight(result) == pytest.approx(reference, abs=1e-6)
    assert result.candidates.filter(pl.col("selected"))["kind"].item() == "stationary"


def test_all_negative_means_choose_the_largest_not_the_largest_absolute_ir():
    a = [-.001 + .01*x for x in X]
    b = [-.001 + .005*y for y in Y]
    result = allocate(a, b, covariance="zero_correlation")
    stationary = result.candidates.filter(pl.col("kind") == "stationary").row(0, named=True)
    # The stationary point (w=0.2) is the most negative IR, which |IR| or IR^2 would select.
    assert stationary["first_strategy_weight"] == pytest.approx(.2) and not stationary["selected"]
    assert stationary["information_ratio"] < summary(result, "optimization_covariance")["value"] < 0
    assert weight(result) == 1. and result.candidates.filter(pl.col("selected"))["kind"].item() == "upper_endpoint"
    assert summary(result, "optimization_covariance")["value"] == pytest.approx(math.sqrt(A)*-.001/(.01*math.sqrt(4/3)))


def test_ties_are_deterministic_and_flat_objectives_are_identified():
    a = [-.002 + .01*x for x in X]
    b = [-.001 + .005*y for y in Y]
    # Both endpoints have IR = sqrt(A) * -0.1732...: smallest weight on the first name wins.
    result = allocate(a, b, covariance="zero_correlation")
    assert result.metadata["optimum"] == "multiple_optima" and weight(result) == 0.
    reversed_order = rt.information_ratio_weights(
        {"B": table(b), "A": table(a)}, strategy_metadata={"B": META, "A": META}, benchmark="zero",
        periods_per_year=A, bounds={"B": (0., 1.), "A": (0., 1.)}, estimation_window=window(a),
        covariance="zero_correlation")
    assert weight(reversed_order) == 0. and reversed_order.weights["strategy"].to_list() == ["A", "B"]
    # Zero mean active returns: IR = 0 at every feasible weight, a valid finite flat objective.
    flat = allocate([.01*x for x in X], [.005*y for y in Y], bounds=((.2, .8), (.2, .8)))
    assert flat.metadata["optimum"] == "flat_objective" and weight(flat) == .2
    assert summary(flat, "optimization_covariance")["value"] == 0.


def test_constant_returns_zero_tracking_error_and_unbounded_ir():
    both_positive = allocate([.002]*4, [.001]*4)
    assert both_positive.metadata["status"] == "unbounded_information_ratio"
    assert both_positive.weights["weight"].to_list() == [None, None]
    assert summary(both_positive, "optimization_covariance")["value"] is None
    zero_active = allocate([.001]*4, [.003]*4, benchmark=table([.001]*4)
                           .with_columns(pl.Series("simple_return", [.001, .001, .001, .001])),
                           benchmark_metadata=BMETA, bounds=((0., 1.), (0., 0.)))
    assert zero_active.metadata["status"] == "undefined_zero_tracking_error"
    with pytest.raises(ValueError, match="no weights"):
        rt.combine_strategies({"A": table([.002]*4), "B": table([.001]*4)}, strategy_metadata={"A": META, "B": META},
                              weights=both_positive, evaluation_window=window([0]*4))
    # A riskless positive strategy: IR grows without bound as its weight reaches 1 ...
    b = [.001 + .005*y for y in Y]
    assert allocate([.0005]*4, b).metadata["status"] == "unbounded_information_ratio"
    # ... but a cap keeps tracking error positive and the optimum finite at the cap.
    capped = allocate([.0005]*4, b, bounds=((0., .9), (0., 1.)))
    assert weight(capped) == .9 and capped.metadata["status"] == "ok"
    assert summary(capped, "optimization_covariance")["value"] == pytest.approx(
        math.sqrt(A)*(.9*.0005 + .1*.001)/(.1*.005*math.sqrt(4/3)))
    assert capped.estimates.filter(pl.col("strategy") == "A")["information_ratio_status"].item() == "zero_tracking_error"


def test_perfect_and_near_perfect_correlation():
    # Perfect negative correlation: w = 0.5 is riskless with mean 0.0015 > 0, so IR is unbounded.
    a = [.001 + .01*x for x in X]
    b = [.002 - .01*x for x in X]
    hedged = allocate(a, b)
    assert hedged.metadata["status"] == "unbounded_information_ratio"
    assert hedged.metadata["covariance_status"] == "singular"
    kinds = dict(hedged.candidates.select("kind", "status").iter_rows())
    assert kinds["minimum_tracking_error"] == "zero_tracking_error_positive_mean"
    # Excluding the hedge point by bounds leaves a finite optimum approaching it.
    bounded = allocate(a, b, bounds=((0., .4), (0., 1.)))
    assert bounded.metadata["status"] == "ok" and weight(bounded) == .4
    # A riskless combination with a negative mean is merely excluded, not chosen.
    negative = allocate([-.001 + .01*x for x in X], [-.002 - .01*x for x in X])
    assert negative.metadata["status"] == "ok" and weight(negative) in (0., 1.)
    # Perfect positive correlation with equal standalone IR: every weight is optimal.
    equal = allocate(a, [2*v for v in a], bounds=((.1, .9), (.1, .9)))
    assert equal.metadata["optimum"] == "flat_objective" and weight(equal) == .1
    # Nearly singular is a valid finite estimate, flagged and checked numerically.
    rng = random.Random(5)
    base = [rng.gauss(.0005, .01) for _ in range(80)]
    near = [.0002 - v + rng.gauss(0, 2e-7) for v in base]
    result = allocate(base, near)
    assert result.metadata["covariance_status"] == "near_singular" and result.metadata["status"] == "ok"
    reference = grid_argmax(base, near)
    assert weight(result) == pytest.approx(reference, abs=1e-6)
    assert math.isfinite(summary(result, "observed_combined_returns")["value"])


def test_active_returns_use_the_varying_benchmark():
    rng = random.Random(9)
    bench = [rng.gauss(.0005, .02) for _ in range(4)]
    a = [m + .002 + .01*x for m, x in zip(bench, X)]
    b = [m + .001 + .005*y for m, y in zip(bench, Y)]
    benchmark = table(bench)
    result = allocate(a, b, benchmark=benchmark, benchmark_metadata=BMETA)
    # Active returns are the orthogonal design, so the interior solution is 1/3 again.
    assert weight(result) == pytest.approx(1/3)
    corr = result.covariance.filter((pl.col("strategy") == "A") & (pl.col("other_strategy") == "B")).row(0, named=True)
    assert corr["empirical_correlation"] == pytest.approx(0, abs=1e-12)
    assert corr["raw_return_correlation"] > .5  # raw returns share the benchmark
    assert result.returns.filter(pl.col("strategy") == "A")["active_return"].to_list() == pytest.approx(
        [.002 + .01*x for x in X])
    assert result.metadata["benchmark"]["metadata"] == BMETA
    assert weight(allocate(a, b)) != pytest.approx(1/3, abs=1e-3)


def test_strict_alignment_on_both_endpoints():
    a = [.002 + .01*x for x in X]
    b = [.001 + .005*y for y in Y]
    w = window(a)

    def run(sa, sb, **kw):
        return rt.information_ratio_weights({"A": sa, "B": sb}, strategy_metadata=kw.pop("meta", {"A": META, "B": META}),
            benchmark=kw.pop("benchmark", "zero"), periods_per_year=A, bounds={"A": (0, 1), "B": (0, 1)},
            estimation_window=kw.pop("estimation_window", w), covariance="estimated", **kw)

    shifted = table(b).with_columns(pl.Series("period_start", [date(2023, 12, 31)] + days(4)[1:4]))
    with pytest.raises(ValueError):
        run(table(a), shifted)
    gap = pl.concat([table(b)[:1], table(b)[2:]])
    with pytest.raises(ValueError, match="contiguous"):
        run(table(a), gap)
    with pytest.raises(ValueError, match="duplicate"):
        run(table(a), pl.concat([table(b), table(b)[:1]]))
    with pytest.raises(ValueError, match="null"):
        run(table(a), table(b).with_columns(pl.when(pl.col("simple_return") > 0).then(None)
                                           .otherwise(pl.col("simple_return")).alias("simple_return")))
    with pytest.raises(ValueError, match="finite"):
        run(table(a), table([math.inf] + b[1:]))
    with pytest.raises(ValueError, match="observed interval endpoints"):
        run(table(a), table(b), estimation_window=(w[0], w[1] + timedelta(days=1)))
    with pytest.raises(ValueError, match="observed interval endpoints"):
        run(table(a), table(b, start=date(2024, 1, 2)))
    with pytest.raises(ValueError, match="crossing"):
        run(table(a), pl.concat([table(b)[:3], pl.DataFrame({"period_start": [date(2024, 1, 4)],
            "session": [date(2024, 1, 6)], "simple_return": [.0]})]), estimation_window=(w[0], date(2024, 1, 5)))
    # Same window, but two-day intervals: both series are contiguous yet do not match.
    coarse = pl.DataFrame({"period_start": [date(2024, 1, 1), date(2024, 1, 3)],
                           "session": [date(2024, 1, 3), date(2024, 1, 5)], "simple_return": [.01, .02]})
    with pytest.raises(ValueError, match="both endpoints exactly"):
        run(table(a), coarse)
    with pytest.raises(ValueError, match="both endpoints exactly"):
        run(table(a), table(b), benchmark=coarse, benchmark_metadata=BMETA)
    with pytest.raises(ValueError, match="currency"):
        run(table(a), table(b), meta={"A": META, "B": {**META, "currency": "EUR"}})
    with pytest.raises(ValueError, match="frequency"):
        run(table(a), table(b), meta={"A": META, "B": {**META, "frequency": "multi_session"}})
    with pytest.raises(ValueError, match="simple returns"):
        run(table(a), table(b), meta={"A": META, "B": {**META, "return_method": "log"}})
    with pytest.raises(ValueError, match="currency/frequency"):
        run(table(a), table(b), benchmark=table([0.]*4), benchmark_metadata={**BMETA, "currency": "EUR"})
    with pytest.raises(ValueError, match="benchmark is required"):
        run(table(a), table(b), benchmark=None)
    # Rows outside a declared window are excluded explicitly and counted, never silently.
    longer = run(table(a + [.5]), table(b + [-.3]))
    assert weight(longer) == pytest.approx(weight(run(table(a), table(b))))
    assert longer.metadata["rows_outside_window"] == {"A": 1, "B": 1}
    assert longer.metadata["estimation_intervals"][-1] == ["2024-01-04", "2024-01-05"]


def report(returns, capital):
    nav, pnl = capital, []
    for r in returns:
        pnl.append(nav*r)
        nav += pnl[-1]
    d = days(len(returns))
    daily = pl.DataFrame({"period_start": d[:-1], "session": d[1:], "pnl": pnl})
    return rt.series_performance(daily, initial_capital=capital, periods_per_year=A, risk_free_annual_effective=0.,
                                 metadata={"source": "synthetic", "currency": "USD"})


def test_differently_sized_accounts_are_combined_by_returns_not_pooled_pnl():
    rng = random.Random(2)
    a = [rng.gauss(.001, .01) for _ in range(30)]
    b = [rng.gauss(.0005, .004) for _ in range(30)]
    small, large = report(a, 1_000_000.), report(b, 50_000_000.)
    from_reports = rt.information_ratio_weights({"A": small, "B": large}, benchmark="zero", periods_per_year=A,
        bounds={"A": (.1, .9), "B": (.1, .9)}, estimation_window=window(a), covariance="estimated")
    from_tables = allocate(a, b, bounds=((.1, .9), (.1, .9)))
    assert weight(from_reports) == pytest.approx(weight(from_tables), rel=1e-9)
    provenance = from_reports.metadata["strategy_provenance"]
    assert provenance["A"]["source_initial_capital"] == 1e6 and provenance["B"]["source_initial_capital"] == 5e7
    combo = rt.combine_strategies({"A": small, "B": large}, weights={"A": .5, "B": .5},
                                  evaluation_window=window(a), initial_capital=1_000_000.)
    expected = [.5*x + .5*y for x, y in zip(a, b)]
    assert combo.returns["combined_return"].to_list() == pytest.approx(expected, rel=1e-9)
    pooled = [(p + q)/(e + f) for p, q, e, f in zip(small.daily["pnl"], large.daily["pnl"],
                                                    small.daily["opening_equity"], large.daily["opening_equity"])]
    assert combo.returns["combined_return"].to_list() != pytest.approx(pooled, rel=1e-6)


def test_contributions_compounding_and_illustrative_equity_by_hand():
    strategies = {"A": table([.1, -.05]), "B": table([0., .02])}
    bench = table([.01, .01])
    combo = rt.combine_strategies(strategies, strategy_metadata={"A": META, "B": META}, weights={"A": .25, "B": .75},
                                  evaluation_window=window([0, 0]), benchmark=bench, benchmark_metadata=BMETA,
                                  initial_capital=1000.)
    # .25*.1 + .75*0 = .025; .25*-.05 + .75*.02 = .0025.
    assert combo.returns["combined_return"].to_list() == pytest.approx([.025, .0025])
    assert combo.returns["active_return"].to_list() == pytest.approx([.015, -.0075])
    assert combo.returns["wealth"].to_list() == pytest.approx([1.025, 1.0275625])
    sums = combo.contributions.group_by("session", maintain_order=True).agg(
        pl.col("return_contribution").sum(), pl.col("active_contribution").sum(), pl.col("illustrative_pnl_contribution").sum())
    assert sums["return_contribution"].to_list() == pytest.approx(combo.returns["combined_return"].to_list())
    assert sums["active_contribution"].to_list() == pytest.approx(combo.returns["active_return"].to_list())
    assert combo.illustrative["pnl"].to_list() == pytest.approx([25., 2.5625])
    assert combo.illustrative["equity"].to_list() == pytest.approx([1025., 1027.5625])
    assert sums["illustrative_pnl_contribution"].to_list() == pytest.approx(combo.illustrative["pnl"].to_list())
    sleeves = combo.contributions.filter(pl.col("strategy") == "B")["illustrative_sleeve_opening_equity"].to_list()
    assert sleeves == pytest.approx([750., 768.75])  # rebalanced to 75% of each opening equity
    assert combo.metadata["sample"] == "estimation_sample_unknown"
    assert any("not an executable" in item for item in combo.metadata["limitations"])
    plain = rt.combine_strategies(strategies, strategy_metadata={"A": META, "B": META}, weights={"A": .25, "B": .75},
                                  evaluation_window=window([0, 0]))
    assert plain.illustrative is None and plain.benchmark is None
    assert plain.returns["active_return"].to_list() == [None, None]
    for bad in ({"A": .5, "B": .6}, {"A": -.1, "B": 1.1}, {"A": 1.}):
        with pytest.raises(ValueError):
            rt.combine_strategies(strategies, strategy_metadata={"A": META, "B": META}, weights=bad,
                                  evaluation_window=window([0, 0]))


def test_zero_correlation_weights_are_also_evaluated_empirically():
    rng = random.Random(4)
    common = [rng.gauss(0, .01) for _ in range(100)]
    a = [.001 + c + rng.gauss(0, .005) for c in common]
    b = [.0006 + .6*c + rng.gauss(0, .004) for c in common]
    assumed = allocate(a, b, covariance="zero_correlation")
    w = weight(assumed)
    va, vb = stdev(a)**2, stdev(b)**2
    objective = math.sqrt(A)*(w*mean(a) + (1 - w)*mean(b))/math.sqrt(w*w*va + (1 - w)**2*vb)
    assert summary(assumed, "optimization_covariance")["value"] == pytest.approx(objective, rel=1e-9)
    assert summary(assumed, "observed_combined_returns")["value"] == pytest.approx(ir(w, a, b), rel=1e-9)
    assert summary(assumed, "observed_combined_returns")["value"] < objective  # positive correlation ignored
    estimated = allocate(a, b)
    assert summary(estimated, "observed_combined_returns")["value"] >= ir(w, a, b)
    zero = assumed.covariance.filter(pl.col("strategy") != pl.col("other_strategy"))
    assert zero["optimization_covariance"].to_list() == [0., 0.] and all(v > .5 for v in zero["empirical_correlation"])
    contributions = assumed.risk_contributions.group_by("basis").agg(pl.col("variance_contribution").sum(),
                                                                     pl.col("fraction_of_variance").sum())
    tracking = {basis: summary(assumed, basis, "annualized_tracking_error")["value"]
                for basis in ("optimization_covariance", "observed_combined_returns")}
    for basis, total, fraction in contributions.iter_rows():
        assert total == pytest.approx(tracking[basis]**2) and fraction == pytest.approx(1)


def test_estimation_and_evaluation_windows():
    rng = random.Random(8)
    a = [rng.gauss(.001, .01) for _ in range(40)]
    b = [rng.gauss(.0005, .005) for _ in range(40)]
    strategies = {"A": table(a), "B": table(b)}
    d = days(40)
    fitted = allocate(a, b, estimation_window=(d[0], d[25]))
    kw = dict(strategy_metadata={"A": META, "B": META}, weights=fitted, benchmark="zero")
    in_sample = rt.combine_strategies(strategies, evaluation_window=(d[0], d[25]), **kw)
    assert in_sample.metadata["sample"] == "in_sample"
    later = rt.combine_strategies(strategies, evaluation_window=(d[25], d[40]), **kw)
    assert later.metadata["sample"] == "after_estimation_window" and later.returns.height == 15
    assert later.metadata["weights"] == dict(fitted.weights.select("strategy", "weight").iter_rows())
    assert "cannot know" in later.metadata["sample_note"]
    expected = [weight(fitted)*x + weight(fitted, "B")*y for x, y in zip(a[25:], b[25:])]
    assert later.returns["combined_return"].to_list() == pytest.approx(expected)
    with pytest.raises(ValueError, match="overlaps"):
        rt.combine_strategies(strategies, evaluation_window=(d[20], d[40]), **kw)
    overlapping = rt.combine_strategies(strategies, evaluation_window=(d[20], d[40]), overlap="identify", **kw)
    assert overlapping.metadata["sample"] == "overlapping_estimation_window"
    assert overlapping.metadata["overlap"]["overlapping_intervals"] == 5
    assert overlapping.metadata["overlap"]["overlap_window"] == [d[20].isoformat(), d[25].isoformat()]
    later_fit = allocate(a, b, estimation_window=(d[20], d[40]))
    with pytest.raises(ValueError, match="precedes"):
        rt.combine_strategies(strategies, strategy_metadata={"A": META, "B": META}, weights=later_fit,
                              evaluation_window=(d[0], d[20]))


def test_combination_feeds_the_existing_series_report_unchanged():
    rng = random.Random(6)
    a = [rng.gauss(.001, .01) for _ in range(50)]
    b = [rng.gauss(.0005, .005) for _ in range(50)]
    bench = [rng.gauss(.0004, .008) for _ in range(50)]
    combo = rt.combine_strategies({"A": table(a), "B": table(b)}, strategy_metadata={"A": META, "B": META},
        weights=pl.DataFrame({"strategy": ["A", "B"], "weight": [.4, .6]}), evaluation_window=window(a),
        benchmark=table(bench), benchmark_metadata=BMETA, initial_capital=250_000.)
    result = rt.series_performance(combo.illustrative, initial_capital=250_000., periods_per_year=A,
        risk_free_annual_effective=0., benchmark=combo.benchmark, benchmark_metadata=combo.metadata["benchmark_metadata"],
        metadata=combo.metadata["series_metadata"], frequency=combo.metadata["frequency"])
    assert result.daily["simple_return"].to_list() == pytest.approx(combo.returns["combined_return"].to_list(), rel=1e-12)
    active = [.4*x + .6*y - m for x, y, m in zip(a, b, bench)]
    information_ratio = result.benchmark_comparison.filter(pl.col("metric") == "information_ratio")["value"].item()
    assert information_ratio == pytest.approx(math.sqrt(A)*mean(active)/stdev(active), rel=1e-9)
    assert result.daily["equity"][-1] == pytest.approx(250_000.*combo.returns["wealth"][-1])


def test_synthetic_example_runs_offline():
    import runpy
    from pathlib import Path
    example = runpy.run_path(str(Path(__file__).resolve().parents[1]/"examples"/"strategy_allocation.py"))
    assumed, empirical, combinations, comparison = example["run_example"]()
    for fit in (assumed, empirical):
        assert fit.metadata["status"] == "ok" and fit.metadata["bounds"] == {"carry": [.1, .9], "trend": [.1, .9]}
        assert all(.1 - 1e-12 <= w <= .9 + 1e-12 for w in fit.weights["weight"])
    providers = assumed.metadata["strategy_provenance"]
    assert providers["carry"]["source_initial_capital"] != providers["trend"]["source_initial_capital"]
    assert {c.metadata["sample"] for c in combinations.values()} == {"after_estimation_window"}
    assert set(comparison.values["scenario"]) == {"zero_correlation", "estimated"}
