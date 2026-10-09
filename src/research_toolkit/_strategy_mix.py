"""Two-strategy information-ratio allocation and analytical fixed-weight combination.

Both functions work on periodic simple net-equity returns of already-costed
strategies. They never pool dollar P&L, never rerun a backtest and never model a
merged executable account.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import math
from statistics import mean

import polars as pl

from ._data import _table
from ._factors import _json, _metadata
from ._portfolio import _finite_above, _number
from ._results import PerformanceResult, StrategyAllocationResult, StrategyCombinationResult
from ._risk_free import RETURN_SCHEMA, _aligned_returns, _intervals

COVARIANCE_MODES = ("estimated", "zero_correlation")
TABLE_KEYS = ("source", "currency", "frequency", "return_basis", "return_method")
WEIGHT_TOLERANCE = 1e-12
# A combination's tracking error is treated as zero when it is at most this
# fraction of w1*vol1 + w2*vol2 (its value under perfect positive correlation).
ZERO_TRACKING_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12
NEAR_SINGULAR = 1e-8
SINGULAR = 1e-14
OBJECTIVE = "sqrt(periods_per_year) * (w' mean(a)) / sqrt(w' covariance(a) w), a_i,t = r_i,t - b_t"
ZERO_CORRELATION_NOTE = (
    "zero_correlation is a counterfactual assumption about the active-return series entering the objective: "
    "each active variance is retained and the off-diagonal covariance is set to zero. With a nonzero benchmark "
    "it concerns active returns, not necessarily raw strategy returns. Observed combined returns are evaluated "
    "separately under the empirical covariance.")
COMBINATION_LIMITATIONS = [
    "fixed weights applied every period: a periodically rebalanced allocation between strategy sleeves, "
    "not buy-and-hold sleeves with drifting weights",
    "supplied strategy returns retain their own modeled costs; no sleeve-reallocation costs are charged",
    "no cross-strategy trade netting, shared collateral benefits or financing offsets are modeled",
    "scaling a strategy's return series does not recalculate its market impact or borrowing economics; "
    "strategies with fixed dollar targets or nonlinear costs may not scale proportionally",
    "analytical return combination, not an executable combined-account backtest",
]


def _strategies(strategies, strategy_metadata, allow_partial):
    """Exactly two named simple net-equity return series sharing currency and frequency."""
    if (not isinstance(strategies, Mapping) or len(strategies) != 2
            or any(not isinstance(k, str) or not k.strip() for k in strategies)):
        raise ValueError("strategies must map exactly two nonblank names to return tables or PerformanceResult objects")
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be boolean")
    names = sorted(strategies)
    tables = [n for n in names if not isinstance(strategies[n], PerformanceResult)]
    supplied = {} if strategy_metadata is None else strategy_metadata
    if not isinstance(supplied, Mapping) or set(supplied) != set(tables):
        raise ValueError(f"strategy_metadata must describe exactly the strategies supplied as return tables: {tables}")
    values, provenance = {}, {}
    for name in names:
        item = strategies[name]
        label = f"strategy {name!r}"
        if isinstance(item, PerformanceResult):
            m = item.metadata
            if m.get("status") not in {"complete", "stopped"} or (m["status"] != "complete" and not allow_partial):
                raise ValueError(f"{label} is a stopped or unknown run; pass allow_partial=True to use its actual coverage")
            if m.get("return_basis") != "net_equity":
                raise ValueError(f"{label} must report net_equity returns")
            if set(RETURN_SCHEMA) - set(item.daily.columns):
                raise ValueError(f"{label} daily table lacks {list(RETURN_SCHEMA)}")
            table = _table(item.daily.select(list(RETURN_SCHEMA)), RETURN_SCHEMA, label, ["session"], nonempty=True)
            meta = {"source": "PerformanceResult", "currency": m.get("currency"), "frequency": m.get("frequency"),
                    "return_basis": "net_equity", "return_method": "simple", "run_status": m["status"],
                    "stop_reason": m.get("stop_reason"), "source_initial_capital": m.get("initial_capital")}
        else:
            meta = _metadata(supplied[name], f"strategy_metadata[{name!r}]", TABLE_KEYS)
            if meta["return_method"] != "simple":
                raise ValueError(f"{label} must contain simple returns; log or other returns are not converted "
                                 "silently, so convert them explicitly before allocation")
            if meta["return_basis"] != "net_equity":
                raise ValueError(f"{label} return_basis must be net_equity (periodic simple returns on equity)")
            table = _table(item, RETURN_SCHEMA, label, ["session"], nonempty=True)
        if (table["simple_return"] <= -1).any():
            raise ValueError(f"{label} simple returns must exceed -1")
        values[name], provenance[name] = table.sort("session"), meta
    currencies = {p["currency"] for p in provenance.values()}
    frequencies = {p["frequency"] for p in provenance.values()}
    if len(currencies) != 1 or not all(isinstance(c, str) and c.strip() for c in currencies):
        raise ValueError(f"strategies must share one declared currency; got {sorted(map(str, currencies))}")
    if len(frequencies) != 1 or not all(isinstance(f, str) and f.strip() for f in frequencies):
        raise ValueError(f"strategies must share one declared frequency; got {sorted(map(str, frequencies))}")
    return names, values, provenance, currencies.pop(), frequencies.pop()


def _window(window, name):
    if (not isinstance(window, tuple) or len(window) != 2 or any(type(d) is not date for d in window)
            or window[0] >= window[1]):
        raise ValueError(f"{name} must be a (first_period_start, last_session) tuple of datetime.date, start < end")
    return window


def _select(table, window, label):
    """Intervals inside an inclusive declared window; crossing or missing endpoints raise."""
    start, end = window
    crossing = table.filter(((pl.col("period_start") < start) & (pl.col("session") > start))
                            | ((pl.col("period_start") < end) & (pl.col("session") > end)))
    if crossing.height:
        raise ValueError(f"{label} has interval(s) crossing the window boundary, e.g. "
                         f"{crossing['period_start'][0]} to {crossing['session'][0]}; nothing is truncated")
    inside = table.filter((pl.col("period_start") >= start) & (pl.col("session") <= end))
    if inside.is_empty() or inside["period_start"].min() != start or inside["session"].max() != end:
        raise ValueError(f"{label} must have an interval starting at {start} and one ending at {end}; "
                         "window endpoints must be observed interval endpoints")
    try:
        _intervals(inside.select("period_start", "session"))
    except ValueError as exc:
        raise ValueError(f"{label} intervals inside the window must be contiguous; missing observations are "
                         "not filled") from exc
    return inside, table.height - inside.height


def _sample(names, values, benchmark, benchmark_metadata, window, currency, frequency, *, required):
    """Strictly matched strategy (and benchmark) returns inside one declared window."""
    selected, outside = {}, {}
    for name in names:
        selected[name], outside[name] = _select(values[name], window, f"strategy {name!r}")
    intervals = selected[names[0]].select("period_start", "session")
    other = selected[names[1]].select("period_start", "session")
    if not other.equals(intervals):
        missing = intervals.join(other, on=["period_start", "session"], how="anti").height
        extra = other.join(intervals, on=["period_start", "session"], how="anti").height
        raise ValueError(f"strategy intervals must match on both endpoints exactly inside the window "
                         f"({missing} interval(s) of {names[0]!r} unmatched, {extra} extra in {names[1]!r}); "
                         "no dates are intersected, filled or dropped")
    if benchmark is None:
        if required:
            raise ValueError("benchmark is required: 'zero' or a strictly aligned return table")
        if benchmark_metadata is not None:
            raise ValueError("benchmark_metadata requires a benchmark")
        bench, bmeta = None, None
    elif isinstance(benchmark, str):
        if benchmark != "zero":
            raise ValueError("benchmark must be 'zero' or a Polars return table")
        if benchmark_metadata is not None:
            raise ValueError("the zero benchmark takes no benchmark_metadata")
        bench, bmeta = [0.0]*intervals.height, {"type": "zero_return"}
    else:
        table = _table(benchmark, RETURN_SCHEMA, "benchmark", ["session"], nonempty=True).sort("session")
        table, out = _select(table, window, "benchmark")
        table, meta = _aligned_returns(table, intervals, benchmark_metadata, currency, "benchmark", frequency)
        bench, bmeta = table["simple_return"].to_list(), {"type": "supplied", "metadata": meta}
        outside["benchmark"] = out
    returns = {n: selected[n]["simple_return"].to_list() for n in names}
    return intervals, returns, bench, bmeta, outside


def _bounds(bounds, names):
    if not isinstance(bounds, Mapping) or set(bounds) != set(names):
        raise ValueError(f"bounds must give (lower, upper) for exactly the strategies {names}")
    checked = {}
    for name in names:
        pair = bounds[name]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"bounds[{name!r}] must be a (lower, upper) pair")
        lo, hi = (_number(v, f"bounds[{name!r}]") for v in pair)
        if lo > hi or hi > 1:
            raise ValueError(f"bounds[{name!r}] must satisfy 0 <= lower <= upper <= 1; strategy-level "
                             "shorting and leverage are not supported")
        checked[name] = (lo, hi)
    (l1, u1), (l2, u2) = checked[names[0]], checked[names[1]]
    # Each feasible end is stored as an exact (w1, w2) pair from whichever bound defines it.
    low = (l1, 1 - l1) if l1 >= 1 - u2 else (1 - u2, u2)
    high = (u1, 1 - u1) if u1 <= 1 - l2 else (1 - l2, l2)
    if low[0] > high[0] + WEIGHT_TOLERANCE:
        raise ValueError(f"infeasible bounds: no nonnegative weights summing to one satisfy {checked}")
    if low[0] > high[0]:
        low = high
    return checked, low, high


def _moments(active, names):
    """Exact means (statistics.mean) and ddof=1 sample covariances of active returns."""
    n = len(active[names[0]])
    mu = {k: mean(active[k]) for k in names}
    dev = {k: [v - mu[k] for v in active[k]] for k in names}
    cov = {(i, j): math.fsum(x*y for x, y in zip(dev[i], dev[j]))/(n - 1) for i in names for j in names}
    if any(not math.isfinite(v) for v in (*mu.values(), *cov.values())):
        raise ValueError("active-return moments are not representable")
    return mu, cov


def _correlation(cov, i, j):
    if cov[i, i] <= 0 or cov[j, j] <= 0:
        return None, "zero_variance"
    return max(-1., min(1., cov[i, j]/math.sqrt(cov[i, i])/math.sqrt(cov[j, j]))), "ok"


def _observed(weights, active, names):
    """Mean and ddof=1 standard deviation of the observed combined active series."""
    combined = [math.fsum(weights[k]*active[k][t] for k in names) for t in range(len(active[names[0]]))]
    m = mean(combined)
    return m, math.sqrt(math.fsum((v - m)**2 for v in combined)/(len(combined) - 1))


def _evaluate(weights, mu, cov, names, active, use_observed):
    m = math.fsum(weights[k]*mu[k] for k in names)
    if use_observed:
        te = _observed(weights, active, names)[1]
    else:
        te = math.sqrt(max(0., math.fsum(weights[i]*weights[j]*cov[i, j] for i in names for j in names)))
    scale = math.fsum(weights[k]*math.sqrt(cov[k, k]) for k in names)
    degenerate = scale == 0 or te <= ZERO_TRACKING_TOLERANCE*scale
    return m, te, degenerate


def _contributions(weights, cov, names, a):
    total = math.fsum(weights[i]*weights[j]*cov[i, j] for i in names for j in names)
    rows = []
    for i in names:
        value = weights[i]*math.fsum(cov[i, j]*weights[j] for j in names)*a
        fraction = value/(total*a) if total > 0 else None
        rows.append((i, weights[i], value, fraction, "ok" if total > 0 else "zero_tracking_error"))
    return rows


def information_ratio_weights(strategies, *, benchmark, periods_per_year, bounds, estimation_window, covariance,
                              strategy_metadata=None, benchmark_metadata=None,
                              allow_partial=False) -> StrategyAllocationResult:
    """Long-only weights of two strategies maximizing the combined information ratio.

    ``strategies`` maps two names to ``period_start, session, simple_return`` tables
    (with ``strategy_metadata``) or to ``PerformanceResult`` objects. ``benchmark``
    is ``'zero'`` or a strictly aligned return table with ``benchmark_metadata``.
    Intervals inside the inclusive ``estimation_window`` must match on both
    endpoints. ``covariance`` is ``'estimated'`` (sample, ddof=1) or
    ``'zero_correlation'`` (variances kept, covariance set to zero). The global
    maximum over the feasible interval is found from its endpoints and the unique
    interior stationary point. Undefined optima return null weights with a status.
    """
    a = _finite_above(periods_per_year, "periods_per_year", lower=0)
    if covariance not in COVARIANCE_MODES:
        raise ValueError(f"covariance must be one of {list(COVARIANCE_MODES)}")
    names, values, provenance, currency, frequency = _strategies(strategies, strategy_metadata, allow_partial)
    window = _window(estimation_window, "estimation_window")
    intervals, returns, bench, bmeta, outside = _sample(names, values, benchmark, benchmark_metadata, window,
                                                         currency, frequency, required=True)
    n = intervals.height
    if n < 2:
        raise ValueError("at least two estimation intervals are required for a ddof=1 covariance")
    checked, low, high = _bounds(bounds, names)
    first, second = names
    active = {k: [r - b for r, b in zip(returns[k], bench)] for k in names}
    mu, empirical = _moments(active, names)
    used = dict(empirical) if covariance == "estimated" else {
        (i, j): (empirical[i, j] if i == j else 0.) for i in names for j in names}
    raw_mu, raw_cov = _moments(returns, names)
    root = math.sqrt(a)

    # Candidates: both feasible ends, the stationary point of IR(w) (linear first-order
    # condition, so at most one), and the minimum-tracking-error point that locates any
    # riskless active combination.
    c11, c22, c12 = used[first, first], used[second, second], used[first, second]
    candidates = [("lower_endpoint", *low)]
    if high[0] != low[0]:
        candidates.append(("upper_endpoint", *high))
    den = mu[first]*c22 - mu[second]*c12 + mu[second]*c11 - mu[first]*c12
    if den != 0:
        w = (mu[first]*c22 - mu[second]*c12)/den
        if low[0] < w < high[0]:
            candidates.append(("stationary", w, 1 - w))
    spread = c11 + c22 - 2*c12
    if spread > 0:
        w = (c22 - c12)/spread
        if low[0] < w < high[0]:
            candidates.append(("minimum_tracking_error", w, 1 - w))
    use_observed = covariance == "estimated"
    mean_tol = TIE_TOLERANCE*max(abs(mu[first]), abs(mu[second]))
    evaluated = []
    for kind, w1, w2 in candidates:
        weights = {first: w1, second: w2}
        m, te, degenerate = _evaluate(weights, mu, used, names, active, use_observed)
        ir = None if degenerate else root*m/te
        status = ("zero_tracking_error_positive_mean" if degenerate and m > mean_tol else
                  "zero_tracking_error" if degenerate else "ok")
        evaluated.append([kind, w1, w2, m*a, te*root, ir, status, False])
    valid = [c for c in evaluated if c[5] is not None]
    if any(c[6] == "zero_tracking_error_positive_mean" for c in evaluated):
        status, optimum, chosen = "unbounded_information_ratio", "none", None
    elif not valid:
        status, optimum, chosen = "undefined_zero_tracking_error", "none", None
    else:
        best = max(c[5] for c in valid)
        tol = TIE_TOLERANCE*max(1., abs(best))
        winners = sorted({c[1]: c for c in valid if c[5] >= best - tol}.values(), key=lambda c: c[1])
        chosen = winners[0]
        status, optimum = "ok", "unique"
        if len(winners) > 1:
            optimum = "multiple_optima"
            lo_w, hi_w = winners[0][1], winners[-1][1]
            between = [c for c in evaluated if c[6] != "ok" and lo_w < c[1] < hi_w]
            mid = {first: (lo_w + hi_w)/2, second: 1 - (lo_w + hi_w)/2}
            m, te, degenerate = _evaluate(mid, mu, used, names, active, use_observed)
            if not between and not degenerate and root*m/te >= best - tol:
                optimum = "flat_objective"
        chosen[7] = True
    for row in evaluated:
        if any(v is not None and not math.isfinite(v) for v in row[1:6]):
            raise ValueError("information-ratio candidate is not representable")
    candidate_table = pl.DataFrame(evaluated, schema={"kind": pl.String, "first_strategy_weight": pl.Float64,
        "second_strategy_weight": pl.Float64, "annualized_mean_active_return": pl.Float64,
        "annualized_tracking_error": pl.Float64, "information_ratio": pl.Float64, "status": pl.String,
        "selected": pl.Boolean}, orient="row")

    selected = {first: chosen[1], second: chosen[2]} if chosen else None
    weight_rows = []
    for k in names:
        lo, hi = checked[k]
        w = selected[k] if selected else None
        weight_rows.append((k, w, lo, hi,
                            None if w is None else abs(w - lo) <= WEIGHT_TOLERANCE,
                            None if w is None else abs(w - hi) <= WEIGHT_TOLERANCE, status))
    weights = pl.DataFrame(weight_rows, schema={"strategy": pl.String, "weight": pl.Float64,
        "lower_bound": pl.Float64, "upper_bound": pl.Float64, "at_lower_bound": pl.Boolean,
        "at_upper_bound": pl.Boolean, "status": pl.String}, orient="row")

    estimate_rows = []
    for k in names:
        vol = math.sqrt(empirical[k, k])
        estimate_rows.append((k, n, raw_mu[k]*a, math.sqrt(raw_cov[k, k]*a), mu[k], mu[k]*a, vol*root,
                              root*mu[k]/vol if vol > 0 else None, "ok" if vol > 0 else "zero_tracking_error"))
    estimates = pl.DataFrame(estimate_rows, schema={"strategy": pl.String, "n_obs": pl.Int64,
        "annualized_mean_return": pl.Float64, "annualized_volatility": pl.Float64,
        "mean_active_return": pl.Float64, "annualized_mean_active_return": pl.Float64,
        "annualized_tracking_error": pl.Float64, "information_ratio": pl.Float64,
        "information_ratio_status": pl.String}, orient="row")

    cov_rows = []
    for i in names:
        for j in names:
            ec, ecs = _correlation(empirical, i, j)
            uc, ucs = _correlation(used, i, j)
            rc, rcs = _correlation(raw_cov, i, j)
            cov_rows.append((i, j, empirical[i, j]*a, used[i, j]*a, ec, ecs, uc, ucs, rc, rcs))
    covariance_table = pl.DataFrame(cov_rows, schema={"strategy": pl.String, "other_strategy": pl.String,
        "empirical_covariance": pl.Float64, "optimization_covariance": pl.Float64,
        "empirical_correlation": pl.Float64, "empirical_correlation_status": pl.String,
        "optimization_correlation": pl.Float64, "optimization_correlation_status": pl.String,
        "raw_return_correlation": pl.Float64, "raw_return_correlation_status": pl.String}, orient="row")

    summary_rows, contribution_rows = [], []
    for basis, matrix, observed in (("optimization_covariance", used, use_observed),
                                    ("observed_combined_returns", empirical, True)):
        if selected is None:
            summary_rows += [(basis, metric, None, unit, n, status) for metric, unit in (
                ("annualized_mean_active_return", "fraction/year"),
                ("annualized_tracking_error", "fraction/sqrt(year)"), ("information_ratio", "ratio"))]
            contribution_rows += [(basis, k, None, None, None, status) for k in names]
            continue
        if observed:
            m, te = _observed(selected, active, names)
            degenerate = te <= ZERO_TRACKING_TOLERANCE*math.fsum(selected[k]*math.sqrt(empirical[k, k]) for k in names)
        else:
            m, te, degenerate = _evaluate(selected, mu, matrix, names, active, False)
        ir_status = "zero_tracking_error" if degenerate else "ok"
        summary_rows += [(basis, "annualized_mean_active_return", m*a, "fraction/year", n, "ok"),
                         (basis, "annualized_tracking_error", te*root, "fraction/sqrt(year)", n, "ok"),
                         (basis, "information_ratio", None if degenerate else root*m/te, "ratio", n, ir_status)]
        contribution_rows += [(basis, *row) for row in _contributions(selected, matrix, names, a)]
    summary = pl.DataFrame(summary_rows, schema={"basis": pl.String, "metric": pl.String, "value": pl.Float64,
        "unit": pl.String, "n_obs": pl.Int64, "status": pl.String}, orient="row")
    contributions = pl.DataFrame(contribution_rows, schema={"basis": pl.String, "strategy": pl.String,
        "weight": pl.Float64, "variance_contribution": pl.Float64, "fraction_of_variance": pl.Float64,
        "status": pl.String}, orient="row")
    for table in (estimates, covariance_table, summary, contributions):
        if any(v is not None and not math.isfinite(v) for c, t in table.schema.items() if t == pl.Float64 for v in table[c]):
            raise ValueError("allocation output is not representable")

    sample = pl.concat([intervals.with_columns(pl.lit(k).alias("strategy"),
                                               pl.Series("simple_return", returns[k], dtype=pl.Float64),
                                               pl.Series("benchmark_return", bench, dtype=pl.Float64),
                                               pl.Series("active_return", active[k], dtype=pl.Float64))
                        for k in names])
    rho = _correlation(used, first, second)[0]
    condition = 1 - rho*rho if rho is not None else None
    covariance_status = ("singular" if condition is None or condition <= SINGULAR else
                         "near_singular" if condition <= NEAR_SINGULAR else "ok")
    meta = dict(
        method="two_strategy_information_ratio", status=status, optimum=optimum, strategies=names,
        first_strategy=first, second_strategy=second, strategy_provenance=provenance, currency=currency,
        frequency=frequency, return_basis="net_equity", return_method="simple", periods_per_year=a,
        covariance=covariance, covariance_status=covariance_status, one_minus_correlation_squared=condition,
        benchmark=bmeta, estimation_window=[window[0].isoformat(), window[1].isoformat()],
        estimation_intervals=[[s.isoformat(), e.isoformat()] for s, e in intervals.iter_rows()],
        n_obs=n, rows_outside_window=outside, ddof=1,
        bounds={k: list(v) for k, v in checked.items()},
        feasible_first_strategy_weight=[low[0], high[0]],
        weight_convention="long_only_fully_invested: nonnegative weights summing to one, no strategy-level "
                          "shorting, leverage or cash",
        objective=OBJECTIVE, active_return_definition="strategy simple return minus the common benchmark return",
        annualization="sqrt_periods_per_year_no_serial_correlation_adjustment",
        units={"mean": "fraction/year", "volatility": "fraction/sqrt(year)", "covariance": "fraction_squared/year",
               "mean_active_return": "fraction/period"},
        zero_correlation_note=ZERO_CORRELATION_NOTE,
        objective_basis=("optimization_covariance is the assumed objective; observed_combined_returns is the IR "
                         "implied by the observed combined active returns (empirical covariance)"),
        selection_rule=("global maximum of IR(w) over feasible endpoints and the interior stationary point; ties "
                        f"within relative {TIE_TOLERANCE} select the smallest weight on the first strategy in name order"),
        tolerances={"weight": WEIGHT_TOLERANCE, "zero_tracking_error_relative": ZERO_TRACKING_TOLERANCE,
                    "tie_relative": TIE_TOLERANCE, "near_singular_one_minus_rho_squared": NEAR_SINGULAR,
                    "singular_one_minus_rho_squared": SINGULAR},
        undefined_policy=("a feasible weight with zero tracking error and positive mean active return makes IR "
                          "unbounded; zero tracking error everywhere leaves it undefined; both return null weights. "
                          "Undefined IR is never replaced by zero and no covariance shrinkage is applied."),
        regularization=None)
    return StrategyAllocationResult(weights, estimates, covariance_table, summary, contributions,
                                    candidate_table, sample, meta)


def _combination_weights(weights, names):
    """Weights from an allocation result, a mapping, or a strategy/weight table."""
    allocation = None
    if isinstance(weights, StrategyAllocationResult):
        allocation = weights
        if weights.metadata.get("status") != "ok":
            raise ValueError(f"allocation status is {weights.metadata.get('status')!r}; it has no weights to apply")
        weights = dict(weights.weights.select("strategy", "weight").iter_rows())
    elif isinstance(weights, pl.DataFrame):
        if weights.schema != {"strategy": pl.String, "weight": pl.Float64}:
            raise ValueError("weights table schema must be strategy: String, weight: Float64")
        if weights["strategy"].is_duplicated().any():
            raise ValueError("duplicate weight strategies")
        weights = dict(weights.iter_rows())
    if not isinstance(weights, Mapping) or set(weights) != set(names):
        raise ValueError(f"weights must cover exactly the strategies {names}")
    checked = {k: _number(weights[k], f"weight[{k!r}]") for k in names}
    total = math.fsum(checked.values())
    if not math.isclose(total, 1., rel_tol=0, abs_tol=WEIGHT_TOLERANCE):
        raise ValueError(f"weights must be nonnegative and sum to one; got {total}")
    return checked, allocation


def _sample_label(allocation, intervals, overlap):
    if allocation is None:
        return "estimation_sample_unknown", None
    estimated = [tuple(date.fromisoformat(d) for d in pair) for pair in allocation.metadata["estimation_intervals"]]
    evaluated = list(intervals.iter_rows())
    est_start, est_end = estimated[0][0], estimated[-1][1]
    ev_start, ev_end = evaluated[0][0], evaluated[-1][1]
    if evaluated == estimated:
        return "in_sample", {"overlapping_intervals": len(evaluated)}
    if ev_start >= est_end:
        return "after_estimation_window", {"overlapping_intervals": 0}
    if ev_end <= est_start:
        raise ValueError("the evaluation window precedes the estimation window: weights fitted on later data "
                         "would use future information")
    shared = sorted(set(estimated) & set(evaluated))
    info = {"overlapping_intervals": len(shared),
            "overlap_window": [max(ev_start, est_start).isoformat(), min(ev_end, est_end).isoformat()]}
    if overlap == "reject":
        raise ValueError(f"evaluation window {ev_start} to {ev_end} overlaps the estimation window {est_start} to "
                         f"{est_end}; pass overlap='identify' to label the overlap explicitly")
    return "overlapping_estimation_window", info


def combine_strategies(strategies, *, weights, evaluation_window, strategy_metadata=None, benchmark=None,
                       benchmark_metadata=None, initial_capital=None, overlap="reject",
                       allow_partial=False) -> StrategyCombinationResult:
    """Fixed-weight analytical combination of two strictly aligned strategy return series.

    Combined return is ``sum_i w_i r_i,t`` each period: a periodically rebalanced
    allocation between already-costed strategy sleeves, not an executable combined
    account. ``weights`` is an ``information_ratio_weights`` result (never
    re-estimated), a mapping, or a strategy/weight table. With an allocation, an
    evaluation window equal to its estimation sample is labeled ``in_sample``;
    partial overlap raises unless ``overlap='identify'``. ``initial_capital`` adds
    clearly illustrative equity and P&L.
    """
    if overlap not in {"reject", "identify"}:
        raise ValueError("overlap must be 'reject' or 'identify'")
    names, values, provenance, currency, frequency = _strategies(strategies, strategy_metadata, allow_partial)
    checked, allocation = _combination_weights(weights, names)
    if allocation is not None and allocation.metadata.get("strategies") != names:
        raise ValueError(f"allocation strategies {allocation.metadata.get('strategies')} differ from {names}")
    window = _window(evaluation_window, "evaluation_window")
    intervals, returns, bench, bmeta, outside = _sample(names, values, benchmark, benchmark_metadata, window,
                                                         currency, frequency, required=False)
    sample, overlap_info = _sample_label(allocation, intervals, overlap)
    capital = None if initial_capital is None else _finite_above(initial_capital, "initial_capital", lower=0)

    rows, contribution_rows, illustrative_rows = [], [], []
    wealth, equity = 1., capital
    for t, (start, session) in enumerate(intervals.iter_rows()):
        parts = {k: checked[k]*returns[k][t] for k in names}
        combined = math.fsum(parts.values())
        b = None if bench is None else bench[t]
        active = None if b is None else combined - b
        wealth *= 1 + combined
        if not math.isfinite(wealth) or wealth <= 0:
            raise ValueError("combined wealth is not representable")
        rows.append((start, session, combined, b, active, wealth, wealth - 1))
        opening = equity
        if capital is not None:
            pnl = opening*combined
            equity = opening + pnl
            illustrative_rows.append((start, session, opening, pnl, equity))
        for k in names:
            contribution_rows.append((start, session, k, checked[k], returns[k][t], parts[k],
                                      None if b is None else returns[k][t] - b,
                                      None if b is None else checked[k]*(returns[k][t] - b),
                                      None if capital is None else opening*checked[k],
                                      None if capital is None else opening*parts[k]))
    combined_table = pl.DataFrame(rows, schema={"period_start": pl.Date, "session": pl.Date,
        "combined_return": pl.Float64, "benchmark_return": pl.Float64, "active_return": pl.Float64,
        "wealth": pl.Float64, "compounded_return": pl.Float64}, orient="row")
    contributions = pl.DataFrame(contribution_rows, schema={"period_start": pl.Date, "session": pl.Date,
        "strategy": pl.String, "weight": pl.Float64, "strategy_return": pl.Float64,
        "return_contribution": pl.Float64, "active_return": pl.Float64, "active_contribution": pl.Float64,
        "illustrative_sleeve_opening_equity": pl.Float64, "illustrative_pnl_contribution": pl.Float64},
        orient="row")
    illustrative = None
    if capital is not None:
        illustrative = pl.DataFrame(illustrative_rows, schema={"period_start": pl.Date, "session": pl.Date,
            "opening_equity": pl.Float64, "pnl": pl.Float64, "equity": pl.Float64}, orient="row")
        if not math.isclose(illustrative["equity"][-1], capital*wealth, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("illustrative equity does not reconcile with compounded wealth")
    benchmark_table = None
    if bench is not None:
        benchmark_table = intervals.with_columns(pl.Series("simple_return", bench, dtype=pl.Float64))
    for table in (combined_table, contributions):
        if any(v is not None and not math.isfinite(v) for c, t in table.schema.items() if t == pl.Float64 for v in table[c]):
            raise ValueError("combination output is not representable")

    zero = bmeta is not None and bmeta["type"] == "zero_return"
    series_meta = {"source": "analytical two-strategy combination", "currency": currency, "frequency": frequency,
                   "strategies": names, "weights": checked, "sample": sample,
                   "construction": "fixed_weights_rebalanced_each_period", "limitations": COMBINATION_LIMITATIONS}
    report_benchmark_meta = None
    if bmeta is not None:
        report_benchmark_meta = ({"source": "zero_return_benchmark", "basis": "zero_return", "currency": currency,
                                  "frequency": frequency} if zero else deepcopy(bmeta["metadata"]))
    meta = dict(
        method="fixed_weight_analytical_combination", strategies=names, weights=checked,
        weights_source="information_ratio_weights" if allocation is not None else "supplied",
        allocation=None if allocation is None else {k: deepcopy(allocation.metadata[k]) for k in (
            "covariance", "estimation_window", "benchmark", "periods_per_year", "bounds", "status", "optimum")},
        sample=sample, overlap_policy=overlap, overlap=overlap_info,
        sample_note=("labels describe date overlap only; the function cannot know whether the evaluation data "
                     "were inspected before the weights were chosen, so no out-of-sample validity is claimed"),
        evaluation_window=[window[0].isoformat(), window[1].isoformat()], n_obs=intervals.height,
        rows_outside_window=outside, strategy_provenance=provenance, currency=currency, frequency=frequency,
        return_basis="net_equity", return_method="simple", benchmark=bmeta,
        construction="fixed weights applied every period (rebalanced sleeves); combined_return = sum_i w_i r_i,t",
        wealth_definition="compounded from 1 at the first period_start", initial_capital=capital,
        illustrative_equity=("illustrative only: initial_capital compounded at the combined return; not an "
                             "executable account" if capital is not None else None),
        limitations=COMBINATION_LIMITATIONS, series_metadata=series_meta,
        benchmark_metadata=report_benchmark_meta)
    return StrategyCombinationResult(combined_table, contributions, illustrative, benchmark_table, meta)
