"""Cross-sectional rank ICs and quantile returns of dated signals against supplied outcomes."""

from collections import defaultdict
import json
import math
from statistics import stdev

import polars as pl

from ._data import _table
from ._metrics import METRIC_SCHEMA, _pair
from ._results import SignalDiagnosticsResult

SIGNAL_SCHEMA = {"signal_date": pl.Date, "asset": pl.String, "signal": pl.Float64}
OUTCOME_SCHEMA = {"signal_date": pl.Date, "asset": pl.String, "period_start": pl.Date,
                  "period_end": pl.Date, "forward_return": pl.Float64}
EXCLUSION_SCHEMA = {"signal_date": pl.Date, "asset": pl.String, "reason": pl.String}
TIMINGS = {"same_session_close_assumed", "after_signal_session"}
GROUPINGS = {"equal_count", "extremes"}
TIES = {"asset_order", "reject_boundary_ties"}
KEYS = ["signal_date", "asset"]


def _average_ranks(values):
    """One-based ranks; tied values share the average of the ranks they occupy."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0]*len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j+1]] == values[order[i]]:
            j += 1
        for t in range(i, j+1):
            ranks[order[t]] = (i + j)/2 + 1
        i = j + 1
    return ranks


def _labels(n, quantiles, grouping):
    """Group labels (1 = lowest signal) by ascending position; None is unassigned."""
    if grouping == "equal_count":
        return [i*quantiles//n + 1 for i in range(n)]
    m = n // quantiles
    middle = n - 2*m
    labels = [1]*m
    if quantiles > 2:
        labels += [2 + j*(quantiles-2)//middle for j in range(middle)]
    else:
        labels += [None]*middle
    return labels + [quantiles]*m


def _rank_ic(signals, returns):
    n = len(signals)
    if n < 2:
        return None, "insufficient_assets"
    if len(set(signals)) == 1:
        return None, "constant_signal"
    if len(set(returns)) == 1:
        return None, "constant_outcome"
    value, _, status, _ = _pair(_average_ranks(signals), _average_ranks(returns))
    return value, status


def _summary_rows(values, unit, prefix, n_total):
    """Time-series mean, sample std and positive share over evaluated dates."""
    n = len(values)
    mean_row = (f"mean_{prefix}", math.fsum(values)/n if n else None, unit, n, "ok" if n else "no_evaluated_dates")
    std_row = (f"{prefix}_std", stdev(values) if n >= 2 else None, unit, n,
               "ok" if n >= 2 else "insufficient_dates")
    share_row = (f"positive_{prefix}_share", sum(v > 0 for v in values)/n if n else None, "fraction", n,
                 "ok" if n else "no_evaluated_dates")
    return [mean_row, std_row, share_row, (f"evaluated_{prefix}_dates", float(n), "dates", n, "ok"),
            (f"unevaluated_{prefix}_dates", float(n_total - n), "dates", n, "ok")]


def signal_diagnostics(signals, outcomes, *, timing, quantiles, grouping, ties, metadata,
                       exclusions=None, weighting="equal") -> SignalDiagnosticsResult:
    """Per-date Spearman rank ICs and equal-weight signal-quantile forward returns.

    Signals (``signal_date, asset, signal``) and outcomes (``signal_date, asset,
    period_start, period_end, forward_return``) are matched on ``signal_date`` and
    ``asset`` only. Every forward interval must end after it starts and must start
    after the signal session, or on it under ``timing='same_session_close_assumed'``;
    all outcomes of one signal date share one interval. An unmatched signal or
    outcome raises unless ``exclusions`` lists it with a reason. Nothing is shifted,
    filled or executed: quantile spreads are analytical return differences.
    """
    if timing not in TIMINGS:
        raise ValueError(f"timing must be one of {sorted(TIMINGS)}")
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise ValueError("quantiles must be an integer of at least 2")
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {sorted(GROUPINGS)}")
    if ties not in TIES:
        raise ValueError(f"ties must be one of {sorted(TIES)}")
    if weighting != "equal":
        raise ValueError("only weighting='equal' is supported; signal weighting is not defined")
    if not isinstance(metadata, dict) or any(not isinstance(metadata.get(k), str) or not metadata[k].strip()
                                             for k in ("signal_source", "return_source", "return_basis")):
        raise ValueError("metadata requires nonblank signal_source, return_source and return_basis")
    try:
        source = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be finite JSON") from exc
    sig = _table(signals, SIGNAL_SCHEMA, "signals", KEYS, nonempty=True)
    out = _table(outcomes, OUTCOME_SCHEMA, "outcomes", KEYS, nonempty=True)
    exc = (pl.DataFrame(schema=EXCLUSION_SCHEMA) if exclusions is None
           else _table(exclusions, EXCLUSION_SCHEMA, "exclusions", KEYS))

    for d, asset, start, end in out.select("signal_date", "asset", "period_start", "period_end").iter_rows():
        if end <= start:
            raise ValueError(f"outcome {asset} {d}: period_end {end} must be after period_start {start}")
        if start < d:
            raise ValueError(f"outcome {asset} {d}: interval starts {start}, before the signal session "
                             "(backward-looking outcome)")
        if start == d and timing != "same_session_close_assumed":
            raise ValueError(f"outcome {asset} {d}: interval starts at the signal session close; declare "
                             "timing='same_session_close_assumed' to permit it")
    per_date = out.group_by("signal_date").agg(pl.col("period_start").n_unique().alias("s"),
                                               pl.col("period_end").n_unique().alias("e"))
    mixed = per_date.filter((pl.col("s") > 1) | (pl.col("e") > 1))["signal_date"].sort().to_list()
    if mixed:
        raise ValueError(f"outcomes for one signal date must share one forward interval; differs on {mixed[:5]}")

    sig_keys, out_keys = set(sig.select(KEYS).rows()), set(out.select(KEYS).rows())
    unmatched = {**{k: "signal_without_outcome" for k in sig_keys - out_keys},
                 **{k: "outcome_without_signal" for k in out_keys - sig_keys}}
    reasons = dict(((d, a), r) for d, a, r in exc.iter_rows())
    if set(reasons) - set(unmatched):
        bad = sorted(set(reasons) - set(unmatched))[:5]
        raise ValueError(f"exclusions list matched signal/outcome pairs {bad}; exclusions explain only unmatched rows")
    unexplained = sorted(set(unmatched) - set(reasons))
    if unexplained:
        raise ValueError(f"{len(unexplained)} unmatched signal/outcome row(s) without an exclusion, e.g. "
                         f"{[(str(d), a, unmatched[(d, a)]) for d, a in unexplained[:5]]}; nothing is dropped "
                         "or filled implicitly")
    excluded = pl.DataFrame([(d, a, unmatched[(d, a)], reasons[(d, a)]) for d, a in sorted(unmatched)],
        schema={"signal_date": pl.Date, "asset": pl.String, "side": pl.String, "reason": pl.String}, orient="row")

    matched = sig.join(out, on=KEYS, how="inner").sort("signal_date", "asset")
    dates = sorted(set(sig["signal_date"].to_list()) | set(out["signal_date"].to_list()))
    groups = defaultdict(list)
    for row in matched.iter_rows(named=True):
        groups[row["signal_date"]].append(row)
    intervals = dict(((d, (s, e)) for d, s, e in out.select("signal_date", "period_start", "period_end").unique().iter_rows()))
    counts = {name: dict(t.group_by("signal_date").len().iter_rows()) for name, t in (("signals", sig), ("outcomes", out))}
    excluded_counts = dict(excluded.group_by("signal_date").len().iter_rows()) if excluded.height else {}

    ic_rows, q_rows, spread_rows, coverage_rows = [], [], [], []
    for d in dates:
        rows = groups.get(d, [])
        n = len(rows)
        start, end = intervals.get(d, (None, None))
        signal_values = [r["signal"] for r in rows]
        return_values = [r["forward_return"] for r in rows]
        ic, ic_status = _rank_ic(signal_values, return_values)
        ic_rows.append((d, start, end, n, ic, ic_status))

        # Grouping order: ascending signal, ties broken by asset identifier (deterministic).
        ordered = sorted(rows, key=lambda r: (r["signal"], r["asset"]))
        group_status, labels = "ok", []
        if n < quantiles:
            group_status = "insufficient_assets_for_quantiles"
        elif len(set(signal_values)) == 1:
            group_status = "constant_signal"
        else:
            labels = _labels(n, quantiles, grouping)
            if ties == "reject_boundary_ties":
                by_signal = defaultdict(set)
                for r, label in zip(ordered, labels):
                    by_signal[r["signal"]].add(label)
                if any(len(v) > 1 for v in by_signal.values()):
                    group_status = "tie_at_group_boundary"
        members = defaultdict(list)
        if group_status == "ok":
            for r, label in zip(ordered, labels):
                if label is not None:
                    members[label].append(r)
        means = {}
        for q in range(1, quantiles+1):
            group = members.get(q, [])
            if group_status == "ok":
                means[q] = math.fsum(r["forward_return"] for r in group)/len(group)
                q_rows.append((d, q, len(group), math.fsum(r["signal"] for r in group)/len(group), means[q], "ok"))
            else:
                q_rows.append((d, q, 0, None, None, group_status))
        unassigned = n - sum(len(v) for v in members.values()) if group_status == "ok" else n
        spread_rows.append((d, start, end, len(members.get(1, [])), len(members.get(quantiles, [])),
                            means[quantiles] - means[1] if group_status == "ok" else None, group_status))
        coverage_rows.append((d, counts["signals"].get(d, 0), counts["outcomes"].get(d, 0), n,
                              excluded_counts.get(d, 0), unassigned))

    ic_table = pl.DataFrame(ic_rows, schema={"signal_date": pl.Date, "period_start": pl.Date, "period_end": pl.Date,
        "n_assets": pl.Int64, "rank_ic": pl.Float64, "status": pl.String}, orient="row")
    q_table = pl.DataFrame(q_rows, schema={"signal_date": pl.Date, "quantile": pl.Int64, "n_assets": pl.Int64,
        "mean_signal": pl.Float64, "mean_forward_return": pl.Float64, "status": pl.String}, orient="row")
    spreads = pl.DataFrame(spread_rows, schema={"signal_date": pl.Date, "period_start": pl.Date,
        "period_end": pl.Date, "n_low": pl.Int64, "n_high": pl.Int64, "high_minus_low": pl.Float64,
        "status": pl.String}, orient="row")
    coverage = pl.DataFrame(coverage_rows, schema={"signal_date": pl.Date, "n_signals": pl.Int64,
        "n_outcomes": pl.Int64, "n_matched": pl.Int64, "n_excluded": pl.Int64, "n_unassigned": pl.Int64}, orient="row")

    ics = [v for v, s in zip(ic_table["rank_ic"], ic_table["status"]) if s == "ok"]
    spread_values = [v for v, s in zip(spreads["high_minus_low"], spreads["status"]) if s == "ok"]
    sizes = coverage["n_matched"]
    rows = (_summary_rows(ics, "correlation", "rank_ic", len(dates))
            + _summary_rows(spread_values, "fraction/holding_period", "high_minus_low", len(dates))
            + [("signal_dates", float(len(dates)), "dates", len(dates), "ok"),
               ("min_assets_per_date", float(sizes.min()), "assets", len(dates), "ok"),
               ("mean_assets_per_date", float(sizes.mean()), "assets", len(dates), "ok"),
               ("max_assets_per_date", float(sizes.max()), "assets", len(dates), "ok"),
               ("asset_date_observations", float(matched.height), "asset_dates", len(dates), "ok")])
    summary = pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row")
    by_quantile = defaultdict(list)
    for q, value, status in q_table.select("quantile", "mean_forward_return", "status").iter_rows():
        if status == "ok":
            by_quantile[q].append(value)
    quantile_summary = pl.DataFrame(
        [(q, math.fsum(by_quantile[q])/len(by_quantile[q]) if by_quantile[q] else None, len(by_quantile[q]),
          "ok" if by_quantile[q] else "no_evaluated_dates") for q in range(1, quantiles+1)],
        schema={"quantile": pl.Int64, "mean_forward_return": pl.Float64, "n_dates": pl.Int64, "status": pl.String},
        orient="row")

    ordered_intervals = sorted((s, e) for s, e in intervals.values())
    overlapping = any(ordered_intervals[i][1] > ordered_intervals[i+1][0] for i in range(len(ordered_intervals)-1))
    meta = dict(source_metadata=source, timing=timing, quantiles=quantiles, grouping=grouping, ties=ties,
        weighting=weighting, matching="explicit keys signal_date and asset; never row position",
        rank_ic="per-date Spearman correlation: Pearson correlation of average ranks (ties share mean rank)",
        ic_aggregation="equal-weight time-series mean and ddof=1 std over evaluated dates; not pooled",
        quantile_assignment=("ascending signal, ties broken by asset identifier; equal_count label floor(i*Q/n)+1"
                             if grouping == "equal_count" else
                             "ascending signal, ties broken by asset identifier; lowest and highest floor(n/Q) "
                             "assets form groups 1 and Q; remaining middle assets split equal-count into groups "
                             "2..Q-1 (unassigned when Q=2)"),
        group_tie_policy=("ties split across groups by asset order" if ties == "asset_order" else
                          "a date whose tied signals straddle a group boundary is not grouped"),
        quantile_returns="equal-weight arithmetic mean of supplied simple forward returns per group and date",
        spread="highest-minus-lowest group mean; an analytical difference, not an executable portfolio "
               "(no costs, financing, hedging or execution)",
        overlapping_forward_intervals=overlapping,
        independence=("evaluated dates are the time-series sample size; asset-date counts are not independent "
                      "time observations" + ("; overlapping forward intervals make dates serially dependent"
                                             if overlapping else "")),
        horizon_calendar_days={"min": min(e - s for s, e in ordered_intervals).days,
                               "max": max(e - s for s, e in ordered_intervals).days} if ordered_intervals else None,
        n_dates=len(dates), n_matched=matched.height, n_excluded=excluded.height)
    return SignalDiagnosticsResult(ic_table, q_table, spreads, summary, quantile_summary, coverage, excluded, meta)
