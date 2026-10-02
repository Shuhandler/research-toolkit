"""Numerical scenario comparisons, with explicit mismatch handling and no prose."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import json
import math

import polars as pl

from ._results import PerformanceResult, ComparisonResult


def compare_performance(reports, *, benchmark_label=None, coverage="strict",
                        assumptions="strict", allow_partial=False) -> ComparisonResult:
    """Combine prepared reports without recalculation or hidden sample intersection.

    'separate' permits differences but retains every row's own coverage/units and
    reports the differences. If benchmark inputs/conventions differ, benchmark
    summaries are separately labeled per scenario. No investment conclusions.
    """
    if not isinstance(reports, Mapping) or not reports or any(not isinstance(k, str) or not k.strip() for k in reports):
        raise ValueError("reports must be a nonempty mapping of nonblank labels to PerformanceResult")
    if coverage not in {"strict", "separate"} or assumptions not in {"strict", "separate"} or type(allow_partial) is not bool:
        raise ValueError("coverage/assumptions must be strict or separate; allow_partial must be boolean")
    if benchmark_label is not None and (not isinstance(benchmark_label, str) or not benchmark_label.strip() or benchmark_label in reports):
        raise ValueError("benchmark_label must be nonblank and distinct from scenario labels")
    rows, diagnostics, seen = [], [], {}
    reference = None
    configs = {}
    for label, report in reports.items():
        if not isinstance(report, PerformanceResult):
            raise ValueError("compare_performance consumes PerformanceResult objects")
        m = report.metadata
        if m.get("status") not in {"complete", "stopped"} or (m["status"] != "complete" and not allow_partial):
            raise ValueError("incomplete/unknown run requires explicit allow_partial=True")
        signature = {k: m.get(k) for k in ("currency", "frequency", "return_basis", "periods_per_year",
            "sharpe_denominator", "minimum_acceptable_return_annual_effective", "sortino_denominator", "ddof", "initial_capital")}
        signature["risk_free"] = {"metadata": m.get("risk_free"), "annual_effective": m.get("risk_free_annual_effective"),
            "values": report.risk_free_returns.to_dicts()}
        signature["coverage"] = {"intervals": report.daily.select("period_start", "session").to_dicts(), "requested_end": m["end_session"]}
        signature["benchmark"] = {"metadata": m.get("benchmark"), "series": report.benchmark_series.to_dicts()}
        if reference is None:
            reference = signature
        for field, value in signature.items():
            # RF observations necessarily have different keys when coverage differs;
            # accepting separate coverage must not silently accept a changed RF rule.
            if value != reference[field]:
                encoded = lambda v: json.dumps(v, default=lambda x: x.isoformat(), sort_keys=True, allow_nan=False)
                diagnostics.append((label, field, encoded(reference[field]), encoded(value)))
                mode = coverage if field == "coverage" else assumptions
                if mode == "strict":
                    raise ValueError(f"scenario {label!r} has incompatible {field}; select explicit separate handling")
        configs[label] = deepcopy(m)
        metrics = pl.concat([report.summary, report.benchmark_comparison.filter(pl.col("metric").is_in(["return_correlation", "beta"]))])
        entries = [(label, "portfolio", metrics)]
        if benchmark_label is not None:
            if report.benchmark_summary.is_empty():
                raise ValueError("benchmark_label requires a benchmark in every report")
            key = json.dumps({"series": report.benchmark_series.to_dicts(), "summary": report.benchmark_summary.to_dicts(),
                "benchmark": m["benchmark"], "signature": signature}, default=lambda x: x.isoformat(), sort_keys=True)
            if key not in seen:
                name = benchmark_label if coverage == assumptions == "strict" else f"{benchmark_label} [{label}]"
                if name in reports:
                    raise ValueError("generated benchmark label conflicts with a scenario")
                entries.append((name, "benchmark", report.benchmark_summary))
                seen[key] = name
        for name, kind, table in entries:
            if table["metric"].is_duplicated().any():
                raise ValueError("report has duplicate metrics")
            for item in table.iter_rows(named=True):
                if item["value"] is not None and not math.isfinite(item["value"]):
                    raise ValueError("report metrics must be finite or explicitly undefined")
                rows.append((name, kind, item["metric"], item["value"], item["unit"], item["n_obs"], item["status"],
                    m["status"], m["stop_reason"], date.fromisoformat(m["entry_session"]), date.fromisoformat(m["actual_end_session"]), date.fromisoformat(m["end_session"])))
    values = pl.DataFrame(rows, schema={"scenario": pl.String, "kind": pl.String, "metric": pl.String,
        "value": pl.Float64, "unit": pl.String, "n_obs": pl.Int64, "metric_status": pl.String,
        "run_status": pl.String, "stop_reason": pl.String, "period_start": pl.Date,
        "actual_end_session": pl.Date, "requested_end_session": pl.Date}, orient="row")
    differences = pl.DataFrame(diagnostics, schema={"scenario": pl.String, "field": pl.String,
        "reference": pl.String, "actual": pl.String}, orient="row")
    return ComparisonResult(values, differences, {"coverage": coverage, "assumptions": assumptions,
        "allow_partial": allow_partial, "scenario_metadata": configs, "benchmark_label": benchmark_label})
