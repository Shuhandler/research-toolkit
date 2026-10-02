"""Chronological research inputs and training-only transforms; no model dependency."""
from copy import deepcopy
from datetime import date
import json
import math
from statistics import mean, stdev
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from ._data import _table, _identity, SESSION_SCHEMA
from ._results import FeatureResult, ResearchSplit

UTC = pl.Datetime("us", "UTC")
KEYS = {"session": pl.Date, "asset": pl.String}
LABEL_SCHEMA = {**KEYS, "label_end": pl.Date, "available_at": UTC, "target": pl.Float64}
TRANSFORM_SCHEMA = {"feature": pl.String, "mean": pl.Float64, "scale": pl.Float64,
                    "n_obs": pl.Int64, "status": pl.String}
SPLIT_TABLES = ("train", "validation", "test", "excluded", "transforms")


def _json_metadata(metadata, required):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a finite JSON dictionary")
    try:
        meta = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be finite JSON") from exc
    if any(not isinstance(meta.get(k), str) or not meta[k].strip() for k in required):
        raise ValueError(f"metadata requires nonblank {', '.join(required)}")
    return meta


def _calendar(sessions, metadata):
    meta = _json_metadata(metadata, ("source", "calendar", "calendar_version", "timezone", "frequency"))
    if meta["frequency"] != "1d":
        raise ValueError("research currently requires frequency='1d'")
    try:
        zone = ZoneInfo(meta["timezone"])
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("metadata.timezone must be a valid IANA timezone") from exc
    table = _table(sessions, SESSION_SCHEMA, "sessions", ["session"], nonempty=True).sort("session")
    days, closes = table["session"].to_list(), table["close_at"].to_list()
    if any(t.astimezone(zone).date() != d for d, t in zip(days, closes)) or any(a >= b for a, b in zip(closes, closes[1:])):
        raise ValueError("session closes must increase and match local session dates")
    return table, meta


def lagged_features(observations, *, sessions, columns, lags, decision_timing, metadata) -> FeatureResult:
    """Lag complete daily close observations by supplied trading-session counts.

    Positive lags only. Null warm-up and delayed availability stay visible; no fill
    or older-observation fallback. Source publication times must be supplied.
    """
    calendar, meta = _calendar(sessions, metadata)
    if decision_timing != "session_close":
        raise ValueError("decision_timing must explicitly be session_close")
    if not isinstance(columns, (list, tuple)) or not columns or len(set(columns)) != len(columns) or any(
            not isinstance(c, str) or not c.strip() or c in {*KEYS, "available_at", "decision_at", "target", "label_end"} for c in columns):
        raise ValueError("columns must be unique nonblank feature names, distinct from reserved fields")
    if not isinstance(lags, (list, tuple)) or not lags or any(type(n) is not int or n < 1 for n in lags) or len(set(lags)) != len(lags):
        raise ValueError("lags must be unique positive integer session counts")
    units = meta.get("feature_units")
    if not isinstance(units, dict) or set(units) != set(columns) or any(not isinstance(v, str) or not v.strip() for v in units.values()):
        raise ValueError("metadata.feature_units must declare every input column's units")
    columns, lags = sorted(columns), sorted(lags)
    schema = {**KEYS, "available_at": UTC, **{c: pl.Float64 for c in columns}}
    table = _table(observations, schema, "observations", list(KEYS), nonempty=True).sort("session", "asset")
    days, assets = calendar["session"].to_list(), sorted(table["asset"].unique())
    closes = dict(calendar.iter_rows())
    if set(table.select(*KEYS).iter_rows()) != {(d, a) for d in days for a in assets}:
        raise ValueError("observations must cover every supplied session/asset exactly")
    lookup = {(r["session"], r["asset"]): r for r in table.iter_rows(named=True)}
    if any(r["available_at"] < closes[r["session"]] for r in lookup.values()):
        raise ValueError("daily close observations cannot be available before their source close")
    feature_names = [f"{c}__lag{lag}" for c in columns for lag in lags]
    values, lineage, diagnostics = [], [], []
    for i, day in enumerate(days):
        for asset in assets:
            row = {"session": day, "asset": asset, "decision_at": closes[day]}
            available, missing = [], 0
            for c in columns:
                for lag in lags:
                    feature = f"{c}__lag{lag}"
                    source = lookup[days[i-lag], asset] if i >= lag else None
                    row[feature] = source[c] if source else None
                    missing += source is None
                    if source:
                        available.append(source["available_at"])
                    lineage.append((day, asset, feature, source["session"] if source else None,
                                    closes[source["session"]] if source else None, source["available_at"] if source else None))
            row["available_at"] = max(available) if not missing else None
            status = "warmup" if missing else "unavailable" if row["available_at"] > closes[day] else "ready"
            values.append(row)
            diagnostics.append((day, asset, status, missing))
    frame = pl.DataFrame(values, schema={**KEYS, "decision_at": UTC, "available_at": UTC,
                                        **{c: pl.Float64 for c in feature_names}})
    lineage = pl.DataFrame(lineage, schema={**KEYS, "feature": pl.String, "source_session": pl.Date,
        "observed_at": UTC, "available_at": UTC}, orient="row")
    diagnostics = pl.DataFrame(diagnostics, schema={**KEYS, "status": pl.String, "n_missing": pl.Int64}, orient="row")
    meta.update(feature_columns=feature_names, lags=lags, input_columns=columns,
        input_id=_identity({"observations": table, "sessions": calendar}, meta),
        decision_timing=decision_timing, lag_unit="supplied_trading_sessions",
        observation_timing="source_session_close", missing_policy="retain_warmup_and_unavailable_no_fill")
    tables = dict(values=frame, availability=lineage, diagnostics=diagnostics, sessions=calendar)
    return FeatureResult(**tables, metadata=meta, snapshot_id=_identity(tables, meta))


def _checked_features(result):
    if not isinstance(result, FeatureResult):
        raise ValueError("expected FeatureResult from lagged_features")
    tables = {k: getattr(result, k) for k in ("values", "availability", "diagnostics", "sessions")}
    if _identity(tables, result.metadata) != result.snapshot_id:
        raise ValueError("FeatureResult changed after preparation")
    return result


def _split_result(tables, metadata):
    return ResearchSplit(**tables, metadata=metadata, snapshot_id=_identity(tables, metadata))


def _checked_split(result):
    if not isinstance(result, ResearchSplit):
        raise ValueError("expected ResearchSplit from chronological_split")
    if _identity({k: getattr(result, k) for k in SPLIT_TABLES}, result.metadata) != result.snapshot_id:
        raise ValueError("ResearchSplit changed after preparation")
    return result


def chronological_split(features, labels, *, train_end, validation_end, test_end,
                        label_metadata, gap_sessions=0) -> ResearchSplit:
    """Partition by decision date, purging labels unavailable by each partition end.

    End dates are inclusive outcome/availability cutoffs. A gap excludes the first
    N supplied decision sessions after train/validation boundaries. Every excluded
    sample is returned with a reason. No random splitting or inferred horizons.
    """
    features = _checked_features(features)
    label_meta = _json_metadata(label_metadata, ("source", "unit", "definition"))
    days = features.sessions["session"].to_list()
    if any(type(d) is not date or d not in days for d in (train_end, validation_end, test_end)) or not train_end < validation_end < test_end:
        raise ValueError("train_end < validation_end < test_end must be supplied sessions")
    if type(gap_sessions) is not int or gap_sessions < 0:
        raise ValueError("gap_sessions must be a nonnegative integer")
    table = _table(labels, LABEL_SCHEMA, "labels", list(KEYS), nonempty=True).sort("session", "asset")
    selected = features.values.filter(pl.col("session") <= test_end)
    if set(table.select(*KEYS).iter_rows()) != set(selected.select(*KEYS).iter_rows()):
        raise ValueError("labels must match every feature key through test_end exactly, including warmup")
    closes = dict(features.sessions.iter_rows())
    for r in table.iter_rows(named=True):
        if r["label_end"] not in closes or r["label_end"] <= r["session"]:
            raise ValueError("label_end must be a later supplied session")
        if r["available_at"] < closes[r["label_end"]]:
            raise ValueError("label availability cannot precede its outcome close")
    joined = selected.join(table.rename({"available_at": "label_available_at"}), on=list(KEYS), how="left")
    joined = joined.join(features.diagnostics.select(*KEYS, pl.col("status").alias("feature_status")), on=list(KEYS), how="left").sort("session", "asset")
    ends = {"train": train_end, "validation": validation_end, "test": test_end}
    starts = {"train": days[0], "validation": days[days.index(train_end)+1], "test": days[days.index(validation_end)+1]}
    gaps = {"train": set(), "validation": set(days[days.index(train_end)+1:days.index(train_end)+1+gap_sessions]),
            "test": set(days[days.index(validation_end)+1:days.index(validation_end)+1+gap_sessions])}
    assignments, reasons = [], []
    for r in joined.iter_rows(named=True):
        part = "train" if r["session"] <= train_end else "validation" if r["session"] <= validation_end else "test"
        reason = (r["feature_status"] if r["feature_status"] != "ready" else
                  "boundary_gap" if r["session"] in gaps[part] else
                  "label_horizon_crosses_boundary" if r["label_end"] > ends[part] else
                  "label_unavailable_at_boundary" if r["label_available_at"] > closes[ends[part]] else "included")
        assignments.append(part)
        reasons.append(reason)
    joined = joined.with_columns(pl.Series("partition", assignments), pl.Series("reason", reasons)).drop("feature_status")
    excluded = joined.filter(pl.col("reason") != "included")
    tables = {p: joined.filter((pl.col("partition") == p) & (pl.col("reason") == "included")).drop("partition", "reason") for p in ends}
    if any(t.is_empty() for t in tables.values()):
        raise ValueError("each partition must retain samples after warmup, gap and horizon/availability purging")
    meta = dict(feature_columns=features.metadata["feature_columns"], feature_metadata=deepcopy(features.metadata),
        feature_snapshot_id=features.snapshot_id, label_metadata=label_meta,
        boundaries={p: {"start": starts[p].isoformat(), "end": ends[p].isoformat(), "cutoff_at": closes[ends[p]].isoformat()} for p in ends},
        gap_sessions=gap_sessions, gap_unit="supplied_trading_sessions", purge_policy="label_end_and_availability_at_or_before_partition_end",
        counts={p: t.height for p, t in tables.items()}, excluded_count=excluded.height, transform=None)
    tables.update(excluded=excluded, transforms=pl.DataFrame(schema=TRANSFORM_SCHEMA))
    return _split_result(tables, meta)


def standardize(split, *, zero_variance="raise") -> ResearchSplit:
    """Fit pooled feature mean/sample std on training rows only; transform all parts.

    No labels or validation/test values enter fitted parameters. Constant training
    features raise unless unit_scale explicitly requests mean subtraction only.
    """
    split = _checked_split(split)
    if zero_variance not in {"raise", "unit_scale"}:
        raise ValueError("zero_variance must be raise or unit_scale")
    if split.metadata["transform"] is not None:
        raise ValueError("split already standardized; use the original split")
    if split.train.height < 2:
        raise ValueError("standardization requires at least two training rows")
    rows = []
    for feature in split.metadata["feature_columns"]:
        values = split.train[feature].to_list()
        center, scale = mean(values), stdev(values)
        if not math.isfinite(center) or not math.isfinite(scale):
            raise ValueError("training transform is not representable")
        if scale == 0 and zero_variance == "raise":
            raise ValueError(f"zero training variance for {feature}; explicitly choose unit_scale if intended")
        rows.append((feature, center, scale if scale else 1., len(values), "ok" if scale else "zero_variance_unit_scale"))
    parameters = pl.DataFrame(rows, schema=TRANSFORM_SCHEMA, orient="row")
    tables = {k: getattr(split, k).clone() for k in SPLIT_TABLES}
    for part in ("train", "validation", "test"):
        tables[part] = tables[part].with_columns(*[((pl.col(f)-m)/s).alias(f) for f, m, s, _, _ in rows])
        if any(not tables[part][f].is_finite().all() for f in split.metadata["feature_columns"]):
            raise ValueError("standardized feature is not representable")
    # Excluded samples stay raw and cannot be used as fitted observations.
    tables["transforms"] = parameters
    meta = deepcopy(split.metadata)
    meta.update(parent_split_id=split.snapshot_id, transform=dict(method="standardize", fit_partition="train",
        pooling="all_training_asset_rows", ddof=1, zero_variance=zero_variance, excluded_values="original_untransformed"))
    return _split_result(tables, meta)
