"""Explicit long-only signal timing; maps instructions, never computes positions."""
from copy import deepcopy

import polars as pl

from ._data import _table, _identity
from ._portfolio import _weights, _number
from ._rebalancing import TARGET_SCHEMA
from ._research import _calendar, UTC
from ._results import SignalResult

SIGNAL_SCHEMA = {"decision_session": pl.Date, "asset": pl.String, "weight": pl.Float64,
                 "gross_leverage": pl.Float64, "observed_at": UTC, "available_at": UTC}
SIGNAL_AUDIT_SCHEMA = {"decision_session": pl.Date, "session": pl.Date, "asset": pl.String,
    "observed_at": UTC, "available_at": UTC, "decision_at": UTC, "order_at": UTC,
    "execution_at": UTC, "weight": pl.Float64, "gross_leverage": pl.Float64, "action": pl.String}


def signal_targets(signals, *, sessions, execution, rebalance, metadata) -> SignalResult:
    """Map dated risky proportions/leverage to the next supplied session close.

    Choose each_signal to reset targets at every signal or on_change to leave
    quantities untouched until the instruction changes. No implicit binary-score,
    probability-to-weight, negative-position or next-open interpretation.
    """
    calendar, meta = _calendar(sessions, metadata)
    if not isinstance(meta.get("signal_definition"), str) or not meta["signal_definition"].strip():
        raise ValueError("metadata.signal_definition must describe how instructions were produced")
    if execution != "next_session_close" or rebalance not in {"each_signal", "on_change"}:
        raise ValueError("require execution='next_session_close' and rebalance='each_signal' or 'on_change'")
    table = _table(signals, SIGNAL_SCHEMA, "signals", ["decision_session", "asset"], nonempty=True).sort("decision_session", "asset")
    days, closes = calendar["session"].to_list(), dict(calendar.iter_rows())
    successors = dict(zip(days, days[1:]))
    assets = sorted(table["asset"].unique())
    targets, audit, previous = [], [], None
    for day in table["decision_session"].unique().sort():
        if day not in successors:
            raise ValueError("every signal needs a decision session and a later execution session in the supplied calendar")
        basket = table.filter(pl.col("decision_session") == day)
        if basket["asset"].to_list() != assets or basket["gross_leverage"].n_unique() != 1:
            raise ValueError("each signal basket must contain every asset and one gross_leverage")
        weights = _weights(basket.select("asset", "weight"), assets)
        leverage = _number(basket["gross_leverage"][0], "gross_leverage")
        # Preserve the exact instruction comparison, not a strategy turnover band.
        instruction = (tuple(weights.items()), leverage)
        action = "target" if rebalance == "each_signal" or instruction != previous else "unchanged_target"
        execution_day = successors[day]
        for r in basket.iter_rows(named=True):
            if not r["observed_at"] <= r["available_at"] <= closes[day]:
                raise ValueError("signal observation/availability must precede its decision close")
            if action == "target":
                targets.append((day, execution_day, r["asset"], r["weight"], leverage))
            audit.append((day, execution_day, r["asset"], r["observed_at"], r["available_at"],
                          closes[day], closes[day], closes[execution_day], r["weight"], leverage, action))
        previous = instruction
    meta.update(execution=execution, rebalance=rebalance, decision_timing="session_close",
        order_timing="decision_close_instruction", weight_denominator="risky_asset_notional",
        leverage_denominator="post_cost_equity", unchanged_comparison="exact_supplied_weights_and_leverage",
        missing_signal_policy="no_new_target_no_fill", signal_provenance="caller_asserted_information_set")
    tables = dict(signals=table, targets=pl.DataFrame(targets, schema=TARGET_SCHEMA, orient="row"),
                  audit=pl.DataFrame(audit, schema=SIGNAL_AUDIT_SCHEMA, orient="row"), sessions=calendar)
    return SignalResult(**tables, metadata=meta, snapshot_id=_identity(tables, meta))


def _checked_signals(result, market, entry, end):
    tables = {k: getattr(result, k) for k in ("signals", "targets", "audit", "sessions")}
    if _identity(tables, result.metadata) != result.snapshot_id:
        raise ValueError("SignalResult changed after preparation")
    rebuilt = signal_targets(result.signals, sessions=result.sessions, execution=result.metadata["execution"],
                             rebalance=result.metadata["rebalance"], metadata=result.metadata)
    if rebuilt.snapshot_id != result.snapshot_id:
        raise ValueError("signal targets disagree with their source instructions")
    if not result.sessions.equals(market.sessions):
        raise ValueError("signal calendar must exactly match the market calendar")
    if any(not entry <= d < end for d in result.audit["session"]):
        raise ValueError("all signal execution dates must lie in [entry_session, end_session); terminal is mark-only")
    return rebuilt


def _attach_signals(run, signals):
    from dataclasses import replace
    processed = {run.metadata["entry_session"]}
    processed.update(d.isoformat() for d in run.rebalances["session"])
    statuses = []
    for r in signals.audit.iter_rows(named=True):
        if run.stop_session is not None and r["session"] > run.stop_session:
            status = "not_reached"
        elif r["action"] == "unchanged_target":
            status = "unchanged_no_target"
        elif r["session"].isoformat() in processed:
            status = "processed"  # An unchanged quantity can still produce no fill.
        else:
            status = "blocked_at_stop"
        statuses.append(status)
    meta = deepcopy(run.metadata)
    meta["signals"] = {**deepcopy(signals.metadata), "snapshot_id": signals.snapshot_id}
    return replace(run, signal_audit=signals.audit.with_columns(pl.Series("status", statuses)), metadata=meta)
