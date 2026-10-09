"""Price return arithmetic with explicit basis and interval identities."""

from copy import deepcopy
import math

import polars as pl

from ._data import DIAGNOSTIC_SCHEMA, _validated_market
from ._results import ReturnResult


def returns(market, *, method: str, basis: str, dividend_policy: str | None = None) -> ReturnResult:
    """Compute simple or log returns from a validated, complete price panel.

    ``basis='price'`` accepts raw or split-adjusted prices. Raw returns retain split
    jumps; this is recorded in diagnostics. ``basis='total_return'`` accepts a
    supplied total-return-adjusted series, or raw inputs with the explicit
    dividend_policy='reinvest_ex_close' analytical convention. That convention
    reinvests ex-date entitlement before actual payment; it is not executable cash
    accounting. Use the ledger for dividends held as receivables/cash.
    The first observation per asset remains null, never a synthetic zero return.
    """
    market = _validated_market(market)
    if market.securities is not None and (market.corporate_actions.height or market.warrants.height
                                          or market.prices.height != market.sessions.height*market.securities.height):
        raise ValueError("lifecycle markets with corporate actions or partial quote panels need "
                         "rt.security_returns, which labels gaps and distributions explicitly")
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'")
    allowed = {"price": {"raw", "split_adjusted"},
               "total_return": {"total_return_adjusted"}}
    action_returns = basis == "total_return" and market.metadata["price_basis"] == "raw"
    if action_returns:
        if dividend_policy != "reinvest_ex_close":
            raise ValueError("incompatible raw total returns: require dividend_policy='reinvest_ex_close'")
    elif dividend_policy is not None:
        raise ValueError("dividend_policy is only supported for raw total returns")
    if not action_returns and (basis not in allowed or market.metadata["price_basis"] not in allowed[basis]):
        raise ValueError("basis is incompatible with the supplied price_basis")
    column = f"{method}_return"
    ratios = {(r["effective_session"], r["asset"]): r["ratio"] for r in market.splits.iter_rows(named=True)}
    dividends = {}
    for r in market.dividends.iter_rows(named=True):
        key = (r["ex_session"], r["asset"])
        dividends[key] = dividends.get(key, 0.) + r["cash_per_share"]
    rows = []
    previous = {}
    for row in market.prices.iter_rows(named=True):
        asset, session, price = row["asset"], row["session"], row["close"]
        prior = previous.get(asset)
        value = None
        if prior is not None:
            comparable = (price+dividends.get((session, asset), 0.))*ratios.get((session, asset), 1.) if action_returns else price
            change = (comparable - prior[1]) / prior[1]
            # log1p preserves small changes; log differences handle extreme ratios.
            value = (change if method == "simple" else
                     math.log1p(change) if math.isfinite(change) and change > -1
                     else math.log(comparable) - math.log(prior[1]))
            if not math.isfinite(value) or (method == "simple" and value <= -1):
                raise ValueError(f"return outside representable positive wealth at {session}/{asset}")
        rows.append({"session": session, "asset": asset,
                     "period_start": prior[0] if prior else None, column: value})
        previous[asset] = (session, price)
    diagnostics = [{"code": "structural_leading_null", "table": "returns",
                    "count": len(previous)}]
    if not action_returns and market.metadata["price_basis"] == "raw" and market.splits.height:
        diagnostics.append({"code": "raw_price_split_discontinuity", "table": "returns",
                            "count": market.splits.height})
    return ReturnResult(
        values=pl.DataFrame(rows, schema={"session": pl.Date, "asset": pl.String,
                                         "period_start": pl.Date, column: pl.Float64}),
        metadata={"method": method, "basis": basis, "frequency": "1d", "unit": "fraction"
                  if method == "simple" else "log_fraction", "snapshot_id": market.snapshot_id,
                  "source": deepcopy(market.metadata),
                  "dividend_policy": dividend_policy,
                  "sessions": [d.isoformat() for d in market.sessions["session"]],
                  "source_session_closes": {d.isoformat(): t.isoformat()
                                            for d, t in market.sessions.iter_rows()},
                  "assets": sorted(previous)},
        diagnostics=pl.DataFrame(diagnostics, schema=DIAGNOSTIC_SCHEMA),
    )


def cumulative_returns(result: ReturnResult, *, method: str) -> ReturnResult:
    """Add ``sum``, ``compound``, or ``wealth`` values without changing return kind.

    Summing simple returns is distinct from compounding. For log inputs, compound
    and wealth explicitly exponentiate the cumulative log return. Leading nulls
    remain null. Invalid or discontinuous return tables are rejected.
    """
    if not isinstance(result, ReturnResult):
        raise ValueError("expected ReturnResult from returns")
    kind = result.metadata.get("method")
    if method not in {"sum", "compound", "wealth"} or kind not in {"simple", "log"}:
        raise ValueError("unknown cumulative method or return kind")
    if result.metadata.get("frequency") != "1d":
        raise ValueError("return frequency must remain '1d'")
    col = f"{kind}_return"
    expected_schema = {"session": pl.Date, "asset": pl.String,
                       "period_start": pl.Date, col: pl.Float64}
    values = result.values
    if not isinstance(values, pl.DataFrame) or any(values.schema.get(k) != v
                                                  for k, v in expected_schema.items()):
        raise ValueError("return table has invalid schema")
    sessions, assets = result.metadata.get("sessions", []), result.metadata.get("assets", [])
    if not sessions or not assets or len(set(sessions)) != len(sessions) or len(set(assets)) != len(assets):
        raise ValueError("return metadata must contain unique sessions and assets")
    if sessions != sorted(sessions):
        raise ValueError("return metadata sessions must be chronological")
    actual = values.select("session", "asset")
    if actual.is_duplicated().any() or any(actual.null_count().row(0)):
        raise ValueError("return keys must be unique and nonnull")
    if {(d.isoformat(), a) for d, a in actual.iter_rows()} != {(d, a) for d in sessions for a in assets}:
        raise ValueError("return table does not match its declared session/asset coverage")
    values = values.sort("session", "asset")
    name = f"cumulative_{kind}_return" if method == "sum" else (
        "compounded_return" if method == "compound" else "wealth_multiple")
    output, accumulated, previous = [], {}, {}
    for row in values.iter_rows(named=True):
        asset, value, start = row["asset"], row[col], row["period_start"]
        if asset not in previous:
            if value is not None or start is not None:
                raise ValueError("first return and period_start must be structural nulls")
            accumulated[asset] = 0.0 if kind == "log" or method == "sum" else 1.0
            output.append(None)
        else:
            if start != previous[asset] or value is None or not math.isfinite(value):
                raise ValueError("return intervals must be contiguous, finite, and nonnull after the first row")
            if kind == "simple" and value <= -1:
                raise ValueError("standalone returns require positive wealth (simple return > -1)")
            accumulated[asset] = (accumulated[asset] + value
                                  if kind == "log" or method == "sum"
                                  else accumulated[asset] * (1 + value))
            try:
                cumulative = (accumulated[asset] if method == "sum" else
                              math.exp(accumulated[asset]) if kind == "log" else accumulated[asset])
            except OverflowError as exc:
                raise ValueError("cumulative wealth overflow") from exc
            if not math.isfinite(cumulative) or (method != "sum" and cumulative <= 0):
                raise ValueError("cumulative result is not representable")
            output.append(cumulative - 1 if method == "compound" else cumulative)
        previous[asset] = row["session"]
    metadata = deepcopy(result.metadata)
    metadata["cumulative_method"] = method
    metadata["cumulative_unit"] = ("multiple" if method == "wealth" else
                                   "log_fraction" if method == "sum" and kind == "log" else "fraction")
    return ReturnResult(values.with_columns(pl.Series(name, output, dtype=pl.Float64)),
                        metadata, result.diagnostics.clone())
