"""Trailing-window allocation and covariance estimates with explicit information cutoffs."""
from copy import deepcopy
from datetime import date
import math
from statistics import mean, stdev

import polars as pl

from ._metrics import _number as _metric_number, _validated_run
from ._portfolio import _number, _weights
from ._rebalancing import _limit
from ._results import AllocationResult, RiskResult, RollingRiskResult
from ._returns import cumulative_returns


def _window(result, decision_session, lookback, periods_per_year):
    if type(decision_session) is not date:
        raise ValueError("decision_session must be datetime.date")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    annualization = _number(periods_per_year, "periods_per_year", positive=True)
    validated = cumulative_returns(result, method="sum")
    meta = deepcopy(result.metadata)
    if meta.get("method") != "simple":
        raise ValueError("allocation/risk estimates require simple returns")
    if not isinstance(meta.get("basis"), str) or not meta["basis"].strip():
        raise ValueError("return basis must be explicit")
    if decision_session.isoformat() not in meta["sessions"]:
        raise ValueError("decision_session must be in the supplied return calendar")
    dates = validated.values.filter((pl.col("session") < decision_session) &
                                    pl.col("period_start").is_not_null())["session"].unique().sort().to_list()
    if len(dates) < lookback:
        raise ValueError("insufficient pre-decision return observations")
    dates = dates[-lookback:]
    table = validated.values.filter(pl.col("session").is_in(dates))
    assets = sorted(meta["assets"])
    series = {a: table.filter(pl.col("asset") == a)["simple_return"].to_list() for a in assets}
    meta.update(decision_session=decision_session.isoformat(), sample_start=dates[0].isoformat(),
        sample_end=dates[-1].isoformat(), sample_period_start=table["period_start"].min().isoformat(),
        lookback=lookback, n_obs=lookback, periods_per_year=annualization, ddof=1,
        information_cutoff="return_session_strictly_before_decision_session")
    return series, meta


def inverse_volatility_weights(result, *, decision_session, lookback,
                               periods_per_year, max_asset_weight) -> AllocationResult:
    """Normalize inverse sample volatility from a strictly pre-decision window.

    Flat/insufficient samples raise; concentration breaches raise without clipping.
    Result weights use risky-asset proportions, independently of portfolio leverage.
    """
    series, meta = _window(result, decision_session, lookback, periods_per_year)
    vols = {a: stdev(values)*math.sqrt(meta["periods_per_year"]) for a, values in series.items()}
    if any(not math.isfinite(v) or v <= 0 for v in vols.values()):
        raise ValueError("inverse volatility requires finite, strictly positive asset volatility")
    # Scaling by the smallest volatility prevents overflow in reciprocal weights.
    inv = {a: min(vols.values())/vols[a] for a in vols}
    total = math.fsum(inv.values())
    weights = {a: v/total for a, v in inv.items()}
    _limit(weights, max_asset_weight)
    table = pl.DataFrame({"asset": list(weights), "weight": list(weights.values())})
    estimates = pl.DataFrame({"asset": list(vols), "annualized_volatility": list(vols.values()),
                             "n_obs": [lookback]*len(vols)})
    meta.update(method="inverse_volatility", max_asset_weight=float(max_asset_weight),
                weight_denominator="risky_asset_notional", volatility_unit="fraction/sqrt(year)")
    return AllocationResult(table, estimates, meta)


def risk_contributions(result, *, weights, gross_leverage, decision_session,
                       lookback, periods_per_year) -> RiskResult:
    """Euler contributions to estimated annualized volatility (not realized P&L).

    The supplied risky proportions are scaled to net-equity exposures by explicit
    gross_leverage. Cash/financing are assumed deterministic in this covariance
    estimate. Concentration enforcement belongs to allocation/execution policy.
    """
    series, meta = _window(result, decision_session, lookback, periods_per_year)
    assets = sorted(series)
    checked = _weights(weights, assets)
    if set(checked) != set(assets):
        raise ValueError("risk weights must cover every return asset, including explicit zeros")
    leverage = _number(gross_leverage, "gross_leverage")
    total = math.fsum(checked.values())
    exposures = {a: checked[a]/total*leverage for a in assets}
    means = {a: mean(series[a]) for a in assets}
    centered = {a: [v-means[a] for v in series[a]] for a in assets}
    cov = {(a, b): math.fsum(x*y for x, y in zip(centered[a], centered[b]))/(lookback-1)*meta["periods_per_year"]
           for a in assets for b in assets}
    # Direct portfolio observations give a nonnegative variance even when nearly
    # cancelling covariance terms would produce a small negative roundoff value.
    portfolio = [math.fsum(exposures[a]*series[a][i] for a in assets) for i in range(lookback)]
    volatility = stdev(portfolio)*math.sqrt(meta["periods_per_year"])
    if not math.isfinite(volatility) or any(not math.isfinite(x) for x in cov.values()):
        raise ValueError("covariance estimate is not representable")
    rows = []
    for a in assets:
        marginal = math.fsum(cov[a, b]*exposures[b] for b in assets)/volatility if volatility else None
        contribution = exposures[a]*marginal if marginal is not None else None
        fraction = contribution/volatility if volatility else None
        if any(v is not None and not math.isfinite(v) for v in (marginal, contribution, fraction)):
            raise ValueError("risk contribution is not representable")
        rows.append((a, exposures[a], marginal, contribution,
                     fraction, lookback,
                     "ok" if volatility else "zero_portfolio_volatility"))
    values = pl.DataFrame(rows, schema={"asset": pl.String, "equity_weight": pl.Float64,
        "marginal_volatility": pl.Float64, "volatility_contribution": pl.Float64,
        "fraction_of_total": pl.Float64, "n_obs": pl.Int64, "status": pl.String}, orient="row")
    if volatility and not math.isclose(values["volatility_contribution"].sum(), volatility, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("covariance contributions do not reconcile at available precision")
    covariance = pl.DataFrame([(a, b, cov[a, b]) for a in assets for b in assets],
        schema={"asset": pl.String, "other_asset": pl.String, "annualized_covariance": pl.Float64}, orient="row")
    meta.update(portfolio_volatility=volatility, gross_leverage=leverage,
                unit="fraction/sqrt(year)", covariance_unit="fraction_squared/year",
                assumption="fixed_equity_exposures_deterministic_cash_and_financing")
    return RiskResult(values, covariance, meta)


def rolling_risk(result, *, window, periods_per_year, risk_free_annual_effective,
                 allow_partial=False) -> RollingRiskResult:
    """Realized net-return trailing risk through each close, for reporting only.

    Full windows are required. Early rows retain nulls with sample counts/status;
    these contemporaneous diagnostics must not be used for same-close decisions.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    daily, _, meta = _validated_run(result, allow_partial)
    a = _number(periods_per_year, "periods_per_year", positive=True)
    rf = _metric_number(risk_free_annual_effective, "risk_free_annual_effective", lower=-1)
    try:
        periodic = math.expm1(math.log1p(rf)/a)
    except OverflowError as exc:
        raise ValueError("periodic risk-free rate is not representable") from exc
    returns = daily["simple_return"].to_list()
    rows = []
    for i, day in enumerate(daily["session"]):
        start = max(0, i-window+1)
        values = returns[start:i+1]
        vol = stdev(values) if len(values) == window else None
        sharpe = math.sqrt(a)*mean(v-periodic for v in values)/vol if vol else None
        annual = vol*math.sqrt(a) if vol is not None else None
        if any(v is not None and not math.isfinite(v) for v in (annual, sharpe)):
            raise ValueError("rolling risk is not representable")
        rows.append((day, daily["period_start"][start], len(values), annual, sharpe,
            "insufficient_samples" if vol is None else "ok",
            "insufficient_samples" if vol is None else "zero_volatility" if vol == 0 else "ok"))
    table = pl.DataFrame(rows, schema={"session": pl.Date, "period_start": pl.Date,
        "n_obs": pl.Int64, "annualized_volatility": pl.Float64, "sharpe": pl.Float64,
        "volatility_status": pl.String, "sharpe_status": pl.String}, orient="row")
    meta.update(window=window, periods_per_year=a, risk_free_annual_effective=rf,
        risk_free_periodic=periodic, ddof=1, information_cutoff="through_current_close_reporting_only",
        annualization="sqrt_periods_per_year_no_serial_correlation_adjustment", allow_partial=allow_partial)
    return RollingRiskResult(table, meta)
