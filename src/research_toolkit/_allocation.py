"""Trailing-window allocation and covariance estimates with explicit information cutoffs."""
from copy import deepcopy
from datetime import date
import math
from statistics import mean, stdev

import polars as pl

from ._metrics import _validated_run, _pair
from ._risk_free import _risk_free, _aligned_returns
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
                               periods_per_year, max_asset_weight, cap_policy="raise") -> AllocationResult:
    """Normalize inverse sample volatility from a strictly pre-decision window.

    Flat/insufficient samples raise. Caps reject by default; redistribution is opt-in.
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
    original = weights.copy()
    if cap_policy not in {"raise", "redistribute"}:
        raise ValueError("cap_policy must be raise or redistribute")
    cap = _number(max_asset_weight, "max_asset_weight", positive=True)
    if cap > 1:
        raise ValueError("max_asset_weight must be in (0, 1]")
    if cap_policy == "redistribute":
        if len(weights)*cap < 1:
            raise ValueError("infeasible concentration cap: asset_count * cap < 1")
        # Water filling preserves inverse-volatility proportions among uncapped assets.
        free, weights = set(inv), {}
        while free:
            remaining = 1-math.fsum(weights.values())
            total = math.fsum(inv[a] for a in free)
            if total <= 0:
                raise ValueError("inverse-volatility proportions are not representable")
            proposed = {a: remaining*inv[a]/total for a in sorted(free)}
            bound = [a for a in proposed if proposed[a] > cap]
            if not bound:
                weights.update(proposed)
                break
            weights.update({a: cap for a in bound})
            free.difference_update(bound)
        weights = dict(sorted(weights.items()))
    _limit(weights, max_asset_weight)
    if not math.isclose(math.fsum(weights.values()), 1., rel_tol=0, abs_tol=1e-12):
        raise ArithmeticError("capped allocation weights do not sum to one")
    table = pl.DataFrame({"asset": list(weights), "weight": list(weights.values())})
    estimates = pl.DataFrame({"asset": list(vols), "annualized_volatility": list(vols.values()),
                             "n_obs": [lookback]*len(vols)})
    meta.update(method="inverse_volatility", max_asset_weight=float(max_asset_weight),
                cap_policy=cap_policy,
                weight_denominator="risky_asset_notional", volatility_unit="fraction/sqrt(year)")
    diagnostics = pl.DataFrame({"asset": list(weights), "uncapped_weight": [original[a] for a in weights],
        "weight_change": [weights[a]-original[a] for a in weights],
        "at_cap": [math.isclose(weights[a], cap, rel_tol=0, abs_tol=1e-12) for a in weights]})
    return AllocationResult(table, estimates, meta, diagnostics)


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


def rolling_risk(result, *, window, periods_per_year, risk_free_annual_effective=None,
                 allow_partial=False, risk_free_returns=None, risk_free_metadata=None,
                 sharpe_denominator="portfolio_returns", benchmark=None,
                 benchmark_metadata=None) -> RollingRiskResult:
    """Realized net-return trailing risk through each close, for reporting only.

    Full windows are required. Early rows retain nulls with sample counts/status;
    these contemporaneous diagnostics must not be used for same-close decisions.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    daily, _, meta = _validated_run(result, allow_partial)
    a = _number(periods_per_year, "periods_per_year", positive=True)
    rf = risk_free_annual_effective
    rf_table, rf_meta, periodic = _risk_free(daily, meta["currency"], rf, risk_free_returns, risk_free_metadata, a)
    if sharpe_denominator not in {"portfolio_returns", "excess_returns"}:
        raise ValueError("sharpe_denominator must be portfolio_returns or excess_returns")
    br, bm_meta = None, None
    if benchmark is not None:
        b, bm_meta = _aligned_returns(benchmark, daily.select("period_start", "session"), benchmark_metadata, meta["currency"], "benchmark")
        br = b["simple_return"].to_list()
    elif benchmark_metadata is not None:
        raise ValueError("benchmark_metadata requires a benchmark")
    returns = daily["simple_return"].to_list()
    rf_values = rf_table["simple_return"].to_list()
    rows = []
    for i, day in enumerate(daily["session"]):
        start = max(0, i-window+1)
        values = returns[start:i+1]
        vol = stdev(values) if len(values) == window else None
        excess = [v-f for v, f in zip(values, rf_values[start:i+1])]
        sv = stdev(excess) if len(values) == window and sharpe_denominator == "excess_returns" else vol
        sharpe = math.sqrt(a)*mean(excess)/sv if sv else None
        annual = vol*math.sqrt(a) if vol is not None else None
        if any(v is not None and not math.isfinite(v) for v in (annual, sharpe)):
            raise ValueError("rolling risk is not representable")
        corr, beta, cs, bs = (None, None, "no_benchmark", "no_benchmark")
        if br is not None:
            corr, beta, cs, bs = _pair(values, br[start:i+1]) if len(values) == window else (None, None, "insufficient_samples", "insufficient_samples")
        rows.append((day, daily["period_start"][start], len(values), annual, sharpe,
            "insufficient_samples" if vol is None else "ok",
            "insufficient_samples" if sv is None else ("zero_excess_volatility" if sharpe_denominator == "excess_returns" else "zero_volatility") if sv == 0 else "ok",
            beta, corr, bs, cs))
    table = pl.DataFrame(rows, schema={"session": pl.Date, "period_start": pl.Date,
        "n_obs": pl.Int64, "annualized_volatility": pl.Float64, "sharpe": pl.Float64,
        "volatility_status": pl.String, "sharpe_status": pl.String, "beta": pl.Float64, "correlation": pl.Float64,
        "beta_status": pl.String, "correlation_status": pl.String}, orient="row")
    meta.update(window=window, periods_per_year=a, risk_free_annual_effective=rf,
        risk_free_periodic=periodic, ddof=1, information_cutoff="through_current_close_reporting_only",
        risk_free=rf_meta, sharpe_denominator=sharpe_denominator, benchmark=bm_meta,
        annualization="sqrt_periods_per_year_no_serial_correlation_adjustment", allow_partial=allow_partial)
    return RollingRiskResult(table, meta)
