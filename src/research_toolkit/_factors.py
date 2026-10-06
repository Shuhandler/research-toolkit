"""Descriptive OLS of portfolio returns on supplied factor returns, with strict alignment."""

from copy import deepcopy
import json
import math
import sys

import polars as pl

from ._data import _table
from ._metrics import METRIC_SCHEMA, _pair
from ._portfolio import _finite_above
from ._results import FactorRegressionResult, PerformanceResult, RiskFreeResult
from ._risk_free import INTERVAL_SCHEMA, RETURN_SCHEMA, _aligned_returns, _intervals

FACTOR_BASES = {"total_return", "excess_return", "long_short"}
PORTFOLIO_BASES = {"total_return", "excess_return"}
RESERVED = {"intercept", "dependent", "period_start", "session"}
IDIO_NOTE = ("sqrt(periods_per_year*SSE/(n-k-1)) with k factors plus an intercept: the degrees-of-freedom-"
             "adjusted residual standard error, annualized by square root. It is not the sample standard "
             "deviation of the fitted residuals (ddof=1), which would divide SSE by n-1.")


def _json(value, name):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON") from exc


def _metadata(metadata, name, keys=("source", "currency", "frequency")):
    if not isinstance(metadata, dict) or any(not isinstance(metadata.get(k), str) or not metadata[k].strip() for k in keys):
        raise ValueError(f"{name} requires nonblank {', '.join(keys)}")
    return _json(metadata, name)


def _portfolio(portfolio, portfolio_basis, portfolio_metadata, allow_partial):
    if isinstance(portfolio, PerformanceResult):
        meta = portfolio.metadata
        if portfolio_metadata is not None:
            raise ValueError("a PerformanceResult carries its own metadata; omit portfolio_metadata")
        if portfolio_basis not in (None, "total_return"):
            raise ValueError("PerformanceResult returns are total net equity returns; portfolio_basis must be total_return")
        if meta.get("status") not in {"complete", "stopped"} or (meta["status"] != "complete" and not allow_partial):
            raise ValueError("a stopped or unknown run requires explicit allow_partial=True")
        if meta.get("return_basis") != "net_equity":
            raise ValueError("PerformanceResult must report net_equity returns")
        values = _table(portfolio.daily.select(list(RETURN_SCHEMA)), RETURN_SCHEMA, "portfolio", ["session"], nonempty=True)
        provenance = {"source": "PerformanceResult", "currency": meta.get("currency"), "frequency": meta.get("frequency"),
                      "return_basis": "net_equity", "run_status": meta["status"], "stop_reason": meta.get("stop_reason")}
        return values.sort("session"), "total_return", provenance
    if portfolio_basis not in PORTFOLIO_BASES:
        raise ValueError("portfolio_basis must be total_return or excess_return for a supplied return table")
    provenance = _metadata(portfolio_metadata, "portfolio_metadata")
    values = _table(portfolio, RETURN_SCHEMA, "portfolio", ["session"], nonempty=True).sort("session")
    return values, portfolio_basis, provenance


def _risk_free_series(risk_free, risk_free_metadata, portfolio, intervals, currency, frequency):
    if isinstance(risk_free, str):
        if risk_free != "report" or not isinstance(portfolio, PerformanceResult):
            raise ValueError("risk_free='report' requires a PerformanceResult portfolio")
        if risk_free_metadata is not None:
            raise ValueError("risk_free='report' uses the report's recorded risk-free convention")
        table = _table(portfolio.risk_free_returns.select(list(RETURN_SCHEMA)), RETURN_SCHEMA, "risk_free", ["session"],
                       nonempty=True).sort("session")
        if not table.select("period_start", "session").equals(intervals):
            raise ValueError("report risk-free intervals must match the regression intervals exactly")
        meta = portfolio.metadata
        return table, {"source": "PerformanceResult.risk_free_returns", "metadata": deepcopy(meta.get("risk_free")),
                       "annual_effective": meta.get("risk_free_annual_effective"),
                       "periodic": meta.get("risk_free_periodic")}
    if isinstance(risk_free, RiskFreeResult):
        if risk_free_metadata is not None:
            raise ValueError("RiskFreeResult already includes risk_free_metadata")
        risk_free, risk_free_metadata = risk_free.values, risk_free.metadata
    table, meta = _aligned_returns(risk_free, intervals, risk_free_metadata, currency, "risk_free", frequency)
    return table, {"source": "supplied", "metadata": meta}


def _householder(columns, y):
    """Column-pivoted Householder QR; returns pivot order, R diagonal/upper, and Q'y."""
    a = [list(c) for c in columns]
    qty = list(y)
    n, p = len(y), len(a)
    order = list(range(p))
    r = [[0.0]*p for _ in range(p)]
    for j in range(min(n, p)):
        norms = [math.sqrt(math.fsum(v*v for v in a[c][j:])) for c in range(j, p)]
        pivot = j + max(range(len(norms)), key=lambda i: norms[i])
        a[j], a[pivot] = a[pivot], a[j]
        order[j], order[pivot] = order[pivot], order[j]
        for row in range(j):
            r[row][j], r[row][pivot] = r[row][pivot], r[row][j]
        x = a[j][j:]
        alpha = math.sqrt(math.fsum(v*v for v in x))
        if alpha == 0:
            break
        alpha = -alpha if x[0] > 0 else alpha
        v = [x[0] - alpha] + x[1:]
        vv = math.fsum(t*t for t in v)
        r[j][j] = alpha
        if vv == 0:
            continue
        for c in range(j+1, p):
            s = 2*math.fsum(vi*ai for vi, ai in zip(v, a[c][j:]))/vv
            a[c][j:] = [ai - s*vi for ai, vi in zip(a[c][j:], v)]
            r[j][c] = a[c][j]
        s = 2*math.fsum(vi*yi for vi, yi in zip(v, qty[j:]))/vv
        qty[j:] = [yi - s*vi for yi, vi in zip(qty[j:], v)]
    return order, r, qty


def _ols(y, x, names):
    """OLS with intercept on centered, unit-norm factors. None coefficients when unidentified."""
    n, k = len(y), len(x)
    means = [math.fsum(c)/n for c in x]
    ybar = math.fsum(y)/n
    constant = [len(set(c)) == 1 for c in x]
    scaled, norms = [], []
    for c, m, flat in zip(x, means, constant):
        d = [0.0]*n if flat else [v - m for v in c]
        norm = math.sqrt(math.fsum(v*v for v in d))
        norms.append(norm)
        scaled.append([v/norm for v in d] if norm else d)
    yc = [v - ybar for v in y]
    order, r, qty = _householder(scaled, yc)
    largest = abs(r[0][0]) if k and n else 0.0
    tolerance = max(n, k)*sys.float_info.epsilon*largest
    rank_factors = sum(1 for j in range(min(n, k)) if abs(r[j][j]) > tolerance)
    info = {"rank": 1 + rank_factors, "tolerance": tolerance,
            "constant_factors": [name for name, flat in zip(names, constant) if flat]}
    if n < k + 1:
        return None, info, "insufficient_observations"
    if rank_factors < k:
        return None, info, "rank_deficient"
    b = [0.0]*k
    for j in reversed(range(k)):
        b[j] = (qty[j] - math.fsum(r[j][c]*b[c] for c in range(j+1, k)))/r[j][j]
    beta = [0.0]*k
    for j, original in enumerate(order):
        beta[original] = b[j]/norms[original]
    alpha = ybar - math.fsum(bj*m for bj, m in zip(beta, means))
    fitted = [alpha + math.fsum(bj*c[i] for bj, c in zip(beta, x)) for i in range(n)]
    return (alpha, beta, fitted), info, "ok"


def factor_regression(portfolio, factors, *, factor_bases, factor_metadata, periods_per_year, model,
                      risk_free=None, risk_free_metadata=None, portfolio_basis=None,
                      portfolio_metadata=None, allow_partial=False) -> FactorRegressionResult:
    """Joint OLS with intercept, plus standalone single-factor fits and correlations.

    ``portfolio`` is a PerformanceResult (net equity total returns) or a
    ``period_start, session, simple_return`` table with ``portfolio_basis`` and
    ``portfolio_metadata``. ``factors`` is a wide table of ``period_start``,
    ``session`` and one Float64 column per named factor; ``factor_bases`` declares
    each as ``total_return``, ``excess_return`` or ``long_short``. With
    ``model='excess_returns'`` only total-return series subtract the matched
    risk-free return; ``model='raw_returns'`` uses every series as supplied.
    Intervals must match on both endpoints. Results are descriptive in-sample fits.
    """
    if model not in {"raw_returns", "excess_returns"}:
        raise ValueError("model must be 'raw_returns' or 'excess_returns'")
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be boolean")
    a = _finite_above(periods_per_year, "periods_per_year", lower=0)
    y_table, y_basis, y_meta = _portfolio(portfolio, portfolio_basis, portfolio_metadata, allow_partial)
    intervals = _intervals(y_table.select("period_start", "session"))
    if y_basis == "total_return" and (y_table["simple_return"] <= -1).any():
        raise ValueError("portfolio total returns must exceed -1")
    currency, frequency = y_meta.get("currency"), y_meta.get("frequency")

    if not isinstance(factors, pl.DataFrame):
        raise ValueError("factors must be a Polars DataFrame")
    names = [c for c in factors.columns if c not in INTERVAL_SCHEMA]
    if set(INTERVAL_SCHEMA) - set(factors.columns) or not names:
        raise ValueError("factors needs period_start, session and at least one named factor column")
    if RESERVED & set(names) or any(not isinstance(nm, str) or not nm.strip() for nm in names):
        raise ValueError(f"factor names must be nonblank and not {sorted(RESERVED)}")
    if not isinstance(factor_bases, dict) or set(factor_bases) != set(names):
        raise ValueError(f"factor_bases must declare exactly the factor columns {names}")
    if any(v not in FACTOR_BASES for v in factor_bases.values()):
        raise ValueError(f"factor bases must be one of {sorted(FACTOR_BASES)}")
    f_meta = _metadata(factor_metadata, "factor_metadata")
    if f_meta["currency"] != currency or f_meta["frequency"] != frequency:
        raise ValueError("factor currency/frequency must match the portfolio")
    schema = {**INTERVAL_SCHEMA, **{nm: pl.Float64 for nm in names}}
    f_table = _table(factors.select(list(schema)), schema, "factors", ["session"], nonempty=True).sort("session")
    if f_table.select("period_start").is_duplicated().any():
        raise ValueError("factors has duplicate interval starts")
    if not f_table.select("period_start", "session").equals(intervals):
        missing = intervals.join(f_table, on=["period_start", "session"], how="anti").height
        extra = f_table.select("period_start", "session").join(intervals, on=["period_start", "session"], how="anti").height
        raise ValueError(f"factor intervals must match the portfolio on both endpoints exactly "
                         f"({missing} portfolio interval(s) unmatched, {extra} extra factor interval(s)); "
                         "no rows are dropped or filled")
    for nm in names:
        if factor_bases[nm] == "total_return" and (f_table[nm] <= -1).any():
            raise ValueError(f"total-return factor {nm!r} must exceed -1")

    subtract = {"dependent": model == "excess_returns" and y_basis == "total_return",
                **{nm: model == "excess_returns" and factor_bases[nm] == "total_return" for nm in names}}
    rf_values, rf_meta = None, None
    if any(subtract.values()):
        if risk_free is None:
            raise ValueError("excess_returns needs matched risk_free returns for total-return series")
        rf_table, rf_meta = _risk_free_series(risk_free, risk_free_metadata, portfolio, intervals, currency, frequency)
        rf_values = rf_table["simple_return"].to_list()
    elif risk_free is not None or risk_free_metadata is not None:
        raise ValueError("risk_free is only used to convert total-return series under model='excess_returns'")

    def series(values, key):
        return [v - f for v, f in zip(values, rf_values)] if subtract[key] else list(values)

    y = series(y_table["simple_return"].to_list(), "dependent")
    x = [series(f_table[nm].to_list(), nm) for nm in names]
    n, k = len(y), len(names)
    estimate, info, status = _ols(y, x, names)
    sst = math.fsum((v - math.fsum(y)/n)**2 for v in y)
    constant_y = len(set(y)) == 1
    df = n - k - 1

    if estimate is None:
        coef_rows = [("intercept", None, "fraction/period", status)] + [(nm, None, "ratio", status) for nm in names]
        fitted = residuals = [None]*n
        sse = r2 = idio = None
        r2_status = idio_status = sse_status = status
    else:
        alpha, beta, fitted = estimate
        coef_rows = [("intercept", alpha, "fraction/period", "ok")] + [(nm, b, "ratio", "ok") for nm, b in zip(names, beta)]
        residuals = [v - f for v, f in zip(y, fitted)]
        sse, sse_status = math.fsum(e*e for e in residuals), "ok"
        r2, r2_status = (None, "constant_dependent") if constant_y else (1 - sse/sst, "ok")
        idio, idio_status = ((math.sqrt(a*sse/df), "ok") if df > 0
                             else (None, "insufficient_residual_degrees_of_freedom"))
    coefficients = pl.DataFrame([(t, v, u, "joint_ols", s) for t, v, u, s in coef_rows],
        schema={"term": pl.String, "estimate": pl.Float64, "unit": pl.String, "estimation": pl.String,
                "status": pl.String}, orient="row")

    standalone_rows = []
    for nm, xs in zip(names, x):
        corr, b, corr_status, b_status = _pair(y, xs)
        if b is None:
            st = "insufficient_samples" if n < 2 else "constant_factor"
            standalone_rows.append((nm, None, None, None, None, None, n - 2, n, st))
            continue
        intercept = math.fsum(y)/n - b*math.fsum(xs)/n
        sse_j = math.fsum((v - intercept - b*f)**2 for v, f in zip(y, xs))
        idio_j = math.sqrt(a*sse_j/(n-2)) if n > 2 else None
        r2_j = corr*corr if corr is not None else None
        st = "constant_dependent" if constant_y else "insufficient_residual_degrees_of_freedom" if n <= 2 else "ok"
        standalone_rows.append((nm, b, intercept, corr, r2_j, idio_j, n - 2, n, st))
    standalone = pl.DataFrame(standalone_rows, schema={"factor": pl.String, "beta": pl.Float64,
        "intercept": pl.Float64, "correlation": pl.Float64, "r_squared": pl.Float64,
        "annualized_idiosyncratic_volatility": pl.Float64, "residual_df": pl.Int64, "n_obs": pl.Int64,
        "status": pl.String}, orient="row").select("factor", pl.lit("single_factor_ols").alias("estimation"), pl.exclude("factor"))

    labels, data = ["dependent"] + names, [y] + x
    corr_rows = []
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            value, _, st, _ = _pair(data[i], data[j])
            corr_rows.append((left, right, value, n, st))
    correlations = pl.DataFrame(corr_rows, schema={"series": pl.String, "other_series": pl.String,
        "correlation": pl.Float64, "n_obs": pl.Int64, "status": pl.String}, orient="row")

    summary = pl.DataFrame([
        ("n_obs", float(n), "intervals", n, "ok"), ("n_factors", float(k), "count", n, "ok"),
        ("rank", float(info["rank"]), "count", n, "ok"), ("residual_df", float(df), "count", n, "ok"),
        ("r_squared", r2, "fraction", n, r2_status), ("sse", sse, "fraction^2", n, sse_status),
        ("annualized_idiosyncratic_volatility", idio, "fraction/sqrt(year)", n, idio_status)],
        schema=METRIC_SCHEMA, orient="row")
    fitted_table = intervals.with_columns(pl.Series("dependent", y, dtype=pl.Float64),
        pl.Series("fitted", fitted, dtype=pl.Float64), pl.Series("residual", residuals, dtype=pl.Float64))
    for table in (coefficients, standalone, summary):
        floats = [c for c, t in table.schema.items() if t == pl.Float64]
        if any(v is not None and not math.isfinite(v) for c in floats for v in table[c]):
            raise ValueError("regression output is not representable")

    meta = dict(model=model, periods_per_year=a, status=status, n_obs=n, n_factors=k, rank=info["rank"],
        residual_df=df, factors=names, factor_bases=deepcopy(factor_bases), portfolio_basis=y_basis,
        portfolio=y_meta, factor_metadata=f_meta, risk_free=rf_meta,
        transformations={key: "minus_risk_free" if flag else "none" for key, flag in subtract.items()},
        long_short_policy="long_short and excess_return factors are never adjusted by the risk-free rate",
        alignment="exact_both_interval_endpoints_no_fill_no_drop",
        estimation="OLS with intercept: Householder QR with column pivoting on centered unit-norm factors",
        rank_tolerance=info["tolerance"], rank_rule="max(n,k)*machine_epsilon*|R[0,0]| on unit-norm centered factors",
        constant_factors=info["constant_factors"],
        coefficient_policy="unidentified (rank-deficient) designs return null coefficients; no pseudoinverse",
        idiosyncratic_volatility_definition=IDIO_NOTE,
        r_squared_definition="1-SSE/SST around the dependent mean (model includes an intercept)",
        standalone_definition="separate single-factor OLS with intercept; not the jointly estimated coefficient",
        interpretation="descriptive in-sample fit; no standard errors, predictive or causal claims")
    return FactorRegressionResult(coefficients, standalone, summary, correlations, fitted_table, meta)
