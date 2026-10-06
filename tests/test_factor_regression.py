from datetime import date, timedelta
import math

import polars as pl
import pytest

import research_toolkit as rt
from test_metrics import metric
from test_series_performance import META, pnl_table

FMETA = {"source": "hand-built factors", "currency": "USD", "frequency": "1d"}
PMETA = {"source": "hand-built portfolio", "currency": "USD", "frequency": "1d"}


def days(n):
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n + 1)]


def portfolio(values):
    d = days(len(values))
    return pl.DataFrame({"period_start": d[:-1], "session": d[1:], "simple_return": values},
                        schema={"period_start": pl.Date, "session": pl.Date, "simple_return": pl.Float64})


def factors(**columns):
    n = len(next(iter(columns.values())))
    d = days(n)
    return pl.DataFrame({"period_start": d[:-1], "session": d[1:], **columns},
                        schema={"period_start": pl.Date, "session": pl.Date, **{k: pl.Float64 for k in columns}})


def raw(y, f, bases=None, **kw):
    bases = bases or {name: "long_short" for name in f.columns[2:]}
    return rt.factor_regression(portfolio(y), f, factor_bases=bases, factor_metadata=FMETA,
                                periods_per_year=kw.pop("periods_per_year", 252), model="raw_returns",
                                portfolio_basis="total_return", portfolio_metadata=PMETA, **kw)


def coef(result, term):
    return result.coefficients.filter(pl.col("term") == term)["estimate"].item()


def test_joint_versus_standalone_betas_by_hand():
    # y = x1 + x2 with correlated factors: jointly (1, 1), but separately 16/10 and 10/4.
    x1 = [-.02, -.01, 0., .01, .02]
    x2 = [-.01, -.01, 0., .01, .01]
    result = raw([a + b for a, b in zip(x1, x2)], factors(x1=x1, x2=x2))
    assert coef(result, "x1") == pytest.approx(1) and coef(result, "x2") == pytest.approx(1)
    assert coef(result, "intercept") == pytest.approx(0, abs=1e-15)
    standalone = dict(result.standalone.select("factor", "beta").iter_rows())
    assert standalone == pytest.approx({"x1": 1.6, "x2": 2.5})
    assert set(result.coefficients["estimation"]) == {"joint_ols"}
    assert set(result.standalone["estimation"]) == {"single_factor_ols"}
    # A perfect fit: R^2 = 1 and zero residual risk with n-k-1 = 2 residual degrees of freedom.
    assert metric(result.summary, "r_squared")["value"] == pytest.approx(1)
    assert metric(result.summary, "residual_df")["value"] == 2
    assert metric(result.summary, "annualized_idiosyncratic_volatility")["value"] == pytest.approx(0, abs=1e-12)
    assert result.fitted["residual"].to_list() == pytest.approx([0]*5, abs=1e-15)


def test_known_coefficients_residual_df_and_idiosyncratic_volatility():
    x = [-.01, .01, -.01, .01]
    e = [.01, .01, -.01, -.01]  # Mean zero and orthogonal to x, so OLS recovers exactly 0.002 and 0.5.
    y = [.002 + .5*a + b for a, b in zip(x, e)]
    result = raw(y, factors(mkt=x))
    assert coef(result, "intercept") == pytest.approx(.002) and coef(result, "mkt") == pytest.approx(.5)
    assert metric(result.summary, "sse")["value"] == pytest.approx(4e-4)
    assert metric(result.summary, "residual_df")["value"] == 2 and metric(result.summary, "rank")["value"] == 2
    idio = metric(result.summary, "annualized_idiosyncratic_volatility")["value"]
    assert idio == pytest.approx(math.sqrt(252*4e-4/2))
    # Not the ddof=1 sample standard deviation of the fitted residuals.
    assert idio != pytest.approx(math.sqrt(252*4e-4/3))
    assert result.fitted["residual"].to_list() == pytest.approx(e)
    assert "degrees-of-freedom" in result.metadata["idiosyncratic_volatility_definition"]
    assert metric(result.summary, "r_squared")["value"] == pytest.approx(1 - 4e-4/(4e-4 + .25*4e-4))


def test_matches_numpy_least_squares():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    f = rng.normal(0, .01, (40, 3))
    y = .0005 + f @ [1.2, -.4, .3] + rng.normal(0, .004, 40)
    result = raw(y.tolist(), factors(a=f[:, 0].tolist(), b=f[:, 1].tolist(), c=f[:, 2].tolist()), periods_per_year=52)
    design = np.column_stack([np.ones(40), f])
    expected, sse, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    assert result.coefficients["estimate"].to_list() == pytest.approx(expected.tolist(), rel=1e-10)
    assert metric(result.summary, "sse")["value"] == pytest.approx(sse[0], rel=1e-10)
    assert metric(result.summary, "rank")["value"] == rank == 4
    assert metric(result.summary, "annualized_idiosyncratic_volatility")["value"] == pytest.approx(
        math.sqrt(52*sse[0]/36), rel=1e-10)
    corr = np.corrcoef(np.column_stack([y, f]).T)
    table = result.correlations.filter((pl.col("series") == "dependent") & (pl.col("other_series") == "b"))
    assert table["correlation"].item() == pytest.approx(corr[0, 2], rel=1e-10)


def test_singular_designs_are_not_estimated_by_pseudoinverse():
    x = [-.01, .02, 0., .01, -.02]
    collinear = raw([.5*v for v in x], factors(a=x, b=[2*v for v in x]))
    assert collinear.metadata["status"] == "rank_deficient" and metric(collinear.summary, "rank")["value"] == 2
    assert collinear.coefficients["estimate"].null_count() == 3
    assert set(collinear.coefficients["status"]) == {"rank_deficient"}
    assert collinear.fitted["residual"].null_count() == 5
    assert metric(collinear.summary, "annualized_idiosyncratic_volatility")["status"] == "rank_deficient"
    # Standalone fits remain separately identified.
    assert dict(collinear.standalone.select("factor", "beta").iter_rows()) == pytest.approx({"a": .5, "b": .25})

    constant = raw(x, factors(a=x, flat=[.001]*5))
    assert constant.metadata["status"] == "rank_deficient" and constant.metadata["constant_factors"] == ["flat"]
    assert constant.standalone.filter(pl.col("factor") == "flat")["status"].item() == "constant_factor"
    corr = constant.correlations.filter((pl.col("series") == "dependent") & (pl.col("other_series") == "flat"))
    assert corr["correlation"].item() is None and corr["status"].item() == "zero_volatility"

    few = raw([.01, .02], factors(a=[.01, .03], b=[.02, -.01]))
    assert few.metadata["status"] == "insufficient_observations"


def test_zero_residual_degrees_of_freedom_and_constant_dependent():
    exact = raw([.01, .03, .02], factors(a=[.01, .02, .04], b=[.03, -.01, .02]))
    assert exact.metadata["status"] == "ok" and metric(exact.summary, "residual_df")["value"] == 0
    row = metric(exact.summary, "annualized_idiosyncratic_volatility")
    assert row["value"] is None and row["status"] == "insufficient_residual_degrees_of_freedom"
    flat = raw([.01]*4, factors(a=[.01, -.02, .03, 0.]))
    assert coef(flat, "a") == pytest.approx(0, abs=1e-15) and coef(flat, "intercept") == pytest.approx(.01)
    r2 = metric(flat.summary, "r_squared")
    assert r2["value"] is None and r2["status"] == "constant_dependent"
    assert flat.standalone["status"].item() == "constant_dependent"


def test_strict_interval_alignment():
    y = [.01, -.02, .015, .005]
    f = factors(a=[.02, -.01, .01, 0.])
    with pytest.raises(ValueError, match="both endpoints"):
        raw(y, f.slice(0, 3))
    with pytest.raises(ValueError, match="both endpoints"):
        raw(y, f.with_columns(pl.Series("period_start", [date(2023, 12, 31)] + f["period_start"][1:].to_list())))
    with pytest.raises(ValueError, match="duplicate"):
        raw(y, pl.concat([f, f.slice(0, 1)]))
    with pytest.raises(ValueError, match="finite"):
        raw(y, f.with_columns(pl.Series("a", [.02, float("nan"), .01, 0.])))
    with pytest.raises(ValueError, match="null"):
        raw(y, f.with_columns(pl.Series("a", [.02, None, .01, 0.], dtype=pl.Float64)))
    with pytest.raises(ValueError, match="factor_bases"):
        raw(y, f, bases={"a": "long_short", "b": "long_short"})
    with pytest.raises(ValueError, match="currency/frequency"):
        rt.factor_regression(portfolio(y), f, factor_bases={"a": "long_short"}, factor_metadata={**FMETA, "frequency": "1w"},
                             periods_per_year=252, model="raw_returns", portfolio_basis="total_return",
                             portfolio_metadata=PMETA)
    # Rows may arrive unsorted; they are matched by interval keys, never by position.
    shuffled = f.reverse()
    assert coef(raw(y, shuffled), "a") == pytest.approx(coef(raw(y, f), "a"))


def test_excess_returns_subtract_risk_free_only_from_total_return_series():
    daily = pnl_table([1., -2., 1.5])
    rep = rt.series_performance(daily, initial_capital=100., periods_per_year=4,
                                risk_free_annual_effective=.0816, metadata=META)
    rf = rep.risk_free_returns["simple_return"][0]
    intervals = rep.daily.select("period_start", "session")
    f = intervals.with_columns(pl.Series("mkt", [.02, -.01, .03]), pl.Series("smb", [.01, .02, -.01]),
                               pl.Series("mkt_rf", [.01, -.02, .02]))
    bases = {"mkt": "total_return", "smb": "long_short", "mkt_rf": "excess_return"}
    result = rt.factor_regression(rep, f.select("period_start", "session", "mkt"), factor_bases={"mkt": "total_return"},
                                  factor_metadata=FMETA, periods_per_year=4, model="excess_returns", risk_free="report")
    assert result.fitted["dependent"].to_list() == pytest.approx([r - rf for r in rep.daily["simple_return"]])
    assert result.metadata["transformations"] == {"dependent": "minus_risk_free", "mkt": "minus_risk_free"}
    full = rt.factor_regression(rep, f, factor_bases=bases, factor_metadata=FMETA, periods_per_year=4,
                                model="excess_returns", risk_free="report")
    assert full.metadata["transformations"] == {"dependent": "minus_risk_free", "mkt": "minus_risk_free",
                                                "smb": "none", "mkt_rf": "none"}
    assert full.metadata["status"] == "insufficient_observations"
    with pytest.raises(ValueError, match="risk_free"):
        rt.factor_regression(rep, f, factor_bases=bases, factor_metadata=FMETA, periods_per_year=4, model="excess_returns")
    with pytest.raises(ValueError, match="only used"):
        rt.factor_regression(rep, f, factor_bases=bases, factor_metadata=FMETA, periods_per_year=4,
                             model="raw_returns", risk_free="report")
    # Already-excess portfolio and long-short factors need no risk-free series at all.
    excess = rt.factor_regression(portfolio([.01, -.02, .015, .005]), factors(smb=[.01, .02, -.01, 0.]),
        factor_bases={"smb": "long_short"}, factor_metadata=FMETA, periods_per_year=252, model="excess_returns",
        portfolio_basis="excess_return", portfolio_metadata=PMETA)
    assert excess.metadata["risk_free"] is None and set(excess.metadata["transformations"].values()) == {"none"}


def test_supplied_risk_free_table_must_align():
    y = [.01, -.02, .015, .005]
    f = factors(mkt=[.02, -.01, .01, 0.])
    rf = portfolio([.0001]*4)
    rf_meta = {"source": "example", "basis": "T-bill", "currency": "USD", "frequency": "1d"}
    kw = dict(factor_bases={"mkt": "total_return"}, factor_metadata=FMETA, periods_per_year=252,
              model="excess_returns", portfolio_basis="total_return", portfolio_metadata=PMETA)
    result = rt.factor_regression(portfolio(y), f, risk_free=rf, risk_free_metadata=rf_meta, **kw)
    assert result.fitted["dependent"].to_list() == pytest.approx([v - .0001 for v in y])
    with pytest.raises(ValueError, match="both endpoints"):
        rt.factor_regression(portfolio(y), f, risk_free=rf.slice(1), risk_free_metadata=rf_meta, **kw)
