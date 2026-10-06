"""Documented rejections, solver guard rails and audit restores that had no direct test.

Each case names the input it breaks and the error it must produce, so a guard that
silently starts accepting bad input (or rejecting good input) fails here.
"""
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit._rebalancing import TARGET_SCHEMA
from test_allocation import panel, OPTIONS
from test_execution_extensions import model
from test_long_short import ls, loan, borrow
from test_metrics import KW, BM, benchmark
from test_research import split, predictions, features
from test_series_performance import KW as SERIES_KW, pnl_table
from test_yahoo_adapter import fixture as yahoo_fixture, raw_options

UTC = timezone.utc
FREE = rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.)
DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]


def hold_policy(leverage=1.):
    return rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
                            initial_gross_leverage=leverage, terminal_action="mark_only")


def rebalance_policy(**changes):
    return rt.RebalancePolicy(**(dict(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
        terminal_action="mark_only", non_session="raise", receivable_policy="require_target", max_asset_weight=1.) | changes))


def schedule(market, baskets, *, costs, threshold=.1):
    """baskets: (execution index, weights, leverage); decisions are the preceding session."""
    dates = market.sessions["session"].to_list()
    assets = sorted(market.prices["asset"].unique())
    targets = pl.DataFrame([(dates[i-1], dates[i], a, float(w[a]), float(lev)) for i, w, lev in baskets for a in assets],
                           schema=TARGET_SCHEMA, orient="row")
    return dict(targets=targets, initial_capital=1000., entry_session=dates[baskets[0][0]], end_session=dates[-1],
        policy=rebalance_policy(), costs=costs, financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
            maintenance_equity_ratio=threshold, on_breach="stop", cash_sweep="repay_debt"))


# --------------------------------------------------------------- solver guard rails

def test_proportional_solver_rejects_cost_rates_at_or_above_100_percent(inputs):
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*4}, dates=DAYS))
    with pytest.raises(ValueError, match="each cost rate < 100%"):
        rt.scheduled_rebalance(market, **schedule(market, [(1, {"A": 1.}, 1.)],
            costs=rt.TradeCosts(commission_bps=10_000., half_spread_bps=0., impact_bps=0.)))


def test_scheduled_basket_that_cannot_pay_its_own_liquidation_costs_raises(inputs):
    # 45% costs at 2x: after a 20% fall, selling everything would cost more than the remaining equity.
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 80., 80.]}, dates=DAYS))
    with pytest.raises(ValueError, match="insufficient equity to fund scheduled trade costs"):
        rt.scheduled_rebalance(market, **schedule(market, [(1, {"A": 1.}, 2.), (2, {"A": 1.}, 2.)],
            costs=rt.TradeCosts(commission_bps=4_500., half_spread_bps=0., impact_bps=0.)))


def test_nonlinear_scheduled_solver_rejects_non_monotonic_funding(inputs):
    market = rt.prepare_market_data(**inputs(series={"A": [10.]*4}, dates=DAYS))
    with pytest.raises(ValueError, match="cannot guarantee monotonic funding"):
        rt.scheduled_rebalance(market, **schedule(market, [(1, {"A": 1.}, 1.), (2, {"A": 1.}, .5)],
            costs=model(("A",), sigma=.1, adv=100., coefficient=50.)))


def test_signed_solvers_reject_unfundable_costs(inputs):
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*4}, dates=DAYS))
    common = dict(initial_capital=100., entry_session=DAYS[0], end_session=DAYS[-1], policy=hold_policy(),
                  financing=loan(), long_short=ls(), stock_borrow=borrow({}))
    with pytest.raises(ValueError, match="cannot guarantee monotonic funding at these costs/exposures"):
        rt.buy_and_hold(market, equity_exposures={"A": 2.}, **common,
                        costs=rt.TradeCosts(commission_bps=6_000., half_spread_bps=0., impact_bps=0.))
    with pytest.raises(ValueError, match="insufficient equity for fixed targets and trade costs"):
        rt.buy_and_hold(market, quantities={"A": 100.}, **common,
                        costs=rt.TradeCosts(commission_bps=10_000., half_spread_bps=0., impact_bps=0.))


# ------------------------------------------------------------ reports and comparisons

def test_compare_performance_guards(market, run):
    result = run(market)
    report = rt.performance(result, **KW)
    with_benchmark = rt.performance(result, benchmark=benchmark(result, [.01, -.01]), benchmark_metadata=BM, **KW)
    cases = [
        (lambda: rt.compare_performance({}), "nonempty mapping"),
        (lambda: rt.compare_performance({" ": report}), "nonblank labels"),
        (lambda: rt.compare_performance({"A": report}, coverage="intersect"), "strict or separate"),
        (lambda: rt.compare_performance({"A": report}, allow_partial="yes"), "allow_partial must be boolean"),
        (lambda: rt.compare_performance({"A": report}, benchmark_label="A"), "distinct from scenario labels"),
        (lambda: rt.compare_performance({"A": result}), "consumes PerformanceResult"),
        (lambda: rt.compare_performance({"A": report}, benchmark_label="Ref"), "benchmark in every report"),
        (lambda: rt.compare_performance({"A": with_benchmark, "Ref [A]": with_benchmark}, benchmark_label="Ref",
                                        assumptions="separate"), "conflicts with a scenario"),
        (lambda: rt.compare_performance({"A": replace(report, summary=pl.concat([report.summary, report.summary.head(1)]))}),
         "duplicate metrics"),
        (lambda: rt.compare_performance({"A": replace(report, summary=report.summary.with_columns(
            pl.lit(float("inf")).alias("value")))}), "finite or explicitly undefined"),
    ]
    for call, match in cases:
        with pytest.raises(ValueError, match=match):
            call()


def test_tail_risk_on_a_report_without_intervals(market, run):
    # performance() and series_performance() never return an empty report; a hand-built one is reported, not raised.
    report = rt.performance(run(market), **KW)
    table = rt.tail_risk(replace(report, daily=report.daily.clear()), confidence=[.9, .95])
    assert table["status"].to_list() == ["empty_sample"]*2 and table["n_obs"].to_list() == [0, 0]
    assert table.select("var_return", "etl_return", "var_pnl", "etl_pnl").null_count().row(0) == (2, 2, 2, 2)
    with pytest.raises(ValueError, match="number or a sequence"):
        rt.tail_risk(report, confidence=object())
    with pytest.raises(ValueError, match="must be finite"):
        rt.tail_risk(replace(report, daily=report.daily.with_columns(pl.lit(float("nan")).alias("pnl"))), confidence=.9)


def test_numpy_numbers_are_accepted_consistently(market, run):
    np = pytest.importorskip("numpy")
    result = run(market)
    numpy_kw = dict(periods_per_year=np.int64(4), risk_free_annual_effective=np.float64(0.),
                    minimum_acceptable_return_annual_effective=np.int64(0))
    assert rt.performance(result, **numpy_kw).summary.equals(rt.performance(result, **KW).summary)
    series = rt.series_performance(pnl_table([1., 2., 3.]), **(SERIES_KW | {"initial_capital": np.int64(100),
                                   "periods_per_year": np.int64(4)}))
    assert series.summary.equals(rt.series_performance(pnl_table([1., 2., 3.]), **SERIES_KW).summary)
    for bad in (True, float("inf"), 0):
        with pytest.raises(ValueError, match="periods_per_year must be finite and greater than 0"):
            rt.performance(result, **(KW | {"periods_per_year": bad}))


def test_liquidity_metadata_and_volume_guards(inputs):
    returns = panel(inputs)
    dates = returns.values["session"].unique().sort().to_list()
    def volume(value=1_000.):
        return pl.DataFrame([(d, a, value, datetime.combine(d, time(21), UTC)) for d in dates for a in ("A", "B")],
            schema={"session": pl.Date, "asset": pl.String, "dollar_volume": pl.Float64, "available_at": pl.Datetime("us", "UTC")},
            orient="row")
    meta = {"source": "synthetic volume", "currency": "USD"}
    estimate = rt.estimate_liquidity(returns, volume(), decision_session=OPTIONS["decision_session"], lookback=2, metadata=meta)
    assert "periods_per_year" not in estimate.metadata  # Daily units only; no placeholder annualization.
    assert estimate.metadata["volatility_unit"] == "daily_decimal" and estimate.metadata["sample_end"] == "2024-01-04"
    for table, metadata, match in [(volume(-1.), meta, "nonnegative"), (volume(), meta | {"currency": "EUR"}, "matching currency"),
                                   (volume(0.), meta, "must be positive")]:
        with pytest.raises(ValueError, match=match):
            rt.estimate_liquidity(returns, table, decision_session=OPTIONS["decision_session"], lookback=2, metadata=metadata)


# ----------------------------------------------------------------- research audit

def test_prior_audit_restore_rejects_inconsistent_history(research_inputs):
    s = split(research_inputs)
    study = rt.ResearchStudy(s)
    study.select(predictions(s.validation, {"a": .5}), parameters={"a": {"k": 1}}, metric="mean_squared_error")
    study.evaluate_test(predictions(s.test, {"a": .5}))
    audit = study.audit
    def at(sequence, column, value):
        return audit.with_columns(pl.when(pl.col("sequence") == sequence).then(pl.lit(value, dtype=audit.schema[column]))
                                  .otherwise(pl.col(column)).alias(column))
    cases = [
        (audit.with_columns(pl.col("sequence") + 1), "contiguous from zero"),
        (audit.with_columns(pl.lit("another split").alias("split_id")), "incompatible split"),
        (at(0, "metric", "hinge"), "incompatible split"),
        (at(0, "value", -1.), "incompatible split"),
        (at(0, "parameters", "[1, 2]"), "finite JSON dictionary"),
        (at(0, "parameters", "{not json"), "finite JSON dictionary"),
        (audit.filter(pl.col("sequence") == 1).with_columns(pl.lit(0, dtype=pl.Int64).alias("sequence")), "follow the recorded"),
        (at(1, "event", "peek"), "unknown research audit event"),
        (at(0, "n_obs", 999), "validation audit status/coverage"),
        (at(1, "n_obs", 999), "test reuse must be recorded"),
    ]
    for table, match in cases:
        with pytest.raises(ValueError, match=match):
            rt.ResearchStudy(s, prior_audit=table)


def test_research_argument_guards(research_inputs):
    data = research_inputs
    s = split(data)
    fresh = rt.ResearchStudy(s)
    val = predictions(s.validation, {"a": .5})
    def lag(**changes):
        kwargs = dict(sessions=data["sessions"], columns=["x"], lags=[1], decision_timing="session_close",
                      metadata=data["metadata"]) | changes
        return rt.lagged_features(data["observations"], **kwargs)
    cases = [
        (lambda: lag(decision_timing="open"), "session_close"),
        (lambda: lag(columns=["session"]), "distinct from reserved"),
        (lambda: lag(metadata=data["metadata"] | {"frequency": "1h"}), "frequency='1d'"),
        (lambda: lag(metadata=data["metadata"] | {"timezone": "Mars/Base"}), "valid IANA"),
        (lambda: lag(metadata=data["metadata"] | {"source": " "}), "nonblank"),
        (lambda: lag(metadata=data["metadata"] | {"bad": float("nan")}), "finite JSON"),
        (lambda: split(data, gap=-1), "nonnegative integer"),
        (lambda: rt.chronological_split(s, data["labels"], train_end=data["days"][4], validation_end=data["days"][8],
            test_end=data["days"][12], label_metadata={"source": "s", "unit": "u", "definition": "d"}), "expected FeatureResult"),
        (lambda: rt.standardize(s, zero_variance="drop"), "raise or unit_scale"),
        (lambda: rt.standardize(features(data)), "expected ResearchSplit"),
        (lambda: fresh.select(val, parameters={"a": {}}, metric="hinge"), "mean_squared_error or mean_absolute_error"),
        (lambda: fresh.select(val, parameters={"a": {}}, metric="mean_squared_error", tie_break="random"), "tie_break"),
        (lambda: fresh.select(val, parameters={"a": {}}, metric="mean_squared_error", reuse="maybe"), "reuse must be"),
        (lambda: fresh.select(val, parameters={"b": {}}, metric="mean_squared_error"), "map every candidate"),
        (lambda: fresh.select(val, parameters={"a": {"x": float("nan")}}, metric="mean_squared_error"), "finite JSON"),
    ]
    for call, match in cases:
        with pytest.raises(ValueError, match=match):
            call()
    assert fresh.audit.is_empty()  # Rejected calls record nothing.


def test_signal_targets_require_a_definition(research_inputs):
    data = research_inputs
    signals = pl.DataFrame([(data["days"][0], "A", 1., 1., datetime.combine(data["days"][0], time(20), UTC),
                             datetime.combine(data["days"][0], time(20), UTC))],
        schema={"decision_session": pl.Date, "asset": pl.String, "weight": pl.Float64, "gross_leverage": pl.Float64,
                "observed_at": pl.Datetime("us", "UTC"), "available_at": pl.Datetime("us", "UTC")}, orient="row")
    with pytest.raises(ValueError, match="signal_definition"):
        rt.signal_targets(signals, sessions=data["sessions"], execution="next_session_close", rebalance="on_change",
                          metadata=data["metadata"])


# ------------------------------------------------------------ provider conversions

def _raw(responses, args, **changes):
    args.update(raw_options(args, [1., 1., 1.]))
    for key, value in changes.items():
        args[key] = value(args) if callable(value) else value


@pytest.mark.parametrize("edit, match", [
    (lambda r, a: r.clear(), "responses must map"),
    (lambda r, a: a.update(availability="observed"), "availability"),
    (lambda r, a: a.update(volume_basis="shares"), "volume_basis"),
    (lambda r, a: a["metadata"].pop("price_basis"), "price_basis must explicitly select"),
    (lambda r, a: a["metadata"].update(retrieved_at="2024-02-01T00:00:00+01:00"), "UTC retrieved_at"),
    (lambda r, a: a["metadata"].update(retrieved_at="2024-01-04T00:00:00Z"), "retrieval precedes"),
    (lambda r, a: a.update(raw_adjustment_factors=raw_options(a, [1., 1., 1.])["raw_adjustment_factors"]), "apply only to raw"),
    (lambda r, a: _raw(r, a, raw_adjustment_factors=lambda a: a["raw_adjustment_factors"].head(2)), "cover every session"),
    (lambda r, a: _raw(r, a, factor_metadata=lambda a: a["factor_metadata"] | {"basis": "latest"}), "through_retrieval basis"),
    (lambda r, a: r["A"]["chart"]["result"].append(r["A"]["chart"]["result"][0]), "one successful chart result"),
    (lambda r, a: r["A"]["chart"]["result"][0].update(timestamp=[]), "nonempty integer"),
    (lambda r, a: r["A"]["chart"]["result"][0]["timestamp"].__setitem__(0, 1.5), "nonempty integer"),
    (lambda r, a: r["A"]["chart"]["result"][0]["indicators"]["quote"].append({}), "one quote block"),
    (lambda r, a: r["A"]["chart"]["result"][0]["indicators"]["adjclose"].append({}), "ambiguous adjusted close"),
    (lambda r, a: r["A"]["chart"]["result"][0]["indicators"]["quote"][0]["volume"].pop(), "mismatched lengths"),
    (lambda r, a: r["A"]["chart"]["result"][0].pop("indicators"), "malformed Yahoo daily chart response"),
])
def test_yahoo_chart_rejects_malformed_inputs(inputs, edit, match):
    responses, args = yahoo_fixture(inputs)
    edit(responses, args)
    with pytest.raises(ValueError, match=match):
        rt.adapters.yahoo_chart(responses, **args)


def test_yahoo_chart_needs_retained_responses_from_a_download(inputs):
    download = rt.ProviderDownloadResult(pl.DataFrame(), pl.DataFrame(), {"assets": ["A"]})
    _, args = yahoo_fixture(inputs)
    with pytest.raises(ValueError, match="does not retain the decoded provider responses"):
        rt.adapters.yahoo_chart(download, **args)


# ---------------------------------------------------------- configuration objects

def test_configuration_and_input_guards(inputs, market, run):
    backtest = run(market)
    report = rt.performance(backtest, **KW)
    returns = rt.returns(market, method="simple", basis="price")
    two = panel(inputs)
    log_two = rt.returns(rt.prepare_market_data(**inputs(series={"A": [100., 110., 99., 108.9, 120.], "B": [100., 120., 96., 115.2, 130.]},
        dates=[date(2024, 1, 2)+timedelta(days=i) for i in range(5)])), method="log", basis="price")
    zero_rf = backtest.daily.select("period_start", "session", pl.lit(0.).alias("simple_return"))
    data = inputs()
    def hold(**changes):
        kwargs = dict(weights={"A": 1.}, initial_capital=100., entry_session=market.sessions["session"][0],
                      end_session=market.sessions["session"][-1], policy=hold_policy(), costs=FREE,
                      cash_rate=0., cash_day_count="ACT/365F") | changes
        return rt.buy_and_hold(market, **kwargs)
    def prepare(**metadata_changes):
        return rt.prepare_market_data(**(data | {"metadata": data["metadata"] | metadata_changes}))
    stale = rt.LiquidityResult(pl.DataFrame({"asset": ["A"], "daily_volatility": [.1], "dollar_adv": [100.], "n_obs": [10]}),
        dict(source="s", currency="USD", volatility_unit="daily_decimal", adv_unit="currency_per_trading_day",
             sample_end="2024-01-02", decision_session="2024-01-02"))
    orders = pl.DataFrame({"asset": ["A"], "signed_notional": [10.], "reference_price": [10.]})
    rf_rates = pl.DataFrame({"date": [date(2024, 1, 3)], "annual_rate": [.01], "available_at": [datetime(2024, 1, 1, tzinfo=UTC)]},
                            schema={"date": pl.Date, "annual_rate": pl.Float64, "available_at": pl.Datetime("us", "UTC")})
    rf_intervals = pl.DataFrame({"period_start": [date(2024, 1, 2)], "session": [date(2024, 1, 3)]})
    def rf(**changes):
        kwargs = dict(intervals=rf_intervals, day_count="ACT/360", compounding="daily", rate_timing="known_at_accrual_start",
                      timezone_name="America/New_York", metadata=BM) | changes
        return rt.risk_free_returns(rf_rates, **kwargs)
    cases = [
        (lambda: rt.returns(market, method="arithmetic", basis="price"), "'simple' or 'log'"),
        (lambda: rt.cumulative_returns(market, method="sum"), "expected ReturnResult"),
        (lambda: rt.cumulative_returns(returns, method="geometric"), "unknown cumulative method"),
        (lambda: rt.cumulative_returns(replace(returns, values=returns.values.drop("period_start")), method="sum"), "invalid schema"),
        (lambda: rt.cumulative_returns(replace(returns, metadata=returns.metadata | {"sessions": returns.metadata["sessions"][::-1]}),
                                       method="sum"), "chronological"),
        (lambda: rt.cumulative_returns(replace(returns, metadata=returns.metadata | {"assets": []}), method="sum"), "unique sessions and assets"),
        (lambda: rt.inverse_volatility_weights(two, **(OPTIONS | {"decision_session": "2024-01-05"}), max_asset_weight=1.), "datetime.date"),
        (lambda: rt.inverse_volatility_weights(two, **OPTIONS, max_asset_weight=1., cap_policy="clip"), "raise or redistribute"),
        (lambda: rt.inverse_volatility_weights(two, **OPTIONS, max_asset_weight=1.5), r"\(0, 1\]"),
        (lambda: rt.inverse_volatility_weights(log_two, **OPTIONS, max_asset_weight=1.), "require simple returns"),
        (lambda: rt.inverse_volatility_weights(replace(two, metadata=two.metadata | {"basis": " "}), **OPTIONS,
                                               max_asset_weight=1.), "basis must be explicit"),
        (lambda: rt.risk_contributions(two, weights={"A": 1.}, gross_leverage=1., **OPTIONS), "cover every return asset"),
        (lambda: rt.rolling_risk(backtest, window=True, periods_per_year=4, risk_free_annual_effective=0.), "integer >= 2"),
        (lambda: rt.rolling_risk(backtest, window=2, periods_per_year=4, risk_free_annual_effective=0.,
                                 sharpe_denominator="downside"), "portfolio_returns or excess_returns"),
        (lambda: rt.rolling_risk(backtest, window=2, periods_per_year=4, risk_free_annual_effective=0.,
                                 benchmark_metadata=BM), "requires a benchmark"),
        (lambda: rt.performance(report, **KW), "expected BacktestResult"),
        (lambda: rt.performance(backtest, allow_partial="no", **KW), "allow_partial must be boolean"),
        (lambda: rt.performance(backtest, sharpe_denominator="downside", **KW), "portfolio_returns or excess_returns"),
        (lambda: rt.performance(backtest, alignment="nearest", **KW), "alignment='strict'"),
        (lambda: rt.performance(backtest, risk_free_metadata=BM, **KW), "requires risk_free_returns"),
        (lambda: rt.performance(backtest, risk_free_returns=zero_rf, risk_free_metadata=BM, **KW), "exactly one"),
        (lambda: rt.equal_weights(5), "collection of asset names"),
        (lambda: hold(weights=pl.DataFrame({"asset": ["A"], "w": [1.]})), "weights schema"),
        (lambda: hold(weights=pl.DataFrame({"asset": ["A", "A"], "weight": [.5, .5]})), "duplicate weight assets"),
        (lambda: hold(costs="free"), "supported cost model"),
        (lambda: hold(dividend_reinvestment=True), "DividendReinvestment object"),
        (lambda: hold(long_short=ls()), "signed position targets"),
        (lambda: hold(weights=None, quantities={"A": 1.}), "require LongShortPolicy"),
        (lambda: hold(weights=None, quantities={"A": 1.}, long_short=ls(), stock_borrow=borrow({}), financing=loan(),
                      cash_rate=None, cash_day_count=None, policy=hold_policy(2.)), "initial_gross_leverage=1"),
        (lambda: hold(weights=None, quantities={"A": float("nan")}, long_short=ls(), stock_borrow=borrow({}), financing=loan(),
                      cash_rate=None, cash_day_count=None), "finite signed numbers"),
        (lambda: rt.scheduled_rebalance(market, targets=None, initial_capital=1., entry_session=None, end_session=None,
                                        policy=hold_policy(), costs=FREE, financing=loan()), "requires RebalancePolicy"),
        (lambda: rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=False,
                                  initial_gross_leverage=1., terminal_action="mark_only"), "fractional_shares=True"),
        (lambda: rebalance_policy(execution="open"), "scheduled_close"),
        (lambda: rebalance_policy(fractional_shares=False), "fractional shares"),
        (lambda: rebalance_policy(non_session="roll_forward"), "non_session"),
        (lambda: rebalance_policy(receivable_policy="borrow"), "reserve or require_target"),
        (lambda: rebalance_policy(max_asset_weight=2.), r"\(0, 1\]"),
        (lambda: rebalance_policy(decision_timing="intraday"), "decision_timing"),
        (lambda: ls(long_margin=0.), r"\(0, 1\]"),
        (lambda: ls(rebate_day_count="30/360"), "ACT/360 or ACT/365F"),
        (lambda: rt.StockBorrow(rates={}, day_count="30/360", metadata={"source": "s", "basis": "modeled"}), "ACT/360 or ACT/365F"),
        (lambda: rt.StockBorrow(rates={}, day_count="ACT/360", metadata={"source": "s", "basis": "guess"}), "modeled/measured"),
        (lambda: rt.StockBorrow(rates={" ": 0.}, day_count="ACT/360", metadata={"source": "s", "basis": "modeled"}), "nonblank"),
        (lambda: rt.StockBorrow(rates=[("A", .01)], day_count="ACT/360", metadata={"source": "s", "basis": "modeled"}),
         "asset mapping or dated Polars table"),
        (lambda: rt.StockBorrow(rates={}, day_count="ACT/360", metadata={"source": "s", "basis": "modeled", "x": float("nan")}),
         "finite JSON"),
        (lambda: rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/360", maintenance_equity_ratio=None,
                              on_breach="stop", cash_sweep="repay_debt"), "ACT/365F"),
        (lambda: rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=None,
                              on_breach="liquidate", cash_sweep="repay_debt"), "on_breach='stop'"),
        (lambda: rt.prepare_market_data(**(data | {"metadata": [("source", "x")]})), "JSON-compatible mapping"),
        (lambda: prepare(extra=float("nan")), "finite JSON-compatible"),
        (lambda: prepare(calendar=" "), "nonblank string"),
        (lambda: prepare(coverage_start=20240102), "coverage_start must be a nonblank string"),
        (lambda: prepare(coverage_start="2024-01-02T00:00"), "ISO date string|YYYY-MM-DD"),
        (lambda: rt.prepare_market_data(**(data | {"sessions": data["sessions"].with_columns(
            pl.col("close_at").reverse())})), "increase strictly|disagrees"),
        (lambda: rt.prepare_market_data(**(data | {"prices": data["prices"].with_columns(pl.lit(" ").alias("asset"))})), "blank"),
        (lambda: rf(day_count="30/360"), "explicit ACT/360 or ACT/365F"),
        (lambda: rf(timezone_name="Mars/Base"), "valid IANA"),
        (lambda: rf(intervals=rf_intervals.with_columns(pl.col("session").alias("period_start"))), "increasing and contiguous"),
        (lambda: rt.estimate_trade_costs(orders, costs=FREE, execution_session=date(2024, 1, 3)), "requires SquareRootImpactCosts"),
        (lambda: rt.size_entry_orders(weights={"A": 1.}, initial_capital=100., gross_leverage=1.,
            prices=orders.select("asset", "reference_price"), costs=FREE, execution_session=date(2024, 1, 3)),
         "requires SquareRootImpactCosts"),
        (lambda: rt.SquareRootImpactCosts(liquidity=pl.DataFrame(), commission_bps=0., commission_per_share=0.,
            half_spread_bps=0., impact_coefficient=0.), "expected LiquidityResult"),
        (lambda: rt.SquareRootImpactCosts(liquidity=stale, commission_bps=0., commission_per_share=0.,
            half_spread_bps=0., impact_coefficient=0.), "strictly before decision_session"),
        (lambda: rt.SquareRootImpactCosts(liquidity=replace(stale, metadata=stale.metadata | {"sample_end": "2023-12-29"}),
            commission_bps={"B": 1.}, commission_per_share=0., half_spread_bps=0., impact_coefficient=0.), "cover exactly"),
        (lambda: rt.SquareRootImpactCosts(liquidity=replace(stale, metadata=stale.metadata | {"adv_unit": "shares"}),
            commission_bps=0., commission_per_share=0., half_spread_bps=0., impact_coefficient=0.), "units are required"),
    ]
    failures = []
    for number, (call, match) in enumerate(cases):
        try:
            with pytest.raises(ValueError, match=match):
                call()
        except AssertionError as exc:  # Report every broken guard at once, not only the first.
            failures.append(f"case {number}: {str(exc).splitlines()[-1] if str(exc) else exc!r}")
        except pytest.fail.Exception as exc:
            failures.append(f"case {number}: {exc}")
    assert not failures, "\n".join(failures)
