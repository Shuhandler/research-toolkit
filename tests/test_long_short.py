"""Small independent cash, share and P&L oracles for signed portfolios."""
from datetime import date, datetime, timezone, timedelta

import polars as pl
import pytest

import research_toolkit as rt


def ls(**kwargs):
    return rt.LongShortPolicy(**(dict(collateral_multiple=1., long_margin=.25,
        short_margin=.30, rebate_rate=0., rebate_day_count="ACT/365F") | kwargs))


def loan(**kwargs):
    return rt.Financing(**(dict(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
        maintenance_equity_ratio=None, on_breach="stop", cash_sweep="repay_debt") | kwargs))


def borrow(rates):
    return rt.StockBorrow(rates=rates, day_count="ACT/365F", metadata={"source": "synthetic assumptions", "basis": "modeled"})


def simulate(market, policy, **kwargs):
    options = dict(quantities={"A": -1.}, initial_capital=100.,
        entry_session=market.sessions["session"][0], end_session=market.sessions["session"][-1],
        policy=policy(), costs=rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.),
        financing=loan(), long_short=ls(), stock_borrow=borrow({"A": 0.}))
    options.update(kwargs)
    return rt.buy_and_hold(market, **options)


def report(run, benchmark=None):
    if benchmark is None:
        benchmark = run.daily.select("period_start", "session", pl.lit(0.).alias("simple_return"))
    return rt.performance(run, periods_per_year=252, risk_free_annual_effective=0.,
        minimum_acceptable_return_annual_effective=0., benchmark=benchmark,
        benchmark_metadata={"source": "supplied zero", "basis": "zero return", "frequency": "1d", "currency": "USD"},
        allow_partial=run.status != "complete")


def test_short_price_pnl_and_no_equity_created(inputs, policy):
    market = rt.prepare_market_data(**inputs({"A": [100, 90, 110]}))
    r = simulate(market, policy)
    assert r.valuations.filter(pl.col("phase") == "post_entry")["equity"].item() == 100
    assert r.daily["equity"].to_list() == [110, 90]
    assert r.daily["pnl"].to_list() == [10, -20]
    assert r.daily["cash"].to_list() == [110, 90]
    assert r.daily["restricted_collateral"].to_list() == [90, 110]
    assert r.daily["short_exposure"].to_list() == [90, 110]
    assert r.daily["net_exposure"].to_list() == [-90, -110]
    assert r.daily["simple_return"].to_list() == pytest.approx([.1, -20/110])
    assert r.dividend_liabilities.is_empty()
    assert r.diagnostics["residual"].abs().max() < 1e-8


def test_mixed_hedge_preserves_longs_and_does_not_fund_them(inputs, policy):
    m = rt.prepare_market_data(**inputs({"A": [100, 110, 100], "H": [100, 90, 110]}))
    base = simulate(m, policy, quantities={"A": 2., "H": 0.}, stock_borrow=borrow({}))
    hedged = simulate(m, policy, quantities={"A": 2., "H": -1.}, stock_borrow=borrow({"H": 0.}))
    e = hedged.valuations.filter(pl.col("phase") == "post_entry").row(0, named=True)
    assert e["debt"] == 100
    assert e["restricted_collateral"] == 100
    assert e["cash"] == 0
    assert hedged.positions.filter(pl.col("asset") == "A")["quantity"].to_list() == base.positions.filter(pl.col("asset") == "A")["quantity"].to_list()
    assert hedged.daily["equity"].to_list() == [130, 90]
    assert hedged.daily["debt"].to_list() == [90, 110]
    assert hedged.daily["gross_exposure"].to_list() == [310, 310]
    assert hedged.daily["net_exposure"].to_list() == [130, 90]
    comparison = rt.compare_performance({"base": report(base), "hedged": report(hedged)})
    assert "information_ratio" in comparison.values["metric"]
    assert report(hedged).allocation.group_by("session").agg(pl.col("weight").sum())["weight"].to_list() == pytest.approx([1, 1, 1])


def test_cost_sizing_signed_exposures_and_exact_quantities(inputs, policy):
    m = rt.prepare_market_data(**inputs({"A": [100, 100, 100], "H": [100, 100, 100]}))
    fees = rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.)
    r = simulate(m, policy, quantities=None, equity_exposures={"A": 1.5, "H": -.5}, stock_borrow=borrow({"H": 0.}), costs=fees)
    # E + 1%*(1.5E + .5E) = 100.
    assert r.daily["equity"][0] == pytest.approx(100/1.02)
    assert r.trades.sort("asset")["signed_notional"].to_list() == pytest.approx([150/1.02, -50/1.02])
    exact = simulate(m, policy, quantities={"A": 1.5, "H": -.5}, stock_borrow=borrow({"H": 0.}), costs=fees)
    assert exact.daily["equity"].to_list() == pytest.approx([98, 98])
    assert exact.valuations.filter(pl.col("phase") == "post_entry")["debt"].item() == pytest.approx(52)
    assert exact.costs["amount"].sum() == 2
    assert exact.trades.height == 2


def test_weekend_dated_borrow_fees_loan_and_gross_rebate(inputs, policy):
    fri, mon, tue = date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)
    m = rt.prepare_market_data(**inputs({"A": [100, 100, 100], "H": [100, 100, 100]}, dates=[fri, mon, tue]))
    dates = [fri+timedelta(days=i) for i in range(1, 5)]
    rates = pl.DataFrame({"date": dates, "asset": ["H"]*4, "annual_rate": [.365, .365, .73, .73],
        "available_at": [datetime(2024,1,5,tzinfo=timezone.utc)]*4},
        schema={"date": pl.Date, "asset": pl.String, "annual_rate": pl.Float64, "available_at": pl.Datetime("us", "UTC")})
    r = simulate(m, policy, quantities={"A": 2., "H": -1.}, stock_borrow=borrow(rates),
        financing=loan(borrowing_rate=.365), long_short=ls(rebate_rate=.1825))
    assert r.stock_borrow_accruals["amount"].to_list() == pytest.approx([.1, .1, .2, .2])
    assert r.short_financing_accruals["rebate"].to_list() == pytest.approx([.05]*4)
    # Each day: debt interest, fee, then the separate rebate repays debt.
    debt = 100.
    expected = []
    for fee in [.1, .1, .2, .2]:
        debt = debt*1.001+fee-.05
        expected.append(debt)
    assert r.daily["debt"].to_list() == pytest.approx([expected[2], expected[3]])
    assert r.daily["equity"][-1] == pytest.approx(200-debt)
    assert r.stock_borrow_accruals["mark_session"].to_list() == [fri, fri, fri, mon]
    with pytest.raises(ValueError, match="every requested calendar date"):
        simulate(m, policy, quantities={"A": 2., "H": -1.}, stock_borrow=borrow(rates.slice(1)))
    with pytest.raises(ValueError, match="midnight"):
        borrow(rates.with_columns(pl.lit(datetime(2024,2,1,tzinfo=timezone.utc)).alias("available_at")))


def test_short_dividend_liability_payment_and_split(inputs, policy):
    dates = [date(2024,1,i) for i in (2,3,4,5)]
    m = rt.prepare_market_data(**inputs({"A": [100, 49, 49, 49]}, dates=dates,
        splits=[("s", "A", dates[1], 2.)], dividends=[("d", "A", dates[1], dates[2], 1.)]))
    r = simulate(m, policy, dividend_reinvestment=rt.DividendReinvestment(execution="first_close_on_or_after_payment",
        funding="before_debt_repayment", scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    assert r.positions["quantity"].to_list() == [-1,-2,-2,-2]
    assert r.daily["equity"].to_list() == [100,100,100]
    assert r.daily["dividend_liability"].to_list() == [2,0,0]
    assert r.daily["restricted_collateral"].to_list() == [98,98,98]
    assert r.events.filter(pl.col("type") == "short_dividend_payment")["cash_delta"].item() == -2
    assert r.dividend_reinvestments.is_empty()
    assert r.trades.height == 1
    assert r.attribution.filter(pl.col("component") == "short_dividend_expense")["pnl"].sum() == -2


def scheduled(market, dates, baskets, *, kind="quantity", **kwargs):
    targets = pl.DataFrame([{"decision_session": dates[i-1], "session": dates[i], "asset": a, kind: float(v)}
        for i, b in baskets.items() for a,v in b.items()],
        schema={"decision_session": pl.Date, "session": pl.Date, "asset": pl.String, kind: pl.Float64})
    options = dict(targets=targets, initial_capital=100., entry_session=dates[1], end_session=dates[-1],
        policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
            terminal_action="mark_only", non_session="raise", receivable_policy="require_target", max_asset_weight=1.),
        costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.), financing=loan(),
        long_short=ls(), stock_borrow=borrow({"A": 0.}))
    return rt.scheduled_rebalance(market, **(options | kwargs))


def test_scheduled_partial_cover_zero_trade_and_crossings(inputs):
    dates = [date(2024,1,i) for i in range(2,9)]
    m = rt.prepare_market_data(**inputs({"A": [10]*7}, dates=dates))
    r = scheduled(m, dates, {1:{"A": -2}, 2:{"A": -1}, 3:{"A": -1}, 4:{"A": 1}, 5:{"A": -1}})
    assert r.trades["signed_quantity"].to_list() == [-2,1,2,-2]
    assert r.trades["trade_cost"].to_list() == pytest.approx([.2,.1,.2,.2])
    assert r.daily["equity"].to_list() == pytest.approx([99.7,99.7,99.5,99.3,99.3])
    assert r.rebalances["gross_traded_notional"].to_list() == [10,0,20,20]
    assert r.daily["restricted_collateral"].to_list() == [10,10,0,10,10]
    assert r.daily["pnl"].sum() == pytest.approx(-.7)
    report(r)


@pytest.mark.parametrize("last,reason,equity", [(160,"maintenance_margin_breach",40), (200,"nonpositive_equity",0), (220,"nonpositive_equity",-20)])
def test_margin_insolvency_and_stopped_reporting(inputs, policy, last, reason, equity):
    m = rt.prepare_market_data(**inputs({"A": [100,last,90]}))
    r = simulate(m, policy, long_short=ls(short_margin=.5))
    assert r.status == "stopped"
    assert r.stop_reason == reason
    assert r.daily.height == 1
    assert r.daily["equity"].item() == equity
    assert r.daily["simple_return"].item() == pytest.approx((equity-100)/100)
    with pytest.raises(ValueError, match="stopped"):
        r.require_complete()
    p = report(r)
    assert p.metadata["actual_end_session"] == m.sessions["session"][1].isoformat()
    if equity < 0:
        assert r.daily["log_return"].item() is None
        assert r.daily["gross_leverage"].item() is None


def test_invalid_signed_assumptions(market, policy):
    with pytest.raises(ValueError, match="exactly one"):
        simulate(market, policy, weights={"A":1.})
    with pytest.raises(ValueError, match="cover exactly"):
        simulate(market, policy, stock_borrow=borrow({}))
    with pytest.raises(ValueError, match="set financing"):
        simulate(market, policy, financing=loan(maintenance_equity_ratio=.25))
    with pytest.raises(ValueError, match="entry violates"):
        simulate(market, policy, quantities={"A":-3.}, long_short=ls(short_margin=.5))
    with pytest.raises(ValueError, match="collateral_multiple"):
        ls(collateral_multiple=.9)


def test_zero_benchmark_information_ratio_and_undefined_status(inputs, policy):
    m = rt.prepare_market_data(**inputs({"A": [100,90,95]}))
    p = report(simulate(m, policy))
    # Portfolio returns .1 and -1/22. Sample standard deviation of two values.
    expected = (252**.5)*((.1-1/22)/2)/((.1+1/22)/(2**.5))
    metric = p.benchmark_comparison.filter(pl.col("metric") == "information_ratio")
    assert metric["value"].item() == pytest.approx(expected)
    assert p.benchmark_comparison.filter(pl.col("metric") == "beta")["status"].item() == "zero_benchmark_variance"
    flat = rt.prepare_market_data(**inputs({"A": [100,100,100]}))
    row = report(simulate(flat, policy)).benchmark_comparison.filter(pl.col("metric") == "information_ratio")
    assert row["value"].item() is None
    assert row["status"].item() == "zero_tracking_error"


def test_short_sqrt_costs_include_per_share_and_post_cost_exposure(inputs, policy):
    from test_execution_extensions import model
    m = rt.prepare_market_data(**inputs({"A": [10,10,10]}))
    costs = model(sigma=.02, adv=10_000., coefficient=.5, commission=2., spread=3., per_share=.01)
    r = simulate(m, policy, quantities={"A": -10.}, costs=costs)
    # 10 shares: $.12 commission, $.03 spread, $.10 impact; no second preview debit.
    assert r.daily["equity"].to_list() == pytest.approx([99.75,99.75])
    assert r.costs["amount"].sum() == pytest.approx(.25)
    assert r.execution_costs["total_cost"].sum() == pytest.approx(.25)
    assert r.trades["trade_cost"].sum() == pytest.approx(.25)
    sized = simulate(m, policy, quantities=None, equity_exposures={"A": -1.}, initial_capital=100.25, costs=costs)
    assert sized.trades["signed_quantity"].item() == pytest.approx(-10)
    assert sized.daily["equity"].to_list() == pytest.approx([100,100])


def test_scheduled_impact_crossing_and_no_trade(inputs):
    from test_execution_extensions import model
    dates = [date(2024,1,i) for i in range(2,7)]
    m = rt.prepare_market_data(**inputs({"A": [10]*5}, dates=dates))
    r = scheduled(m, dates, {1:{"A": -10},2:{"A":-10},3:{"A":10}},
        costs=model(sigma=.02, adv=10_000., coefficient=.5, commission=2., spread=3., per_share=.01))
    assert r.trades["signed_quantity"].to_list() == [-10,20]
    assert r.rebalances["trade_cost"][0] == 0
    # Crossing trades $200: $.24 commission + $.06 spread + $.2*sqrt(2) impact.
    assert r.trades["trade_cost"][1] == pytest.approx(.30+.2*2**.5)
    assert r.daily["equity"][-1] == pytest.approx(100-.25-.30-.2*2**.5)


def test_historical_sofr_and_stock_loan_are_separate(inputs, policy):
    from test_sofr import sofr_config
    dates = [date(2024,1,d) for d in (5,8,9,10)]
    m = rt.prepare_market_data(**inputs({"A":[100]*4, "H":[100]*4}, dates=dates))
    r = simulate(m, policy, quantities={"A":2.,"H":-1.}, stock_borrow=borrow({"H": .365}),
        financing=sofr_config(maintenance_equity_ratio=None))
    assert r.financing_accruals["sofr"].to_list() == [.036,.036,.036,.072,.108]
    assert r.stock_borrow_accruals["amount"].to_list() == pytest.approx([.1]*5)
    debt = 100
    for factor in (1.0001,1.0001,1.0001,1.0002,1.0003):
        debt = debt*factor+.1
    assert r.daily["debt"][-1] == pytest.approx(debt)
    assert r.daily["equity"][-1] == pytest.approx(200-debt)
    assert r.financing_accruals["opening_cash"].to_list() == [0]*5
    assert r.metadata["borrowing_rate"] is None


def test_extra_collateral_and_independent_interest_bases(inputs, policy):
    dates = [date(2024,1,d) for d in (5,8,9)]
    m = rt.prepare_market_data(**inputs({"A":[100]*3}, dates=dates))
    r = simulate(m, policy, long_short=ls(collateral_multiple=1.5, rebate_rate=.36, rebate_day_count="ACT/360"),
        financing=loan(cash_rate=.365))
    # $100 capital + $100 sale - $150 segregated = $50 unrestricted cash.
    entry = r.valuations.filter(pl.col("phase") == "post_entry").row(0,named=True)
    assert entry["cash"] == 50
    assert entry["restricted_collateral"] == 150
    assert entry["equity"] == 100
    assert r.short_financing_accruals["rebate"].to_list() == pytest.approx([.15]*4)
    cash = 50
    for _ in range(4):
        cash = cash*1.001+.15
    assert r.daily["cash"][-1] == pytest.approx(cash)
    assert r.daily["equity"][-1] == pytest.approx(cash+50)


def test_long_drip_is_free_and_short_obligations_do_not_spend_reserved_cash(inputs, policy):
    fri, mon, tue = date(2024,1,5), date(2024,1,8), date(2024,1,9)
    m = rt.prepare_market_data(**inputs({"A":[100,99,99],"H":[100,99,99]}, dates=[fri,mon,tue],
        dividends=[("long", "A", mon, mon, 1.), ("short", "H", mon, mon, 1.)]))
    r = simulate(m, policy, quantities={"A":2.,"H":-1.}, stock_borrow=borrow({"H": .365}),
        costs=rt.TradeCosts(commission_bps=10.,half_spread_bps=10.,impact_bps=10.),
        dividend_reinvestment=rt.DividendReinvestment(execution="first_close_on_or_after_payment",
            funding="before_debt_repayment", scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    drip = r.trades.filter(pl.col("execution") == "dividend_reinvestment_close")
    assert drip["asset"].to_list() == ["A"]
    assert drip["signed_notional"].item() == 2
    assert drip["trade_cost"].item() == 0
    assert r.dividend_reinvestments["paid_amount"].item() == 2
    assert r.costs.filter(pl.col("component").is_in(["commission","half_spread","impact"]))["amount"].sum() == pytest.approx(.9)
    assert r.daily["equity"][0] == pytest.approx(98.8)  # entry .9 and weekend borrow .3
    report(r)


def test_short_dividend_still_paid_after_cover(inputs):
    dates = [date(2024,1,i) for i in range(2,8)]
    m = rt.prepare_market_data(**inputs({"A":[100,100,98,98,98,98]}, dates=dates,
        dividends=[("d","A",dates[2],dates[4],2.)]))
    r = scheduled(m, dates, {1:{"A":-1},2:{"A":0}},
        costs=rt.TradeCosts(commission_bps=0.,half_spread_bps=0.,impact_bps=0.))
    assert r.daily["equity"].to_list() == [100]*4
    assert r.daily["dividend_liability"].to_list() == [2,2,0,0]
    assert r.events.filter(pl.col("type") == "short_dividend_payment")["cash_delta"].item() == -2


def test_pre_trade_margin_cannot_be_hidden_by_cover(inputs):
    dates = [date(2024,1,i) for i in range(2,7)]
    m = rt.prepare_market_data(**inputs({"A":[100,100,160,90,90]}, dates=dates))
    r = scheduled(m, dates, {1:{"A":-1},2:{"A":0}},long_short=ls(short_margin=.5),
        costs=rt.TradeCosts(commission_bps=0.,half_spread_bps=0.,impact_bps=0.))
    assert r.stop_reason == "maintenance_margin_breach"
    assert r.trades.height == 1
    assert r.rebalances.is_empty()
    assert r.daily.height == 1
    report(r)


def test_signed_target_exposures_and_explicit_dates(inputs):
    dates = [date(2024,1,i) for i in range(2,7)]
    m = rt.prepare_market_data(**inputs({"A":[100,100,110,110,110]}, dates=dates))
    r = scheduled(m, dates, {1:{"A":-.5},2:{"A":.5}}, kind="equity_exposure")
    assert r.positions.filter(pl.col("session") == dates[2])["weight"].item() == pytest.approx(.5)
    assert r.rebalances["actual_gross_leverage"].item() == pytest.approx(.5)
    bad = r.targets.with_columns(pl.col("session").alias("decision_session"))
    with pytest.raises(ValueError, match="prior decision"):
        scheduled(m, dates, {1:{"A":-.5}}, targets=bad)
    with pytest.raises(ValueError, match="require_target"):
        scheduled(m, dates, {1:{"A":-.5}}, policy=rt.RebalancePolicy(execution="scheduled_close",
            sizing="post_cost_equity", fractional_shares=True, terminal_action="mark_only",
            non_session="raise", receivable_policy="reserve", max_asset_weight=1.))


def test_signed_all_cash_and_margin_boundary(inputs, policy):
    m = rt.prepare_market_data(**inputs({"A":[100,100,100]}))
    cash = simulate(m, policy, quantities={"A":0.},stock_borrow=borrow({}))
    assert cash.trades.is_empty()
    assert cash.daily["equity"].to_list() == [100,100]
    assert cash.daily["gross_leverage"].to_list() == [0,0]
    assert cash.daily["margin_required"].to_list() == [0,0]
    equal = simulate(m, policy, quantities={"A":-2.},long_short=ls(short_margin=.5))
    assert equal.status == "complete"
    assert equal.daily["margin_required"].to_list() == [100,100]


def test_signed_plots_keep_numerical_reports(inputs, policy):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = rt.prepare_market_data(**inputs({"A":[100,90,95]}))
    r = simulate(m,policy)
    p = report(r)
    for function in (rt.plots.allocation, rt.plots.exposures, rt.plots.equity, rt.plots.attribution):
        fig, ax = function(p)
        assert fig is ax.figure
        plt.close(fig)


def test_old_long_dividend_does_not_cover_now_short_payer(inputs):
    dates = [date(2024,1,i) for i in range(2,8)]
    m = rt.prepare_market_data(**inputs({"A":[100,100,99,99,99,99]}, dates=dates,
        dividends=[("d","A",dates[2],dates[4],1.)]))
    r = scheduled(m, dates, {1:{"A":1.},2:{"A":-1.}},
        costs=rt.TradeCosts(commission_bps=0.,half_spread_bps=0.,impact_bps=0.),
        dividend_reinvestment=rt.DividendReinvestment(execution="first_close_on_or_after_payment",
            funding="before_debt_repayment", scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    assert r.trades.height == 2
    assert r.positions.filter(pl.col("session") == dates[4])["quantity"].item() == -1.
    assert r.dividend_reinvestments["status"].item() == "payer_now_short"
    assert r.dividend_reinvestments["cash_released"].item() == 1.
    assert r.daily["equity"].to_list() == [100]*4
    report(r)


def test_act360_fee_and_rate_mutation_rejected(inputs, policy):
    dates = [date(2024,1,d) for d in (5,8,9)]
    m = rt.prepare_market_data(**inputs({"A":[100,90,90]}, dates=dates))
    b = rt.StockBorrow(rates={"A": .36},day_count="ACT/360",metadata={"source":"assumed","basis":"modeled"})
    r = simulate(m,policy,stock_borrow=b)
    assert r.stock_borrow_accruals["amount"].to_list() == pytest.approx([.1,.1,.1,.09])
    assert r.daily["equity"].to_list() == pytest.approx([109.7,109.61])
    b.rates["A"] = .5
    with pytest.raises(ValueError,match="changed after construction"):
        simulate(m,policy,stock_borrow=b)


def test_entry_and_terminal_ex_date_entitlement_and_unpaid_liability(inputs, policy):
    dates = [date(2024,1,d) for d in (2,3,4)]
    m = rt.prepare_market_data(**inputs({"A":[100,100,99]},dates=dates,
        dividends=[("entry","A",dates[0],date(2024,1,8),1.),("last","A",dates[-1],date(2024,1,8),1.)]))
    r = simulate(m,policy)
    assert r.daily["equity"].to_list() == [100,100]
    assert r.dividend_liabilities["action_id"].to_list() == ["last"]
    assert r.dividend_liabilities["outstanding"].item() == 1
    assert r.events.filter(pl.col("type") == "short_dividend_payment").is_empty()


def test_reports_reject_benchmark_mismatches_and_comparisons_reject_capital(inputs, policy):
    m = rt.prepare_market_data(**inputs({"A":[100,90,95]}))
    r = simulate(m,policy)
    b = r.daily.select("period_start","session",pl.lit(0.).alias("simple_return"))
    with pytest.raises(ValueError, match="align|interval"):
        report(r, b.with_columns((pl.col("period_start")-pl.duration(days=1)).alias("period_start")))
    with pytest.raises(ValueError, match="initial_capital"):
        rt.compare_performance({"one":report(r),"two":report(simulate(m,policy,initial_capital=200.))})


def test_signed_events_reconstruct_every_closing_balance(inputs):
    dates = [date(2024,1,i) for i in range(2,8)]
    m = rt.prepare_market_data(**inputs({"A":[100,100,90,90,90,90]},dates=dates,
        dividends=[("d","A",dates[2],dates[4],1.)]))
    r = scheduled(m,dates,{1:{"A":-1.},3:{"A":.5}},stock_borrow=borrow({"A":.365}),
        financing=loan(borrowing_rate=.365),long_short=ls(collateral_multiple=1.1))
    # Reconstruct from exported event deltas, independent of engine accumulators.
    for day in dates[1:]:
        events = r.events.filter(pl.col("date") <= day)
        balances = {name: events[f"{name}_delta"].sum() for name in
                    ("cash","debt","receivable","collateral","dividend_liability")}
        balances["cash"] += 100
        position = events["quantity_delta"].sum()
        mark = m.prices.filter(pl.col("session") == day)["close"].item()
        equity = position*mark + balances["cash"]+balances["collateral"]+balances["receivable"]-balances["debt"]-balances["dividend_liability"]
        close = r.valuations.filter((pl.col("session") == day) & pl.col("phase").is_in(["post_entry","close"])).row(0,named=True)
        assert equity == pytest.approx(close["equity"])
        assert balances["cash"] == pytest.approx(close["cash"])
        assert balances["collateral"] == pytest.approx(close["restricted_collateral"])
        assert balances["dividend_liability"] == pytest.approx(close["dividend_liability"])
    # The cover trades at Jan 5 close, so Jan 6 and later borrow fees are zero.
    assert r.stock_borrow_accruals["amount"].to_list() == pytest.approx([.1,.09,0,0])


def test_long_short_example_is_offline_and_preserves_long_entry():
    from examples.long_short_hedge import run_example
    runs, reports, comparisons, costs = run_example()
    a,b = runs["unhedged"],runs["hedged"]
    a.require_complete()
    b.require_complete()
    day = a.positions["session"][0]
    longs = a.positions.filter(pl.col("session") == day).select("asset","quantity")
    assert longs.equals(b.positions.filter((pl.col("session") == day) & pl.col("asset").is_in(longs["asset"].to_list())).select("asset","quantity"))
    assert a.metadata["initial_capital"] == b.metadata["initial_capital"]
    assert "stock_borrow_fee" in costs["component"]
    assert comparisons.diagnostics.is_empty()
