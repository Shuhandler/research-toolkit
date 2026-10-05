"""Fixed currency baskets and exact decision-dated execution models."""
from datetime import date, datetime, time, timedelta, timezone
from dataclasses import replace

import polars as pl
import pytest
import research_toolkit as rt
from test_long_short import scheduled, borrow

ZERO = rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.)
DATES = [date(2024, 1, 2) + timedelta(days=i) for i in range(9)]


def run(market, baskets, **kwargs):
    kwargs.setdefault("stock_borrow", borrow({a:0. for b in baskets.values() for a,v in b.items() if v < 0}))
    return scheduled(market, DATES, baskets, kind="target_notional", **kwargs)


def cost_model(decision, *, assets=("A",), adv=10_000., **metadata):
    liq = rt.LiquidityResult(pl.DataFrame({"asset": list(assets), "daily_volatility": [.02]*len(assets),
        "dollar_adv": [float(adv)]*len(assets), "n_obs": [2]*len(assets)}), dict(
            source="synthetic estimates", currency="USD", volatility_unit="daily_decimal",
            adv_unit="currency_per_trading_day", sample_start=DATES[0].isoformat(),
            sample_end=(decision-timedelta(days=1)).isoformat(), decision_session=decision.isoformat(),
            decision_at=datetime.combine(decision, time(), timezone.utc).isoformat(), **metadata))
    return rt.SquareRootImpactCosts(liquidity=liq, commission_bps=2., commission_per_share=.01,
        half_spread_bps=3., impact_coefficient=.5)


def dated_run(market, baskets, models, **kwargs):
    return run(market, baskets, costs=models, entry_session=DATES[min(baskets)], **kwargs)


def test_fixed_gross_gains_losses_costs_and_drift(inputs):
    m = rt.prepare_market_data(**inputs({"A": [10,10,12,9,9,9,9,9,9],
        "H": [10,10,9,12,12,12,12,12,12]}, dates=DATES))
    r = run(m, {1:{"A":100,"H":-100}, 2:{"A":100,"H":-100}, 3:{"A":100,"H":-100}},
        stock_borrow=borrow({"H":0.}))
    r.require_complete()
    assert r.target_executions.group_by("session").agg(pl.col("actual_notional").abs().sum())["actual_notional"].to_list() == pytest.approx([200]*3)
    assert r.rebalances["gross_traded_notional"].to_list() == pytest.approx([30, 25+100/3])
    assert r.rebalances["equity_after"].to_list() == pytest.approx([127.7,127.7-25-100/3-(25+100/3)*.01])
    assert r.valuations.filter(pl.col("phase")=="post_entry")["equity"].item() == 98
    assert r.target_executions["notional_residual"].abs().max() < 1e-10
    assert r.costs["amount"].sum() == pytest.approx(r.trades["trade_cost"].sum())
    assert r.diagnostics["residual"].abs().max() < 1e-8


def test_execution_price_not_decision_price_and_no_preceding_return(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10,20,30,30,30,30,30,30,30]}, dates=DATES))
    r = run(m, {1:{"A":100}}, costs=ZERO)
    assert r.trades["signed_quantity"].item() == 5
    assert r.daily["pnl"][0] == 50
    assert r.daily["period_start"][0] == DATES[1]
    assert r.target_executions["decision_session"].item() == DATES[0]


def test_covers_crossings_exits_and_no_trades(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10]*9}, dates=DATES))
    r = run(m, {1:{"A":-100},2:{"A":-50},3:{"A":-50},4:{"A":100},5:{"A":0}})
    assert r.trades["signed_notional"].to_list() == [-100,50,150,-100]
    assert r.trades["trade_cost"].to_list() == [1,.5,1.5,1]
    assert r.target_executions["actual_notional"].to_list() == [-100,-50,-50,100,0]
    assert r.target_executions["signed_quantity"].to_list() == [-10,5,0,15,-10]
    assert r.target_executions["trade_id"][2] is None
    assert r.daily["equity"][-1] == 96


def test_split_dividend_and_drip_before_next_target(inputs):
    # Ten shares split to twenty; $1/share dividend buys five more at $4.
    m = rt.prepare_market_data(**inputs({"A":[10,10,4,4,4,4,4,4,4]}, dates=DATES,
        splits=[("s","A",DATES[2],2.)], dividends=[("d","A",DATES[2],DATES[2],1.)]))
    r = run(m, {1:{"A":100},3:{"A":100}}, costs=ZERO, stock_borrow=borrow({}),
        dividend_reinvestment=rt.DividendReinvestment(execution="first_close_on_or_after_payment",
            funding="before_debt_repayment", scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    assert r.target_executions["pre_trade_quantity"].to_list() == [0,25]
    assert r.target_executions["signed_quantity"].to_list() == [10,0]
    assert r.dividend_reinvestments["trade_cost"].sum() == 0
    assert r.daily["equity"].to_list() == pytest.approx([100]*7)
    # Without DRIP the actual split holdings are twenty and scheduled order is $20.
    plain = run(m, {1:{"A":100},2:{"A":100}}, costs=ZERO, stock_borrow=borrow({}))
    assert plain.target_executions["pre_trade_quantity"].to_list() == [0,20]
    assert plain.trades["signed_notional"].to_list() == [100,20]
    assert plain.daily["equity"].to_list() == pytest.approx([100]*7)


def test_dated_models_cost_components_and_continuous_ledger(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10]*9}, dates=DATES))
    models = {DATES[2]:cost_model(DATES[2]), DATES[3]:cost_model(DATES[3],adv=2500)}
    r = dated_run(m, {3:{"A":-100},4:{"A":0}}, models)
    assert r.execution_costs["impact"].to_list() == pytest.approx([.1,.2])
    assert r.execution_costs["commission"].to_list() == pytest.approx([.12,.12])
    assert r.execution_costs["half_spread"].to_list() == pytest.approx([.03,.03])
    assert r.execution_costs["total_cost"].to_list() == pytest.approx([.25,.35])
    assert r.daily["equity"][-1] == pytest.approx(99.4)
    assert r.costs["amount"].sum() == pytest.approx(.6)
    assert r.execution_costs["model_id"].to_list() == [v.model_id for v in models.values()]
    assert r.cost_model_selections["status"].to_list() == ["executed"]*2
    assert r.metadata["cost_models"][1]["liquidity"]["sample_end"] == DATES[2].isoformat()
    assert r.diagnostics["residual"].abs().max() < 1e-8


@pytest.mark.parametrize("issue", ["missing", "extra", "wrong_key", "future", "late", "currency", "exit_asset", "mutated"])
def test_dated_models_reject_bad_coverage_and_information(inputs, issue):
    m = rt.prepare_market_data(**inputs({"A":[10]*9,"H":[10]*9}, dates=DATES))
    models = {d:cost_model(d,assets=("A","H")) for d in [DATES[2],DATES[3]]}
    if issue == "missing": models.pop(DATES[3])
    elif issue == "extra": models[DATES[4]] = cost_model(DATES[4],assets=("A","H"))
    elif issue == "exit_asset": models[DATES[3]] = cost_model(DATES[3],assets=("H",))
    elif issue == "mutated": models[DATES[3]].liquidity.metadata["source"] = "changed"
    else:
        changes = {"wrong_key":{"decision_session":DATES[4].isoformat()},
            "future":{"sample_end":DATES[4].isoformat(), "decision_session":DATES[5].isoformat()},
            "late":{"decision_at":"2024-02-01T00:00:00+00:00"},
            "currency":{"currency":"EUR"}}[issue]
        v = models[DATES[3]]
        models[DATES[3]] = replace(v, liquidity=replace(v.liquidity, metadata=v.liquidity.metadata | changes))
    with pytest.raises(ValueError):
        dated_run(m, {3:{"A":-100,"H":0},4:{"A":0,"H":100}}, models)


def test_margin_stop_keeps_unexecuted_targets_and_models(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10,10,10,10,16,16,16,16,16]}, dates=DATES))
    models = {d:cost_model(d) for d in [DATES[2],DATES[3]]}
    r = dated_run(m, {3:{"A":-100},4:{"A":0}}, models)
    assert r.status != "complete"
    assert r.stop_session == DATES[4]
    assert r.cost_model_selections["status"].to_list() == ["executed","not_executed"]
    assert r.target_executions.height == 1
    assert r.targets.height == 2
    with pytest.raises(ValueError): r.require_complete()


def test_post_trade_margin_stop_and_unfundable_entry(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10]*9}, dates=DATES))
    r = run(m, {1:{"A":100},2:{"A":400}})
    assert r.status != "complete"
    assert r.target_executions["actual_notional"].to_list() == [100,400]
    assert r.daily["equity"][0] == 96
    with pytest.raises(ValueError, match="margin"):
        run(m, {1:{"A":400}})
    with pytest.raises(ValueError, match="equity"):
        run(m, {1:{"A":10_000}})


def test_timing_terminal_and_ambiguous_modes_rejected(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10]*9}, dates=DATES))
    with pytest.raises(ValueError): run(m, {1:{"A":100},8:{"A":100}})
    targets = pl.DataFrame({"decision_session":[DATES[1]],"session":[DATES[1]],"asset":["A"],"target_notional":[100.]})
    with pytest.raises(ValueError): run(m, {1:{"A":100}}, targets=targets)
    with pytest.raises(ValueError): run(m, {1:{"A":100}}, targets=targets.with_columns(pl.lit(1.).alias("quantity")))


@pytest.mark.parametrize("kind,baskets", [
    ("quantity",{3:{"A":-10},4:{"A":5}}),
    ("equity_exposure",{3:{"A":-.5},4:{"A":.5}}),
    ("target_notional",{3:{"A":-100},4:{"A":50}}),
])
def test_static_and_dated_equivalence_for_all_signed_modes(inputs, kind, baskets):
    m = rt.prepare_market_data(**inputs({"A":[10,10,10,10,12,12,12,12,12]}, dates=DATES))
    common = dict(kind=kind,entry_session=DATES[3])
    static = scheduled(m,DATES,baskets,costs=cost_model(DATES[2]),**common)
    dated = scheduled(m,DATES,baskets,costs={d:cost_model(d) for d in [DATES[2],DATES[3]]},**common)
    for field in ["daily","trades","costs","events","positions","turnover"]:
        assert getattr(static,field).equals(getattr(dated,field))


def test_dated_model_with_unchanged_holdings_and_free_drip(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10,10,10,10,9,9,9,9,9]},dates=DATES,
        dividends=[("d","A",DATES[4],DATES[4],1.)]))
    models = {d:cost_model(d) for d in [DATES[2],DATES[5]]}
    r = dated_run(m,{3:{"A":100},6:{"A":100}},models,stock_borrow=borrow({}),
        dividend_reinvestment=rt.DividendReinvestment(execution="first_close_on_or_after_payment",
            funding="before_debt_repayment",scheduled_collision="rebalance_only",terminal_action="hold_cash"))
    assert r.dividend_reinvestments["trade_cost"].sum() == 0
    assert r.execution_costs.height == 1
    assert r.target_executions["signed_quantity"][1] == 0
    assert r.cost_model_selections["status"].to_list() == ["executed"]*2


def test_estimated_snapshot_lag_and_long_only_dated_models(inputs):
    m = rt.prepare_market_data(**inputs({"A":[10,11,10,12,9,11,10,12,13]},dates=DATES))
    returns = rt.returns(m,method="simple",basis="price")
    volumes = m.sessions.select("session",pl.col("close_at").alias("available_at")).with_columns(
        pl.lit("A").alias("asset"),pl.lit(10_000.).alias("dollar_volume")).select(
            "session","asset","dollar_volume","available_at")
    def estimate(d, r=returns, v=volumes):
        return rt.estimate_liquidity(r,v,decision_session=d,lookback=2,metadata={"source":"synthetic","currency":"USD"})
    models = {d:replace(cost_model(d),liquidity=estimate(d)) for d in [DATES[3],DATES[4]]}
    changed = replace(returns,values=returns.values.with_columns(pl.when(pl.col("session")>=DATES[3])
        .then(.99).otherwise(pl.col("simple_return")).alias("simple_return")))
    changed_volumes = volumes.with_columns(pl.when(pl.col("session")>=DATES[3]).then(1.)
        .otherwise(pl.col("dollar_volume")).alias("dollar_volume"))
    assert estimate(DATES[3]).estimates.equals(estimate(DATES[3],changed,changed_volumes).estimates)
    assert models[DATES[3]].liquidity.metadata["sample_end"] == DATES[2].isoformat()
    targets=pl.DataFrame({"decision_session":[DATES[3],DATES[4]],"session":[DATES[4],DATES[5]],
        "asset":["A"]*2,"weight":[1.,1.],"gross_leverage":[1.,1.]})
    # Existing long-only weight schema and accounting also support dated models.
    from test_long_short import loan
    r=rt.scheduled_rebalance(m,targets=targets,initial_capital=100.,entry_session=DATES[4],end_session=DATES[-1],
        policy=rt.RebalancePolicy(execution="scheduled_close",sizing="post_cost_equity",fractional_shares=True,
            terminal_action="mark_only",non_session="raise",receivable_policy="require_target",max_asset_weight=1.),
        financing=loan(),costs=models)
    assert r.status == "complete"
    assert r.cost_model_selections["decision_session"].to_list() == [DATES[3],DATES[4]]
    assert r.diagnostics["residual"].abs().max() < 1e-8


def test_cost_model_identity_includes_assumptions_and_freezes_mappings():
    base = cost_model(DATES[2])
    different = replace(base,half_spread_bps={"A":4.})
    assert base.snapshot_id == different.snapshot_id
    assert base.model_id != different.model_id
    different.half_spread_bps["A"] = 5.
    with pytest.raises(ValueError,match="parameters changed"):
        rt.estimate_trade_costs(pl.DataFrame({"asset":["A"],"signed_notional":[100.],"reference_price":[10.]}),
            costs=different,execution_session=DATES[3])


def test_fixed_dollar_notebook_offline(monkeypatch):
    import json
    from pathlib import Path
    import sys
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("IPython")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    notebook=json.loads((Path(__file__).resolve().parents[1]/"examples/fixed_dollar_rebalancing.ipynb").read_text())
    namespace={}
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code": exec("".join(cell["source"]),namespace)
        r=namespace["run"]
        assert r.daily["equity"].n_unique() > 1
        assert r.cost_model_selections["model_id"].n_unique() == 3
        assert namespace["gross"]["executed_gross"].to_list() == pytest.approx([2_000_000]*3)
        assert r.execution_costs["total_cost"].sum() == pytest.approx(r.trades["trade_cost"].sum())
    finally:
        plt.close("all")
