"""Offline Milestone 2 demonstration on the committed synthetic weekday calendar.

Run: python examples/scheduled_rebalancing.py
No network calls. Source prices/actions are synthetic, not investment evidence.
"""
from pathlib import Path

import polars as pl
import research_toolkit as rt

ROOT = Path(__file__).resolve().parents[1]


def run_example(*, lookback=40, warmup_sessions=60, periods_per_year=252,
                gross_leverage=1.25, max_asset_weight=.4, initial_capital=10_000_000.):
    market = rt.load_snapshot(ROOT/"examples"/"snapshots"/"v1"/"equities")
    returns = rt.returns(market, method="simple", basis="total_return", dividend_policy="reinvest_ex_close")
    calendar = market.sessions["session"].to_list()
    annualization, leverage = periods_per_year, gross_leverage
    # First SUPPLIED session in each calendar month, after a warm-up. Dates are
    # explicitly materialized; the engine does not move holidays or infer a schedule.
    executions = [i for i in range(1, len(calendar)-1)
                  if i >= warmup_sessions and calendar[i].month != calendar[i-1].month]
    if not executions:
        raise ValueError("no monthly execution sessions remain after the warm-up")
    allocations, frames = [], []
    for i in executions:
        decision = calendar[i-1]
        allocation = rt.inverse_volatility_weights(returns, decision_session=decision,
            lookback=lookback, periods_per_year=annualization, max_asset_weight=max_asset_weight)
        allocations.append(allocation)
        frames.append(allocation.weights.with_columns(pl.lit(decision).alias("decision_session"),
            pl.lit(calendar[i]).alias("session"), pl.lit(leverage).alias("gross_leverage"))
            .select("decision_session", "session", "asset", "weight", "gross_leverage"))
    targets = pl.concat(frames)
    costs = rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=3.)
    financing = rt.Financing(cash_rate=.02, borrowing_rate=.06, day_count="ACT/365F",
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")
    policy = rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity",
        fractional_shares=True, terminal_action="mark_only", non_session="raise",
        receivable_policy="reserve", max_asset_weight=max_asset_weight)
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=initial_capital,
        entry_session=calendar[executions[0]], end_session=calendar[-1],
        policy=policy, costs=costs, financing=financing).require_complete()
    hold = rt.buy_and_hold(market, weights=allocations[0].weights, initial_capital=initial_capital,
        entry_session=calendar[executions[0]], end_session=calendar[-1],
        policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
            initial_gross_leverage=leverage, terminal_action="mark_only"), costs=costs, financing=financing).require_complete()
    options = dict(periods_per_year=annualization, risk_free_annual_effective=.03,
                   minimum_acceptable_return_annual_effective=0.)
    report, hold_report = rt.performance(result, **options), rt.performance(hold, **options)
    rolling = rt.rolling_risk(result, window=40, periods_per_year=annualization, risk_free_annual_effective=.03)
    risk = rt.risk_contributions(returns, weights=allocations[-1].weights, gross_leverage=leverage,
        decision_session=calendar[executions[-1]-1], lookback=lookback, periods_per_year=annualization)
    assert (result.diagnostics["residual"].abs() <= result.diagnostics["tolerance"]).all()
    return dict(result=result, hold=hold, report=report, hold_report=hold_report,
                allocations=allocations, risk=risk, rolling=rolling)


def main():
    outputs = run_example()
    result = outputs["result"]
    print(result.targets)
    print(result.rebalances)
    print(result.turnover)
    print(outputs["report"].summary)
    print(outputs["hold_report"].summary)
    print(outputs["risk"].values)
    output = ROOT/"artifacts"/"milestone2"
    output.mkdir(parents=True, exist_ok=True)
    result.daily.write_csv(output/"daily.csv")
    result.trades.write_csv(output/"trades.csv")
    result.rebalances.write_csv(output/"rebalances.csv")
    result.turnover.write_csv(output/"turnover.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, obj in (("equity", outputs["report"]), ("allocation", outputs["report"]),
                      ("exposures", outputs["report"]), ("turnover", result),
                      ("rolling_risk", outputs["rolling"]), ("risk_contributions", outputs["risk"])):
        fig, ax = getattr(rt.plots, name)(obj)
        fig.savefig(output/f"{name}.png", dpi=130)
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
