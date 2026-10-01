"""Offline comparison of cash/debt treatment and payment-funded reinvestment.

Uses self-authored synthetic weekday data, not historical market observations or
an exchange calendar. Rates, closing fills and proportional costs are assumptions.
"""
from pathlib import Path

import polars as pl
import research_toolkit as rt

ROOT = Path(__file__).resolve().parents[1]


def run_example(*, initial_capital=100_000_000., gross_leverage=1.5):
    market = rt.load_snapshot(ROOT / "examples/snapshots/v1/equities")
    cases = {
        "cash_sweep": None,
        "debt_first": rt.DividendReinvestment(
            execution="first_close_on_or_after_payment", funding="after_debt_repayment",
            scheduled_collision="rebalance_only", terminal_action="hold_cash"),
        "reinvest_first": rt.DividendReinvestment(
            execution="first_close_on_or_after_payment", funding="before_debt_repayment",
            scheduled_collision="rebalance_only", terminal_action="hold_cash"),
    }
    runs, reports = {}, {}
    for name, reinvestment in cases.items():
        run = rt.buy_and_hold(
            market, weights=rt.equal_weights(market.prices["asset"].unique()),
            initial_capital=initial_capital,
            entry_session=market.sessions["session"][0], end_session=market.sessions["session"][-1],
            policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
                fractional_shares=True, initial_gross_leverage=gross_leverage, terminal_action="mark_only"),
            costs=rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=3.),
            financing=rt.Financing(cash_rate=.02, borrowing_rate=.06, day_count="ACT/365F",
                maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt"),
            dividend_reinvestment=reinvestment,
        ).require_complete()
        runs[name] = run
        reports[name] = rt.performance(run, periods_per_year=252, risk_free_annual_effective=.02,
            minimum_acceptable_return_annual_effective=0.)
    summary = pl.DataFrame([
        dict(scenario=name, ending_equity=run.daily["equity"][-1],
             compounded_return=run.daily["compounded_return"][-1],
             ending_debt=run.daily["debt"][-1], trades=run.trades.height,
             reinvested_notional=run.dividend_reinvestments["signed_notional"].sum(),
             reinvestment_cost=run.dividend_reinvestments["trade_cost"].sum())
        for name, run in runs.items()
    ])
    return runs, reports, summary


def main():
    runs, reports, summary = run_example()
    print(summary)
    destination = ROOT / "artifacts/dividend_reinvestment"
    destination.mkdir(parents=True, exist_ok=True)
    summary.write_csv(destination / "comparison.csv")
    for name, run in runs.items():
        run.dividend_reinvestments.write_csv(destination / f"{name}_payments.csv")
        run.daily.write_csv(destination / f"{name}_daily.csv")
    # Plotting is optional and happens only in this presentation entry point.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = None
    for name, report in reports.items():
        fig, ax = rt.plots.equity(report, ax=ax)
        ax.lines[-1].set_label(name.replace("_", " "))
    ax.legend(fontsize="small")
    ax.set_title("Synthetic portfolio: paid-dividend funding policies (initial leverage 1.5×)")
    fig.savefig(destination / "equity.png", dpi=160)
    plt.close(fig)
    fig, _ = rt.plots.turnover(runs["reinvest_first"])
    fig.savefig(destination / "turnover.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
