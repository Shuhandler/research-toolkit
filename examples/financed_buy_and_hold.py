"""Offline 1C example: explicit financing, initial leverage, and a stopped run.

Run: python examples/financed_buy_and_hold.py
Prices, rates, costs, and the margin threshold are synthetic research assumptions.
"""

import polars as pl

import research_toolkit as rt
from unlevered_buy_and_hold import make_market


def simulate(market, initial_leverage):
    return rt.buy_and_hold(
        market, weights=rt.equal_weights(["SYNTH_A", "SYNTH_B"]), initial_capital=1_000.,
        entry_session=market.sessions["session"][0], end_session=market.sessions["session"][-1],
        policy=rt.BuyHoldPolicy(
            execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
            initial_gross_leverage=initial_leverage, terminal_action="mark_only",
        ),
        costs=rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=0.),
        financing=rt.Financing(
            cash_rate=.02, borrowing_rate=.08, day_count="ACT/365F",
            maintenance_equity_ratio=.4, on_breach="stop", cash_sweep="repay_debt",
        ),
    )


def main():
    market = make_market()
    for initial_leverage in (1., 2.):
        result = simulate(market, initial_leverage).require_complete()
        print({"initial_leverage": initial_leverage, "status": result.status})
        print(result.daily.select("session", "equity", "pnl", "debt", "cash", "gross_leverage"))
        print(result.costs)

    # Prepare a separate, clearly labeled synthetic price shock. Revalidation
    # creates a new input identity; the original snapshot remains unchanged.
    shocked = rt.prepare_market_data(
        prices=market.prices.with_columns(
            pl.when(pl.col("session") > market.sessions["session"][0])
            .then(pl.col("close") * .7).otherwise(pl.col("close")).alias("close")
        ),
        sessions=market.sessions, splits=market.splits, dividends=market.dividends,
        metadata=market.metadata | {"source": "synthetic example v1 with 30% price shock"},
    )
    partial = simulate(shocked, 2.)
    print({"status": partial.status, "stop_reason": partial.stop_reason,
           "requested_end": partial.metadata["end_session"],
           "actual_end": partial.metadata["actual_end_session"]})
    print(partial.daily)
    print(partial.events)
    # partial.require_complete() would raise; these are partial-run records.


if __name__ == "__main__":
    main()
