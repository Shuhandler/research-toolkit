"""Runnable offline milestone 1A/1B example. All data and rates are synthetic.

Run from an installed checkout: python examples/unlevered_buy_and_hold.py
This is a numerical smoke example, not the one-year acceptance notebook (1D).
"""

from datetime import date, datetime, time, timezone

import polars as pl

import research_toolkit as rt


def make_market():
    """Prepare the synthetic snapshot shared by both buy-and-hold examples."""
    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    prices = pl.DataFrame({
        "session": sessions * 2,
        "asset": ["SYNTH_A"] * 3 + ["SYNTH_B"] * 3,
        "close": [100.0, 49.0, 50.0, 50.0, 51.0, 52.0],
    })
    calendar = pl.DataFrame({
        "session": sessions,
        "close_at": [datetime.combine(d, time(21), tzinfo=timezone.utc) for d in sessions],
    })
    splits = pl.DataFrame({
        "action_id": ["split-A"], "asset": ["SYNTH_A"],
        "effective_session": [sessions[1]], "ratio": [2.0],
    })
    dividends = pl.DataFrame({
        "action_id": ["dividend-A"], "asset": ["SYNTH_A"],
        "ex_session": [sessions[1]], "pay_date": [sessions[2]], "cash_per_share": [1.0],
    })
    return rt.prepare_market_data(
        prices=prices, sessions=calendar, splits=splits, dividends=dividends,
        metadata={
            "source": "synthetic example v1", "retrieved_at": "2024-02-01T00:00:00Z",
            "currency": "USD", "asset_currencies": {"SYNTH_A": "USD", "SYNTH_B": "USD"},
            "calendar": "supplied synthetic sessions", "calendar_version": "1",
            "timezone": "America/New_York", "price_basis": "raw", "frequency": "1d",
            "coverage_start": "2024-01-02", "coverage_end": "2024-01-04",
            "actions_complete": True, "dividend_basis": "post_split_share",
        },
        missing="raise",
    )


def main():
    market = make_market()
    sessions = market.sessions["session"].to_list()
    # These price-only returns intentionally retain the split jump. The ledger
    # below instead accounts for the extra shares and cash dividend explicitly.
    price_returns = rt.returns(market, method="simple", basis="price")
    print(price_returns.values)
    print(rt.cumulative_returns(price_returns, method="compound").values)

    result = rt.buy_and_hold(
        market,
        weights=rt.equal_weights(["SYNTH_A", "SYNTH_B"]),
        initial_capital=1_000.0,
        entry_session=sessions[0], end_session=sessions[-1],
        policy=rt.BuyHoldPolicy(
            execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
            initial_gross_leverage=1.0, terminal_action="mark_only",
        ),
        costs=rt.TradeCosts(commission_bps=5.0, half_spread_bps=2.0, impact_bps=0.0),
        cash_rate=0.0, cash_day_count="ACT/365F",
    )
    print(result.daily.select("session", "pnl", "simple_return", "cumulative_simple_return",
                               "compounded_return", "equity", "cash", "dividend_receivable"))
    print(result.trades)
    print(result.receivables)
    print(result.diagnostics)


if __name__ == "__main__":
    main()
