"""Corporate actions and security lifecycles on synthetic securities, fully offline.

Run: python examples/corporate_actions.py

A signed portfolio holds a parent that spins off a child (delayed delivery), a
target acquired for stock and cash (delayed cash settlement), and a short in a
target acquired for cash (the short owes the consideration). All securities,
prices, terms, rates and costs are invented example inputs, not market data.
"""
from datetime import date, datetime, time, timedelta, timezone

import polars as pl
import research_toolkit as rt

SESSIONS = [date(2024, 2, 1) + timedelta(days=i) for i in range(28)
            if (date(2024, 2, 1) + timedelta(days=i)).weekday() < 5][:15]
ANNOUNCED = datetime(2024, 1, 15, tzinfo=timezone.utc)


def synthetic_market():
    s = SESSIONS
    paths = {  # None means no quote is expected (before listing or after termination).
        "PARENT": [100, 101, 102, 101, 84, 85, 86, 85, 86, 87, 88, 87, 88, 89, 90],
        "CHILD": [None]*4 + [17, 17.2, 17.1, 17.4, 17.6, 17.5, 17.8, 18, 17.9, 18.1, 18.2],
        "SELLER": [40, 40.5, 41, 41.2, 41.5, 41.8, 42, 42.1, 42.3, None, None, None, None, None, None],
        "BUYER": [45, 45.5, 46, 46.2, 46.5, 46.8, 47, 47.1, 47.3, 47.5, 47.2, 47.4, 47.6, 47.8, 48],
        "SHORTED": [50, 50.2, 50.1, 51.5, 51.8, 51.9, None, None, None, None, None, None, None, None, None],
    }
    prices = pl.DataFrame([{"session": d, "asset": a, "close": float(p)} for a, path in paths.items()
                           for d, p in zip(s, path) if p is not None],
                          schema={"session": pl.Date, "asset": pl.String, "close": pl.Float64})
    sessions = pl.DataFrame({"session": s, "close_at": [datetime.combine(d, time(21), tzinfo=timezone.utc) for d in s]})
    lifecycle = rt.corporate_action_inputs(
        securities=[
            dict(asset="PARENT", security_type="equity", first_session=s[0], unquoted_valuation="none", source="example"),
            dict(asset="CHILD", security_type="equity", first_session=s[4], unquoted_valuation="none", source="example"),
            dict(asset="SELLER", security_type="equity", first_session=s[0], last_session=s[8],
                 terminated_date=s[9], unquoted_valuation="none", source="example"),
            dict(asset="BUYER", security_type="equity", first_session=s[0], unquoted_valuation="none", source="example"),
            dict(asset="SHORTED", security_type="equity", first_session=s[0], last_session=s[5],
                 terminated_date=s[6], unquoted_valuation="none", source="example")],
        aliases=[dict(asset="PARENT", ticker="PRNT", exchange="XEXA", valid_from=s[0], valid_to=s[2], source="example"),
                 dict(asset="PARENT", ticker="PRNX", exchange="XEXA", valid_from=s[3], source="example")],
        actions=[
            dict(action_id="SPIN", action_type="distribution", source_asset="PARENT", announced_at=ANNOUNCED,
                 effective_date=s[4], record_date=s[4], entitlement_basis="ex_date", provider_label="split",
                 classification_source="example issuer notice: spin-off, not a split"),
            dict(action_id="CASHDEAL", action_type="cash_acquisition", source_asset="SHORTED", announced_at=ANNOUNCED,
                 effective_date=s[6], entitlement_basis="effective_date", classification_source="example merger terms"),
            dict(action_id="STOCKDEAL", action_type="stock_acquisition", source_asset="SELLER", announced_at=ANNOUNCED,
                 effective_date=s[9], entitlement_basis="effective_date", classification_source="example merger terms")],
        legs=[
            dict(action_id="SPIN", leg_id="child", leg_type="security", asset="CHILD", units_per_source_share=1.,
                 due_date=s[7], fraction_policy="fractional"),
            dict(action_id="CASHDEAL", leg_id="cash", leg_type="cash", cash_per_source_share=52., due_date=s[8]),
            dict(action_id="STOCKDEAL", leg_id="stock", leg_type="security", asset="BUYER", units_per_source_share=.8,
                 due_date=s[9], fraction_policy="cash_in_lieu", cash_in_lieu_price=47.5),
            dict(action_id="STOCKDEAL", leg_id="cash", leg_type="cash", cash_per_source_share=4., due_date=s[11])])
    metadata = {"source": "synthetic corporate-action example", "retrieved_at": "2024-03-01T00:00:00Z",
                "currency": "USD", "asset_currencies": {a: "USD" for a in paths},
                "calendar": "synthetic_weekdays", "calendar_version": "example", "timezone": "America/New_York",
                "price_basis": "raw", "frequency": "1d", "coverage_start": s[0].isoformat(),
                "coverage_end": s[-1].isoformat(), "actions_complete": True, "dividend_basis": "post_split_share"}
    empty_splits = pl.DataFrame(schema={"action_id": pl.String, "asset": pl.String, "effective_session": pl.Date, "ratio": pl.Float64})
    empty_dividends = pl.DataFrame(schema={"action_id": pl.String, "asset": pl.String, "ex_session": pl.Date,
                                           "pay_date": pl.Date, "cash_per_share": pl.Float64})
    return rt.prepare_market_data(prices=prices, sessions=sessions, splits=empty_splits, dividends=empty_dividends,
                                  metadata=metadata, **lifecycle)


def run_example():
    market = synthetic_market()
    result = rt.buy_and_hold(
        market, quantities={"PARENT": 500., "SELLER": 1000., "SHORTED": -800.},
        initial_capital=100_000., entry_session=SESSIONS[0], end_session=SESSIONS[-1],
        policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
                                initial_gross_leverage=1., terminal_action="mark_only"),
        costs=rt.TradeCosts(commission_bps=1., half_spread_bps=2., impact_bps=0.),
        financing=rt.Financing(cash_rate=.02, borrowing_rate=.06, day_count="ACT/365F",
                               maintenance_equity_ratio=None, on_breach="stop", cash_sweep="repay_debt"),
        long_short=rt.LongShortPolicy(collateral_multiple=1.02, long_margin=.25, short_margin=.3,
                                      rebate_rate=0., rebate_day_count="ACT/360"),
        stock_borrow=rt.StockBorrow(rates={"SHORTED": .03}, day_count="ACT/360",
                                    metadata={"source": "example borrow-fee assumption", "basis": "modeled"}),
        corporate_actions=rt.CorporateActionPolicy(
            distributed_securities="retain", short_obligations="cash_at_delivery_close",
            pending_claim_margin=1., obligation_margin=.3, warrant_margin=1., reorganization_fee=0.))
    report = rt.performance(result.require_complete(), periods_per_year=252, risk_free_annual_effective=0.,
                            minimum_acceptable_return_annual_effective=0.)
    balances = result.daily.select("session", "equity", "cash", "debt", "restricted_collateral",
                                   "pending_cash_receivable", "pending_cash_payable", "pending_security_receivable")
    audit = result.corporate_actions.select("date", "action_id", "stage", "asset", "entitled_quantity",
                                            "quantity_delta", "cash_amount", "value", "pnl", "valuation")
    attribution = report.attribution
    status = rt.security_status(market).filter(pl.col("asset").is_in(["PARENT", "SHORTED"])
                                               & pl.col("session").is_in([SESSIONS[2], SESSIONS[3], SESSIONS[6]]))
    parent_returns = rt.security_returns(market).values.filter(pl.col("asset") == "PARENT")
    return market, result, report, {"balances": balances, "audit": audit, "attribution": attribution,
                                    "status": status, "parent_returns": parent_returns}


def main():
    _, _, _, tables = run_example()
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=160):
        for name, table in tables.items():
            print(f"\n{name}\n{table}")


if __name__ == "__main__":
    main()
