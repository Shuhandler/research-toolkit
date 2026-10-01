"""Daily buy-and-hold ledger: shares, cash, corporate actions, and trade costs."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version
import math
import platform

import polars as pl

from ._costs import TradeCosts
from ._data import _validated_market
from ._financing import Financing, _below_margin
from ._portfolio import BuyHoldPolicy, _number, _weights
from ._results import BacktestResult


try:
    PACKAGE_VERSION = version("research-toolkit")
except PackageNotFoundError:
    PACKAGE_VERSION = "uninstalled"


F, S, D, I = pl.Float64, pl.String, pl.Date, pl.Int64
UTC = pl.Datetime("us", "UTC")
SCHEMAS = {
    "daily": {"session": D, "period_start": D, "opening_equity": F, "equity": F,
              "pnl": F, "simple_return": F, "log_return": F, "cumulative_pnl": F,
              "cumulative_simple_return": F, "compounded_return": F, "cash": F,
              "debt": F, "dividend_receivable": F, "gross_exposure": F,
              "net_exposure": F, "gross_leverage": F, "drawdown": F,
              "equity_ratio": F, "margin_breached": pl.Boolean,
              "includes_entry_costs": pl.Boolean},
    "positions": {"session": D, "asset": S, "quantity": F, "raw_mark": F,
                  "market_value": F, "weight": F},
    "trades": {"trade_id": S, "session": D, "time": UTC, "asset": S,
               "signed_quantity": F, "reference_price": F, "signed_notional": F,
               "execution": S, "trade_cost": F},
    "costs": {"cost_id": S, "date": D, "time": UTC, "trade_id": S,
              "asset": S, "component": S, "amount": F, "basis": S},
    "events": {"event_id": S, "date": D, "time": UTC, "sequence": I, "phase": S,
               "type": S, "asset": S, "action_id": S, "trade_id": S,
               "quantity_delta": F, "cash_delta": F, "debt_delta": F,
               "receivable_delta": F},
    "valuations": {"session": D, "time": UTC, "phase": S, "market_value": F,
                   "cash": F, "debt": F, "dividend_receivable": F, "equity": F},
    "attribution": {"session": D, "component": S, "asset": S, "pnl": F},
    "receivables": {"session": D, "action_id": S, "asset": S, "ex_session": D,
                    "pay_date": D, "entitled_quantity": F, "amount": F, "outstanding": F},
    "diagnostics": {"session": D, "code": S, "residual": F, "tolerance": F},
}


def _check(actual, expected, scale, context):
    # Dollar tolerance remains below a cent even for large accounts.
    tolerance = min(0.009, 1e-8 + 1e-12 * scale)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"nonfinite accounting value: {context}")
    residual = actual - expected
    if abs(residual) > tolerance:
        raise ArithmeticError(f"{context} reconciliation failed: residual={residual}")
    return residual, tolerance


def buy_and_hold(market, *, weights, initial_capital: float, entry_session: date,
                 end_session: date, policy: BuyHoldPolicy, costs: TradeCosts,
                 cash_rate: float | None = None, cash_day_count: str | None = None,
                 financing: Financing | None = None) -> BacktestResult:
    """Run a fractional-share, raw-close buy-and-hold portfolio with explicit funding.

    Supply ``financing`` for leveraged runs. The existing explicit ``cash_rate``
    and ``cash_day_count`` pair remains supported for unlevered calls; do not mix
    the two configurations. Rates capitalize daily, including nontrading dates.
    Breached margin or nonpositive equity stops at the failure close, preserving
    balances and setting incomplete status. No liquidation or rebalancing occurs.
    Entry costs reduce the first holding interval's P&L, whose denominator is
    original capital. Dividends accrue on ex-date and pay on their actual date.

    All results are numerical Polars tables. See ``docs/api.md`` for the complete
    input and output contracts. This function performs no network or file I/O.
    """
    market = _validated_market(market)
    if market.metadata["price_basis"] != "raw":
        raise ValueError("buy_and_hold requires raw execution prices")
    if not isinstance(policy, BuyHoldPolicy) or not isinstance(costs, TradeCosts):
        raise ValueError("explicit BuyHoldPolicy and TradeCosts objects are required")
    policy.__post_init__()
    capital = _number(initial_capital, "initial_capital", positive=True)
    exposure = policy.initial_gross_leverage
    if financing is not None:
        if not isinstance(financing, Financing):
            raise ValueError("financing must be an explicit Financing object")
        if cash_rate is not None or cash_day_count is not None:
            raise ValueError("do not mix financing with cash_rate/cash_day_count")
        financing.__post_init__()
        rate, borrowing_rate = financing.cash_rate, financing.borrowing_rate
        cash_day_count = financing.day_count
        threshold = financing.maintenance_equity_ratio
    else:
        if exposure > 1:
            raise ValueError("initial leverage above 1 requires explicit Financing")
        rate = _number(cash_rate, "cash_rate")
        if cash_day_count != "ACT/365F":
            raise ValueError("only cash_day_count='ACT/365F' is supported")
        borrowing_rate, threshold = 0.0, None
    if exposure > 1 and threshold is None:
        raise ValueError("borrowing requires maintenance_equity_ratio")
    if exposure and _below_margin(1 / exposure, threshold):
        raise ValueError("initial leverage violates maintenance_equity_ratio")
    if type(entry_session) is not date or type(end_session) is not date:
        raise ValueError("entry_session and end_session must be datetime.date values")
    all_sessions = market.sessions["session"].to_list()
    if entry_session not in all_sessions or end_session not in all_sessions or entry_session >= end_session:
        raise ValueError("entry/end must be supplied sessions with at least one holding interval")
    allocation = _weights(weights, market.prices["asset"].unique().to_list())
    assets = sorted(allocation)
    total_weight = math.fsum(allocation.values())
    resolved_weights = {a: allocation[a] / total_weight for a in assets}
    rates = costs.resolve(assets)
    fee_rate = math.fsum(resolved_weights[a] * math.fsum(rates[c][a] for c in rates)
                         for a in assets)
    gross = capital * exposure / (1 + exposure * fee_rate)
    if not math.isfinite(gross):
        raise ValueError("initial notional is not representable")
    dates = [d for d in all_sessions if entry_session <= d <= end_session]
    closes = dict(market.sessions.select("session", "close_at").iter_rows())
    prices = {(d, a): p for d, a, p in market.prices.iter_rows()}
    split_dates, ex_dates = defaultdict(list), defaultdict(list)
    for action in market.splits.iter_rows(named=True):
        if action["asset"] in allocation:
            split_dates[action["effective_session"]].append(action)
    for action in market.dividends.iter_rows(named=True):
        if action["asset"] in allocation:
            ex_dates[action["ex_session"]].append(action)
    records = {name: [] for name in SCHEMAS}
    quantity = {a: 0.0 for a in assets}
    cash = capital
    debt = 0.0
    receivables = {}
    pending_pnl = defaultdict(list)
    cash_deltas, receivable_deltas, debt_deltas = [], [], []
    quantity_deltas = defaultdict(list)

    def event(day, kind, *, phase="before_close", asset=None, action_id=None,
              trade_id=None, dq=0.0, dc=0.0, dr=0.0, dd=0.0):
        seq = len(records["events"])
        records["events"].append({
            "event_id": f"E{seq:06d}", "date": day,
            "time": closes[day] if phase in {"entry_close", "close"} else None,
            "sequence": seq, "phase": phase, "type": kind, "asset": asset,
            "action_id": action_id, "trade_id": trade_id,
            "quantity_delta": dq, "cash_delta": dc, "debt_delta": dd,
            "receivable_delta": dr,
        })
        cash_deltas.append(dc)
        receivable_deltas.append(dr)
        debt_deltas.append(dd)
        if asset is not None:
            quantity_deltas[asset].append(dq)

    def balances(day, phase):
        values = {a: quantity[a] * prices[day, a] for a in assets}
        mv = math.fsum(values.values())
        receivable = math.fsum(item["outstanding"] for item in receivables.values())
        equity = math.fsum([mv, cash, receivable, -debt])
        if (not all(math.isfinite(v) for v in (mv, cash, receivable, debt, equity))
                or min(cash, receivable, debt) < 0):
            raise ArithmeticError("ledger produced invalid balances")
        if phase != "close" and equity <= 0:
            raise ValueError("initial equity after costs must be positive")
        records["valuations"].append({"session": day, "time": closes[day], "phase": phase,
                                      "market_value": mv, "cash": cash, "debt": debt,
                                      "dividend_receivable": receivable, "equity": equity})
        checks = {
            "cash_reconciliation": (cash, math.fsum([capital, *cash_deltas])),
            "receivable_reconciliation": (receivable, math.fsum(receivable_deltas)),
            "debt_reconciliation": (debt, math.fsum(debt_deltas)),
            "balance_sheet": (equity, math.fsum([*values.values(), cash, receivable, -debt])),
        }
        for a in assets:
            if not math.isclose(quantity[a], math.fsum(quantity_deltas[a]), rel_tol=1e-12, abs_tol=1e-12):
                raise ArithmeticError(f"quantity reconciliation failed for {a}")
        for code, (actual, expected) in checks.items():
            residual, tolerance = _check(actual, expected, max(capital, equity), code)
            records["diagnostics"].append({"session": day, "code": f"{phase}:{code}",
                                            "residual": residual, "tolerance": tolerance})
        return values, receivable, equity

    balances(entry_session, "pre_entry")
    entry_plan = []
    for asset in assets:
        notional = gross * resolved_weights[asset]
        if notional == 0:
            continue
        mark = prices[entry_session, asset]
        qty = notional / mark
        if not math.isfinite(qty) or qty <= 0:
            raise ValueError(f"initial quantity is not representable for {asset}")
        components = {c: notional * rates[c][asset] for c in rates}
        cost = math.fsum(components.values())
        entry_plan.append((asset, notional, mark, qty, components, cost))
    entry_cost = math.fsum(item[5] for item in entry_plan)
    entry_notional = math.fsum(item[1] for item in entry_plan)
    # Fund the whole order basket before any trade event; borrowing is not profit.
    if exposure > 1:
        debt = max(math.fsum([entry_notional, entry_cost, -capital]), 0.0)
        if not math.isfinite(debt) or debt <= 0:
            raise ValueError("initial borrowing is not representable")
        cash += debt
        event(entry_session, "borrowing", phase="entry_close", dc=debt, dd=debt)
    for asset, notional, mark, qty, components, cost in entry_plan:
        trade_id = f"T{len(records['trades']):06d}"
        quantity[asset] = qty
        event(entry_session, "trade", phase="entry_close", asset=asset,
              trade_id=trade_id, dq=qty, dc=-notional)
        records["trades"].append({"trade_id": trade_id, "session": entry_session,
                                    "time": closes[entry_session], "asset": asset,
                                    "signed_quantity": qty, "reference_price": mark,
                                    "signed_notional": notional, "execution": policy.execution,
                                    "trade_cost": cost})
        for component, amount in components.items():
            if amount == 0:
                continue
            records["costs"].append({"cost_id": f"C{len(records['costs']):06d}",
                                     "date": entry_session, "time": closes[entry_session],
                                     "trade_id": trade_id, "asset": asset,
                                     "component": component, "amount": amount, "basis": "modeled"})
            event(entry_session, component, phase="entry_close", asset=asset,
                  trade_id=trade_id, dc=-amount)
            pending_pnl[component, asset].append(-amount)
    cash = math.fsum([capital, debt, -entry_notional, -entry_cost])
    # At full investment, only floating-point subtraction can leave residual cash.
    if exposure >= 1 or cash < 0:
        residual, tolerance = _check(cash, 0.0, capital, "entry_cash_roundoff")
        cash = 0.0
        records["diagnostics"].append({"session": entry_session, "code": "entry_cash_roundoff",
                                       "residual": residual, "tolerance": tolerance})
    previous_values, outstanding, post_entry_equity = balances(entry_session, "post_entry")
    _check(post_entry_equity, capital - entry_cost, capital, "entry_cost_equity")
    actual_gross = math.fsum(previous_values.values())
    if actual_gross and _below_margin(post_entry_equity / actual_gross, threshold):
        raise ValueError("funded entry violates maintenance_equity_ratio")

    def position_rows(day, values, equity):
        for asset in assets:
            records["positions"].append({"session": day, "asset": asset,
                                          "quantity": quantity[asset], "raw_mark": prices[day, asset],
                                          "market_value": values[asset],
                                          "weight": values[asset] / equity if equity > 0 else None})

    position_rows(entry_session, previous_values, post_entry_equity)
    opening_equity, previous_session = capital, entry_session
    cumulative_pnls, simple_returns = [], []
    wealth = 1.0
    peak = max(capital, post_entry_equity)
    day = entry_session + timedelta(days=1)
    session_set = set(dates)
    status, stop_reason, stop_session, stop_time = "complete", None, None, None
    while day <= end_session:
        interest = cash * (rate / 365)
        if interest:
            cash += interest
            event(day, "cash_interest", dc=interest)
            pending_pnl["cash_interest", None].append(interest)
        borrowing_charge = debt * (borrowing_rate / 365)
        if borrowing_charge:
            debt += borrowing_charge
            event(day, "borrowing_interest", dd=borrowing_charge)
            records["costs"].append({"cost_id": f"C{len(records['costs']):06d}",
                                     "date": day, "time": None, "trade_id": None, "asset": None,
                                     "component": "borrowing_interest", "amount": borrowing_charge,
                                     "basis": "modeled"})
            pending_pnl["borrowing_interest", None].append(-borrowing_charge)
        for action in split_dates[day]:
            asset = action["asset"]
            old = quantity[asset]
            quantity[asset] *= action["ratio"]
            if old and (not math.isfinite(quantity[asset]) or quantity[asset] <= 0):
                raise ValueError(f"split quantity is not representable for {asset} on {day}")
            if old:
                event(day, "split", asset=asset, action_id=action["action_id"], dq=quantity[asset] - old)
        for action in ex_dates[day]:
            asset = action["asset"]
            amount = quantity[asset] * action["cash_per_share"]
            if amount:
                receivables[action["action_id"]] = {
                    **action, "entitled_quantity": quantity[asset], "amount": amount, "outstanding": amount,
                }
                event(day, "dividend_accrual", asset=asset, action_id=action["action_id"], dr=amount)
                pending_pnl["dividend_income", asset].append(amount)
        for action_id, item in receivables.items():
            if item["pay_date"] == day and item["outstanding"]:
                amount = item["outstanding"]
                cash += amount
                item["outstanding"] = 0.0
                event(day, "dividend_payment", asset=item["asset"], action_id=action_id, dc=amount, dr=-amount)
        if debt > 0 and cash > 0:
            repayment = min(cash, debt)
            cash -= repayment
            debt -= repayment
            event(day, "debt_repayment", dc=-repayment, dd=-repayment)
        if day in session_set:
            values, outstanding, equity = balances(day, "close")
            for asset in assets:
                pending_pnl["price_pnl", asset].append(values[asset] - previous_values[asset])
            components = {key: math.fsum(amounts) for key, amounts in pending_pnl.items()}
            pnl = equity - opening_equity
            cumulative_pnls.append(pnl)
            simple = pnl / opening_equity
            if not math.isfinite(simple) or (equity > 0 and simple <= -1):
                raise ValueError(f"daily return outside representable positive wealth on {day}")
            simple_returns.append(simple)
            wealth *= 1 + simple
            if not math.isfinite(wealth) or (equity > 0 and wealth <= 0):
                raise ValueError(f"cumulative wealth is not representable on {day}")
            peak = max(peak, equity)
            gross_value = math.fsum(values.values())
            equity_ratio = equity / gross_value if gross_value else None
            margin_breached = _below_margin(equity_ratio, threshold)
            if equity <= 0 or margin_breached:
                status = "stopped"
                stop_reason = "nonpositive_equity" if equity <= 0 else "maintenance_margin_breach"
                stop_session, stop_time = day, closes[day]
            for code, actual, expected in (
                ("pnl_attribution", math.fsum(components.values()), pnl),
                ("cumulative_pnl", math.fsum(cumulative_pnls), equity - capital),
                ("compounded_equity", capital * wealth, equity),
            ):
                residual, tolerance = _check(actual, expected, max(capital, equity), code)
                records["diagnostics"].append({"session": day, "code": code,
                                                "residual": residual, "tolerance": tolerance})
            records["daily"].append({
                "session": day, "period_start": previous_session, "opening_equity": opening_equity,
                "equity": equity, "pnl": pnl, "simple_return": simple,
                "log_return": math.log(equity) - math.log(opening_equity) if equity > 0 else None,
                "cumulative_pnl": math.fsum(cumulative_pnls),
                "cumulative_simple_return": math.fsum(simple_returns), "compounded_return": wealth - 1,
                "cash": cash, "debt": debt, "dividend_receivable": outstanding,
                "gross_exposure": gross_value, "net_exposure": gross_value,
                "gross_leverage": gross_value / equity if equity > 0 else None,
                "equity_ratio": equity_ratio, "margin_breached": margin_breached,
                "drawdown": equity / peak - 1, "includes_entry_costs": previous_session == entry_session,
            })
            for (component, asset), amount in sorted(components.items(), key=lambda x: (x[0][0], x[0][1] or "")):
                records["attribution"].append({"session": day, "component": component,
                                                "asset": asset, "pnl": amount})
            position_rows(day, values, equity)
            for action_id, item in receivables.items():
                records["receivables"].append({"session": day, "action_id": action_id,
                                                **{k: item[k] for k in SCHEMAS["receivables"]
                                                   if k not in {"session", "action_id"}}})
            pending_pnl.clear()
            previous_values, opening_equity, previous_session = values, equity, day
            if status == "stopped":
                event(day, stop_reason, phase="close")
                break
        if day == end_session:
            break
        day += timedelta(days=1)
    metadata = {
        "snapshot_id": market.snapshot_id, "source": deepcopy(market.metadata),
        "currency": market.metadata["currency"], "frequency": "1d", "return_basis": "net_equity",
        "initial_capital": capital, "entry_session": entry_session.isoformat(),
        "end_session": end_session.isoformat(), "weights": allocation, "resolved_weights": resolved_weights,
        "actual_end_session": previous_session.isoformat(), "status": status,
        "stop_reason": stop_reason, "stop_session": stop_session.isoformat() if stop_session else None,
        "stop_time": stop_time.isoformat() if stop_time else None,
        "policy": asdict(policy), "cost_rates": rates, "cost_basis": "modeled",
        "cash_rate": rate, "cash_day_count": cash_day_count, "cash_capitalization": "daily_calendar_date",
        "financing": asdict(financing) if financing else None,
        "borrowing_rate": borrowing_rate, "maintenance_equity_ratio": threshold,
        "margin_monitoring": "session_close", "margin_comparison_relative_tolerance": 1e-12,
        "return_denominator": "first_interval_pre_entry_capital_then_prior_closing_equity",
        "dividend_policy": "ex_date_receivable_pay_date_cash_no_reinvestment",
        "cash_sweep": financing.cash_sweep if financing else "hold_cash_no_debt",
        "settlement": "immediate", "taxes": "excluded",
        "package_version": PACKAGE_VERSION, "python_version": platform.python_version(),
        "polars_version": pl.__version__,
    }
    return BacktestResult(**{name: pl.DataFrame(rows, schema=SCHEMAS[name])
                             for name, rows in records.items()}, metadata=metadata,
                          status=status, stop_reason=stop_reason,
                          stop_session=stop_session, stop_time=stop_time)
