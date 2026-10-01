"""Shared daily ledger for buy-and-hold and explicitly scheduled target baskets."""

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
from ._sofr import SOFRFinancing, _sofr_plan
from ._dividends import DividendReinvestment
from ._portfolio import BuyHoldPolicy, _number, _weights
from ._results import BacktestResult
from ._rebalancing import RebalancePolicy, _targets, _basket


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
    "targets": {"decision_session": D, "session": D, "asset": S, "weight": F, "gross_leverage": F},
    "rebalances": {"session": D, "decision_session": D, "equity_before": F, "equity_after": F,
                   "target_gross_leverage": F, "actual_gross_leverage": F,
                   "trade_cost": F, "gross_traded_notional": F, "receivable_reserved": F},
    "turnover": {"session": D, "phase": S, "gross_traded_notional": F,
                 "equity_before": F, "turnover": F},
    "dividend_reinvestments": {"action_id": S, "asset": S, "pay_date": D,
        "session": D, "paid_amount": F, "debt_repaid": F, "cash_released": F,
        "signed_notional": F, "trade_cost": F, "trade_id": S, "status": S},
    "financing_accruals": {"date": D, "cutoff_at": UTC, "observation_date": D,
        "available_at": UTC, "rate_age_days": I, "sofr": F, "borrowing_spread_bps": F,
        "borrowing_rate": F, "cash_rate": F, "borrowing_day_fraction": F, "cash_day_fraction": F,
        "opening_debt": F, "opening_cash": F, "borrowing_interest": F, "cash_interest": F},
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
                 financing: Financing | SOFRFinancing | None = None,
                 dividend_reinvestment: DividendReinvestment | None = None) -> BacktestResult:
    """Run a fractional-share, raw-close buy-and-hold portfolio with explicit funding.

    Supply ``financing`` for leveraged runs. The existing explicit ``cash_rate``
    and ``cash_day_count`` pair remains supported for unlevered calls; do not mix
    the two configurations. Rates capitalize daily, including nontrading dates.
    Breached margin or nonpositive equity stops at the failure close, preserving
    balances and setting incomplete status. No liquidation or rebalancing occurs.
    Optional ``dividend_reinvestment`` purchases the paying asset after payment;
    without it, quantities change only for splits.
    Entry costs reduce the first holding interval's P&L, whose denominator is
    original capital. Dividends accrue on ex-date and pay on their actual date.

    All results are numerical Polars tables. See ``docs/api.md`` for the complete
    input and output contracts. This function performs no network or file I/O.
    """
    return _simulate(market, weights=weights, initial_capital=initial_capital,
        entry_session=entry_session, end_session=end_session, policy=policy, costs=costs,
        cash_rate=cash_rate, cash_day_count=cash_day_count, financing=financing,
        dividend_reinvestment=dividend_reinvestment)


def scheduled_rebalance(market, *, targets, initial_capital, entry_session, end_session,
                        policy: RebalancePolicy, costs: TradeCosts, financing: Financing | SOFRFinancing,
                        dividend_reinvestment: DividendReinvestment | None = None):
    """Execute explicit dated long-only targets through the shared daily ledger.

    Target decisions precede execution, every basket includes zero-weight exits,
    and no trade executes on the terminal session. See docs/api.md for schemas,
    receivable funding rules, turnover denominators, and research margin stops.
    """
    if not isinstance(policy, RebalancePolicy) or not isinstance(financing, (Financing, SOFRFinancing)):
        raise ValueError("scheduled_rebalance requires RebalancePolicy and Financing or SOFRFinancing")
    policy.__post_init__()
    market = _validated_market(market)
    table, plans = _targets(targets, market, entry_session, end_session, policy)
    first_weights, first_leverage, _ = plans[entry_session]
    for _, leverage, _ in plans.values():
        if leverage > 1 and financing.maintenance_equity_ratio is None:
            raise ValueError("any leveraged target requires maintenance_equity_ratio")
        if leverage and _below_margin(1/leverage, financing.maintenance_equity_ratio):
            raise ValueError("target leverage violates maintenance_equity_ratio")
    initial_policy = BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
        fractional_shares=True, initial_gross_leverage=first_leverage, terminal_action="mark_only")
    return _simulate(market, weights=first_weights, initial_capital=initial_capital,
        entry_session=entry_session, end_session=end_session, policy=initial_policy, costs=costs,
        financing=financing, schedule=plans, target_table=table, rebalance_policy=policy,
        dividend_reinvestment=dividend_reinvestment)


def _simulate(market, *, weights, initial_capital, entry_session, end_session, policy,
              costs, cash_rate=None, cash_day_count=None, financing=None,
              schedule=None, target_table=None, rebalance_policy=None, dividend_reinvestment=None):
    market = _validated_market(market)
    if market.metadata["price_basis"] != "raw":
        raise ValueError("buy_and_hold requires raw execution prices")
    if not isinstance(policy, BuyHoldPolicy) or not isinstance(costs, TradeCosts):
        raise ValueError("explicit BuyHoldPolicy and TradeCosts objects are required")
    policy.__post_init__()
    if dividend_reinvestment is not None:
        if not isinstance(dividend_reinvestment, DividendReinvestment):
            raise ValueError("dividend_reinvestment must be a DividendReinvestment object or None")
        dividend_reinvestment.__post_init__()
    capital = _number(initial_capital, "initial_capital", positive=True)
    exposure = policy.initial_gross_leverage
    sofr_plan = None
    borrowing_denominator, cash_denominator = 365, 365
    if financing is not None:
        if not isinstance(financing, (Financing, SOFRFinancing)):
            raise ValueError("financing must be an explicit Financing object or SOFRFinancing object")
        if cash_rate is not None or cash_day_count is not None:
            raise ValueError("do not mix financing with cash_rate/cash_day_count")
        rate = financing.cash_rate
        if isinstance(financing, SOFRFinancing):
            # Validate/copy mutable rate inputs below, after checking run dates.
            borrowing_rate = None
            cash_day_count = financing.cash_day_count
            borrowing_denominator = 360 if financing.day_count == "ACT/360" else 365
            cash_denominator = 360 if cash_day_count == "ACT/360" else 365
        else:
            financing.__post_init__()
            borrowing_rate = financing.borrowing_rate
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
    if isinstance(financing, SOFRFinancing):
        sofr_plan, financing_metadata = _sofr_plan(financing, entry_session, end_session, market.metadata["currency"])
    else:
        financing_metadata = asdict(financing) if financing else None
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
    if target_table is not None:
        records["targets"] = target_table.to_dicts()
        # Validate all cost/leverage combinations before processing the first fill.
        for target_weights, target_leverage, _ in schedule.values():
            _basket({a: 0. for a in assets}, capital, 0., target_weights,
                    target_leverage, rates, rebalance_policy.receivable_policy)
    quantity = {a: 0.0 for a in assets}
    cash = capital
    debt = 0.0
    receivables = {}
    reinvestment_rows, reinvestment_budgets = {}, {}
    pending_pnl = defaultdict(list)
    cash_deltas, receivable_deltas, debt_deltas = [], [], []
    quantity_deltas = defaultdict(list)

    def event(day, kind, *, phase="before_close", asset=None, action_id=None,
              trade_id=None, dq=0.0, dc=0.0, dr=0.0, dd=0.0):
        seq = len(records["events"])
        records["events"].append({
            "event_id": f"E{seq:06d}", "date": day,
            "time": closes[day] if phase in {"entry_close", "rebalance_close", "reinvestment_close", "close"} else None,
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
        if phase in {"pre_entry", "post_entry"} and equity <= 0:
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
    records["turnover"].append({"session": entry_session, "phase": "entry",
        "gross_traded_notional": entry_notional, "equity_before": capital,
        "turnover": entry_notional/capital})

    def rebalance(day, values, outstanding, equity):
        nonlocal cash, debt
        target_weights, target_leverage, decision = schedule[day]
        changes, cost = _basket(values, equity, outstanding, target_weights,
                                target_leverage, rates, rebalance_policy.receivable_policy)
        new_net_cash = math.fsum([cash, -debt, -math.fsum(changes.values()), -cost])
        new_debt = max(-new_net_cash, 0.)
        tolerance = min(.009, 1e-8 + 1e-12*max(capital, equity))
        if new_debt <= tolerance and target_leverage <= 1:
            new_debt = 0.
        if target_leverage <= 1 and new_debt > tolerance and (
                rebalance_policy.receivable_policy == "require_target" or new_debt > debt + tolerance):
            raise ValueError(f"target on {day} requires financing unspendable receivables; use reserve or hold cash")
        if new_debt > 0 and threshold is None:
            raise ValueError("borrowing requires maintenance_equity_ratio")
        # Borrow only the funded basket's required net debt, execute sales before
        # purchases, then repay excess debt. No transient funding deficits.
        if new_debt > debt:
            borrowing = new_debt-debt
            cash += borrowing
            debt += borrowing
            event(day, "borrowing", phase="rebalance_close", dc=borrowing, dd=borrowing)
        for asset in sorted(assets, key=lambda a: (changes[a] >= 0, a)):
            notional = changes[asset]
            if notional == 0:
                continue
            mark = prices[day, asset]
            delta = notional/mark
            if values[asset]+notional == 0.:
                delta = -quantity[asset]
            next_quantity = quantity[asset]+delta
            if next_quantity < 0 and math.isclose(next_quantity, 0., abs_tol=1e-12):
                next_quantity = 0.
            if next_quantity < 0 or not math.isfinite(next_quantity):
                raise ValueError("scheduled quantity is not representable")
            component_costs = {c: abs(notional)*rates[c][asset] for c in rates}
            trade_cost = math.fsum(component_costs.values())
            trade_id = f"T{len(records['trades']):06d}"
            quantity[asset] = next_quantity
            cash -= notional+trade_cost
            event(day, "trade", phase="rebalance_close", asset=asset, trade_id=trade_id,
                  dq=delta, dc=-notional)
            records["trades"].append({"trade_id": trade_id, "session": day, "time": closes[day],
                "asset": asset, "signed_quantity": delta, "reference_price": mark,
                "signed_notional": notional, "execution": "scheduled_close", "trade_cost": trade_cost})
            for component, amount in component_costs.items():
                if not amount:
                    continue
                records["costs"].append({"cost_id": f"C{len(records['costs']):06d}", "date": day,
                    "time": closes[day], "trade_id": trade_id, "asset": asset,
                    "component": component, "amount": amount, "basis": "modeled"})
                event(day, component, phase="rebalance_close", asset=asset, trade_id=trade_id, dc=-amount)
                pending_pnl[component, asset].append(-amount)
            if cash < -tolerance:
                raise ArithmeticError("scheduled trade basket has an unfunded purchase")
        if debt > new_debt:
            repayment = debt-new_debt
            debt = new_debt
            cash -= repayment
            event(day, "debt_repayment", phase="rebalance_close", dc=-repayment, dd=-repayment)
        expected_cash = max(new_net_cash, 0.)
        residual, tol = _check(cash, expected_cash, max(capital, equity), "rebalance_cash_roundoff")
        records["diagnostics"].append({"session": day, "code": "rebalance_cash_roundoff",
                                      "residual": residual, "tolerance": tol})
        cash = expected_cash
        gross = math.fsum(quantity[a]*prices[day, a] for a in assets)
        after = math.fsum([gross, cash, outstanding, -debt])
        residual, tol = _check(after, equity-cost, max(capital, equity), "rebalance_cost_equity")
        records["diagnostics"].append({"session": day, "code": "rebalance_cost_equity",
                                      "residual": residual, "tolerance": tol})
        if after <= 0:
            raise ValueError("positive post-trade equity is not representable")
        traded = math.fsum(abs(v) for v in changes.values())
        records["rebalances"].append({"session": day, "decision_session": decision,
            "equity_before": equity, "equity_after": after, "target_gross_leverage": target_leverage,
            "actual_gross_leverage": gross/after, "trade_cost": cost, "gross_traded_notional": traded,
            "receivable_reserved": (max(0., target_leverage*after-max(0., after-outstanding))
                if target_leverage <= 1 and rebalance_policy.receivable_policy == "reserve" else 0.)})
        records["turnover"].append({"session": day, "phase": "rebalance", "gross_traded_notional": traded,
                                   "equity_before": equity, "turnover": traded/equity})

    def release_reinvestment(day, reason):
        # Release an earmark, not cash itself: it was already credited at payment.
        for action_id, budget in reinvestment_budgets.items():
            reinvestment_rows[action_id].update(session=day, cash_released=budget, status=reason)
        reinvestment_budgets.clear()

    def sweep_cash(day):
        nonlocal cash, debt
        reserved = math.fsum(reinvestment_budgets.values())
        if dividend_reinvestment is None or dividend_reinvestment.funding == "after_debt_repayment":
            reserved = 0.
        if debt > 0 and cash > reserved:
            repayment = min(cash-reserved, debt)
            cash -= repayment
            debt -= repayment
            event(day, "debt_repayment", dc=-repayment, dd=-repayment)
            # Unreserved account cash pays first. If dividend cash is needed,
            # distribute the reduction pro rata across outstanding earmarks.
            total = math.fsum(reinvestment_budgets.values())
            used = min(total, max(0., total-cash))
            if used:
                for action_id, budget in list(reinvestment_budgets.items()):
                    reduction = budget*(used/total)
                    remaining = budget-reduction
                    reinvestment_rows[action_id]["debt_repaid"] += reduction
                    if remaining == 0.:
                        reinvestment_rows[action_id]["status"] = "debt_repaid"
                        del reinvestment_budgets[action_id]
                    else:
                        reinvestment_budgets[action_id] = remaining

    def reinvest(day, equity):
        nonlocal cash
        notionals, expenses = [], []
        for action_id, budget in reinvestment_budgets.items():
            row = reinvestment_rows[action_id]
            asset = row["asset"]
            k = math.fsum(rates[c][asset] for c in rates)
            notional = budget/(1+k)
            delta = notional/prices[day, asset]
            updated_quantity = quantity[asset]+delta
            if (not all(math.isfinite(v) and v > 0 for v in (notional, delta, updated_quantity))
                    or updated_quantity <= quantity[asset]):
                raise ValueError("dividend reinvestment quantity is not representable")
            components = {c: notional*rates[c][asset] for c in rates}
            cost = math.fsum(components.values())
            _check(notional+cost, budget, max(capital, equity), "reinvestment_budget")
            trade_id = f"T{len(records['trades']):06d}"
            quantity[asset] = updated_quantity
            cash -= budget
            event(day, "trade", phase="reinvestment_close", asset=asset, action_id=action_id,
                  trade_id=trade_id, dq=delta, dc=-notional)
            records["trades"].append({"trade_id": trade_id, "session": day, "time": closes[day],
                "asset": asset, "signed_quantity": delta, "reference_price": prices[day, asset],
                "signed_notional": notional, "execution": "dividend_reinvestment_close", "trade_cost": cost})
            for component, amount in components.items():
                if not amount:
                    continue
                records["costs"].append({"cost_id": f"C{len(records['costs']):06d}", "date": day,
                    "time": closes[day], "trade_id": trade_id, "asset": asset,
                    "component": component, "amount": amount, "basis": "modeled"})
                event(day, component, phase="reinvestment_close", asset=asset, action_id=action_id,
                      trade_id=trade_id, dc=-amount)
                pending_pnl[component, asset].append(-amount)
            row.update(session=day, signed_notional=notional, trade_cost=cost,
                       trade_id=trade_id, status="reinvested")
            notionals.append(notional)
            expenses.append(cost)
        if cash < 0:
            _check(cash, 0., max(capital, equity), "reinvestment_cash_roundoff")
            cash = 0.
        outstanding = math.fsum(item["outstanding"] for item in receivables.values())
        after = math.fsum([*(quantity[a]*prices[day, a] for a in assets), cash, outstanding, -debt])
        residual, tolerance = _check(after, equity-math.fsum(expenses), max(capital, equity), "reinvestment_cost_equity")
        records["diagnostics"].append({"session": day, "code": "reinvestment_cost_equity",
                                       "residual": residual, "tolerance": tolerance})
        traded = math.fsum(notionals)
        records["turnover"].append({"session": day, "phase": "dividend_reinvestment",
            "gross_traded_notional": traded, "equity_before": equity, "turnover": traded/equity})
        reinvestment_budgets.clear()

    opening_equity, previous_session = capital, entry_session
    cumulative_pnls, simple_returns = [], []
    wealth = 1.0
    peak = max(capital, post_entry_equity)
    day = entry_session + timedelta(days=1)
    session_set = set(dates)
    status, stop_reason, stop_session, stop_time = "complete", None, None, None
    while day <= end_session:
        daily_borrowing_rate = sofr_plan[day]["borrowing_rate"] if sofr_plan is not None else borrowing_rate
        interest = cash * (rate / cash_denominator)
        borrowing_charge = debt * (daily_borrowing_rate / borrowing_denominator)
        if sofr_plan is not None:
            records["financing_accruals"].append({"date": day, **sofr_plan[day],
                "cash_rate": rate, "borrowing_day_fraction": 1/borrowing_denominator,
                "cash_day_fraction": 1/cash_denominator, "opening_debt": debt, "opening_cash": cash,
                "borrowing_interest": borrowing_charge, "cash_interest": interest})
        if interest:
            cash += interest
            event(day, "cash_interest", dc=interest)
            pending_pnl["cash_interest", None].append(interest)
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
                if dividend_reinvestment is not None:
                    row = {"action_id": action_id, "asset": item["asset"], "pay_date": day,
                        "session": None, "paid_amount": amount, "debt_repaid": 0., "cash_released": 0.,
                        "signed_notional": 0., "trade_cost": 0., "trade_id": None, "status": "pending"}
                    reinvestment_rows[action_id] = row
                    records["dividend_reinvestments"].append(row)
                    reinvestment_budgets[action_id] = amount
        if day == end_session:
            release_reinvestment(day, "terminal_cash")
        elif schedule and day in schedule:
            release_reinvestment(day, "scheduled_rebalance")
        sweep_cash(day)
        if day in session_set:
            # Old holdings earn the interval's price move. Rebalancing at this
            # close affects only future price P&L, with trade costs booked today.
            before = {a: quantity[a]*prices[day, a] for a in assets}
            for asset in assets:
                pending_pnl["price_pnl", asset].append(before[asset] - previous_values[asset])
            if schedule and day in schedule:
                before, outstanding, before_equity = balances(day, "pre_rebalance")
                peak = max(peak, before_equity)
                before_gross = math.fsum(before.values())
                # A pre-trade failure cannot be hidden by a scheduled deleveraging.
                if before_equity > 0 and not _below_margin(
                        before_equity/before_gross if before_gross else None, threshold):
                    rebalance(day, before, outstanding, before_equity)
            elif reinvestment_budgets:
                before, outstanding, before_equity = balances(day, "pre_reinvestment")
                peak = max(peak, before_equity)
                before_gross = math.fsum(before.values())
                if before_equity > 0 and not _below_margin(
                        before_equity/before_gross if before_gross else None, threshold):
                    reinvest(day, before_equity)
                else:
                    release_reinvestment(day, "stopped_before_trade")
            values, outstanding, equity = balances(day, "close")
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
    for row in records["dividend_reinvestments"]:
        residual, tolerance = _check(row["paid_amount"], math.fsum(row[k] for k in (
            "debt_repaid", "cash_released", "signed_notional", "trade_cost")), capital, "dividend_budget_allocation")
        records["diagnostics"].append({"session": row["session"] or row["pay_date"],
            "code": "dividend_budget_allocation", "residual": residual, "tolerance": tolerance})
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
        "financing": financing_metadata,
        "borrowing_rate": borrowing_rate, "maintenance_equity_ratio": threshold,
        "margin_monitoring": "session_close", "margin_comparison_relative_tolerance": 1e-12,
        "return_denominator": "first_interval_pre_entry_capital_then_prior_closing_equity",
        "dividend_policy": "ex_date_receivable_pay_date_cash_no_reinvestment",
        "cash_sweep": financing.cash_sweep if financing else "hold_cash_no_debt",
        "settlement": "immediate", "taxes": "excluded",
        "package_version": PACKAGE_VERSION, "python_version": platform.python_version(),
        "polars_version": pl.__version__,
    }
    if schedule is not None:
        metadata.update(strategy="scheduled_rebalance", rebalance_policy=asdict(rebalance_policy),
            dividend_policy="ex_date_receivable_pay_date_cash_reinvest_only_via_scheduled_trades",
            decision_timing="decision_session_strictly_before_execution",
            concentration_denominator="target_risky_asset_proportion",
            target_weight_normalization="divide_by_basket_sum_within_1e-12_roundoff",
            margin_monitoring="before_scheduled_trade_and_after_session_close",
            turnover_denominator="pre_trade_equity_entry_separately_labeled",
            no_trade_relative_tolerance=1e-13)
    if dividend_reinvestment is not None:
        metadata.update(dividend_reinvestment=asdict(dividend_reinvestment),
            dividend_policy="ex_date_receivable_pay_date_cash_explicit_reinvestment",
            reinvestment_payment_availability="before_close_on_pay_date_assumed",
            reinvestment_asset_policy="paying_asset_even_if_previously_sold",
            reinvestment_budget="paid_dividend_principal_net_of_assigned_debt_repayment_including_trade_costs",
            margin_monitoring="before_scheduled_or_reinvestment_trade_and_after_session_close",
            turnover_denominator="pre_trade_equity_entry_separately_labeled")
    return BacktestResult(**{name: pl.DataFrame(rows, schema=SCHEMAS[name])
                             for name, rows in records.items()}, metadata=metadata,
                          status=status, stop_reason=stop_reason,
                          stop_session=stop_session, stop_time=stop_time)
