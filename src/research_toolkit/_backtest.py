"""Shared daily ledger for buy-and-hold and explicitly scheduled target baskets."""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version
import math
import platform

import polars as pl

from ._shorts import LongShortPolicy, StockBorrow, _signed, _signed_targets, _signed_size, _borrow_plan, _cutoff
from ._costs import TradeCosts
from ._execution import SquareRootImpactCosts, COST_SCHEMA, _entry_size, _impact_basket, _bind_cost_schedule
from ._data import _validated_market
from ._financing import Financing, _below_margin
from ._sofr import SOFRFinancing, _sofr_plan
from ._dividends import DividendReinvestment
from ._portfolio import BuyHoldPolicy, _number, _weights
from ._results import BacktestResult, SignalResult
from ._signals import _checked_signals, _attach_signals, SIGNAL_AUDIT_SCHEMA
from ._rebalancing import RebalancePolicy, _targets, _basket


try:
    PACKAGE_VERSION = version("research-toolkit")
except PackageNotFoundError:
    PACKAGE_VERSION = "uninstalled"


F, S, D, I = pl.Float64, pl.String, pl.Date, pl.Int64
UTC = pl.Datetime("us", "UTC")
SCHEMAS = {
    "target_executions": {"decision_session": D, "session": D, "asset": S,
        "target_notional": F, "pre_trade_quantity": F, "pre_trade_notional": F,
        "reference_price": F, "signed_quantity": F, "quantity": F, "actual_notional": F,
        "notional_residual": F, "tolerance": F, "trade_id": S, "trade_cost": F},
    "cost_model_selections": {"decision_session": D, "session": D, "model_id": S,
        "snapshot_id": S, "sample_start": D, "sample_end": D, "status": S},
    "stock_borrow_accruals": {"date": D, "asset": S, "mark_session": D, "cutoff_at": UTC,
        "available_at": UTC, "short_market_value": F, "annual_rate": F, "day_fraction": F, "amount": F},
    "short_financing_accruals": {"date": D, "opening_collateral": F, "rebate_rate": F,
        "day_fraction": F, "rebate": F},
    "dividend_liabilities": {"session": D, "action_id": S, "asset": S, "ex_session": D,
        "pay_date": D, "entitled_quantity": F, "amount": F, "outstanding": F},
    "signal_audit": {**SIGNAL_AUDIT_SCHEMA, "status": pl.String},
    "execution_costs": {"trade_id": S, "session": D, "decision_session": D,
        "model_id": S, "snapshot_id": S, **COST_SCHEMA},
    "daily": {"session": D, "period_start": D, "opening_equity": F, "equity": F,
              "pnl": F, "simple_return": F, "log_return": F, "cumulative_pnl": F,
              "cumulative_simple_return": F, "compounded_return": F, "cash": F,
              "debt": F, "dividend_receivable": F, "dividend_liability": F, "restricted_collateral": F,
              "long_exposure": F, "short_exposure": F, "short_liability": F, "margin_required": F, "gross_exposure": F,
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
               "receivable_delta": F, "collateral_delta": F, "dividend_liability_delta": F},
    "valuations": {"session": D, "time": UTC, "phase": S, "market_value": F,
                   "cash": F, "debt": F, "dividend_receivable": F, "dividend_liability": F,
                   "restricted_collateral": F, "short_liability": F, "equity": F},
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


def buy_and_hold(market, *, weights=None, equity_exposures=None, quantities=None,
                 long_short: LongShortPolicy | None = None, stock_borrow: StockBorrow | None = None,
                 initial_capital: float, entry_session: date,
                 end_session: date, policy: BuyHoldPolicy, costs: TradeCosts | SquareRootImpactCosts,
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

    Supply exactly one of ``weights`` (existing nonnegative risky proportions),
    ``equity_exposures`` (signed multiples of post-cost equity), or ``quantities``
    (exact signed shares). The latter two require LongShortPolicy, StockBorrow,
    explicit financing, and policy.initial_gross_leverage=1. Exact shares are never
    rescaled to pay costs. Restricted collateral cannot fund the long holdings.

    All results are numerical Polars tables. See ``docs/api.md`` for the complete
    input and output contracts. This function performs no network or file I/O.
    """
    return _simulate(market, weights=weights, initial_capital=initial_capital,
        entry_session=entry_session, end_session=end_session, policy=policy, costs=costs,
        cash_rate=cash_rate, cash_day_count=cash_day_count, financing=financing,
        dividend_reinvestment=dividend_reinvestment, equity_exposures=equity_exposures,
        quantities=quantities, long_short=long_short, stock_borrow=stock_borrow)


def scheduled_rebalance(market, *, targets, initial_capital, entry_session, end_session,
                        policy: RebalancePolicy,
                        costs: TradeCosts | SquareRootImpactCosts | Mapping[date, SquareRootImpactCosts],
                        financing: Financing | SOFRFinancing,
                        dividend_reinvestment: DividendReinvestment | None = None,
                        long_short: LongShortPolicy | None = None, stock_borrow: StockBorrow | None = None):
    """Execute dated long-only or explicitly signed targets through the shared ledger.

    Signed targets require long_short and stock_borrow and use equity_exposure,
    quantity or target_notional columns. Existing long-only target schemas remain unchanged.
    Dollar amounts divide by execution raw prices without rescaling for costs.
    Costs accept one static model or an exact decision-date square-root mapping.
    Target decisions precede execution, every basket includes explicit zero exits,
    and no trade executes on the terminal session. See docs/api.md for schemas,
    receivable funding rules, turnover denominators, and research margin stops.
    """
    if not isinstance(policy, RebalancePolicy) or not isinstance(financing, (Financing, SOFRFinancing)):
        raise ValueError("scheduled_rebalance requires RebalancePolicy and Financing or SOFRFinancing")
    policy.__post_init__()
    market = _validated_market(market)
    if long_short is not None:
        if policy.receivable_policy != "require_target":
            raise ValueError("signed targets require receivable_policy='require_target'; declared loans fund unavailable receivables")
        table, plans, kind = _signed_targets(targets, market, entry_session, end_session)
        first, _, _ = plans[entry_session]
        initial_policy = BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
            fractional_shares=True, initial_gross_leverage=1., terminal_action="mark_only")
        return _simulate(market, weights=None, initial_capital=initial_capital,
            entry_session=entry_session, end_session=end_session, policy=initial_policy, costs=costs,
            financing=financing, schedule=plans, target_table=table, rebalance_policy=policy,
            dividend_reinvestment=dividend_reinvestment, long_short=long_short, stock_borrow=stock_borrow,
            equity_exposures=first if kind == "equity_exposure" else None,
            quantities=first if kind == "quantity" else None,
            target_notionals=first if kind == "target_notional" else None)
    if stock_borrow is not None:
        raise ValueError("stock_borrow requires long_short")
    signals = _checked_signals(targets, market, entry_session, end_session) if isinstance(targets, SignalResult) else None
    table, plans = _targets(signals.targets if signals else targets, market, entry_session, end_session, policy)
    first_weights, first_leverage, _ = plans[entry_session]
    for _, leverage, _ in plans.values():
        if leverage > 1 and financing.maintenance_equity_ratio is None:
            raise ValueError("any leveraged target requires maintenance_equity_ratio")
        if leverage and _below_margin(1/leverage, financing.maintenance_equity_ratio):
            raise ValueError("target leverage violates maintenance_equity_ratio")
    initial_policy = BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
        fractional_shares=True, initial_gross_leverage=first_leverage, terminal_action="mark_only")
    result = _simulate(market, weights=first_weights, initial_capital=initial_capital,
        entry_session=entry_session, end_session=end_session, policy=initial_policy, costs=costs,
        financing=financing, schedule=plans, target_table=table, rebalance_policy=policy,
        dividend_reinvestment=dividend_reinvestment)

    return _attach_signals(result, signals) if signals else result


def _simulate(market, *, weights, initial_capital, entry_session, end_session, policy,
              costs, cash_rate=None, cash_day_count=None, financing=None,
              schedule=None, target_table=None, rebalance_policy=None, dividend_reinvestment=None,
              equity_exposures=None, quantities=None, long_short=None, stock_borrow=None, target_notionals=None):
    market = _validated_market(market)
    signed = long_short is not None
    if sum(v is not None for v in (weights, equity_exposures, quantities, target_notionals)) != 1:
        raise ValueError("supply exactly one of weights, equity_exposures, quantities or scheduled target_notionals")
    if signed:
        if not isinstance(long_short, LongShortPolicy) or weights is not None:
            raise ValueError("LongShortPolicy requires signed position targets, not long-allocation weights")
        long_short.__post_init__()
        if not isinstance(financing, (Financing, SOFRFinancing)) or not isinstance(policy, BuyHoldPolicy) or policy.initial_gross_leverage != 1.:
            raise ValueError("signed sizing requires financing and policy.initial_gross_leverage=1; signed targets set the size")
        if financing.maintenance_equity_ratio is not None:
            raise ValueError("signed margin uses LongShortPolicy; set financing.maintenance_equity_ratio=None")
    elif any(v is not None for v in (equity_exposures, quantities, target_notionals, stock_borrow)):
        raise ValueError("signed inputs require LongShortPolicy and StockBorrow")
    if market.metadata["price_basis"] != "raw":
        raise ValueError("buy_and_hold requires raw execution prices")
    if not isinstance(policy, BuyHoldPolicy) or not isinstance(costs, (TradeCosts, SquareRootImpactCosts, Mapping)):
        raise ValueError("explicit BuyHoldPolicy and supported cost model objects are required")
    policy.__post_init__()
    if isinstance(costs, Mapping) and schedule is None:
        raise ValueError("dated cost models require scheduled_rebalance with explicit decision dates")
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
    if target_notionals is not None:
        signed_kind, signed_input = "target_notional", target_notionals
    elif quantities is not None:
        signed_kind, signed_input = "quantity", quantities
    else:
        signed_kind, signed_input = "equity_exposure", equity_exposures
    allocation = (_signed(signed_input,
        market.prices["asset"].unique().to_list(), signed_kind) if signed else
        _weights(weights, market.prices["asset"].unique().to_list()))
    borrow_plan, borrow_metadata = {}, None
    if signed:
        short_assets = sorted({a for basket in ([p[0] for p in schedule.values()] if schedule else [allocation])
                               for a, v in basket.items() if v < 0})
        borrow_plan, borrow_metadata = _borrow_plan(stock_borrow, short_assets, entry_session, end_session)
    assets = sorted(allocation)
    total_weight = math.fsum(allocation.values())
    resolved_weights = allocation.copy() if signed else {a: allocation[a] / total_weight for a in assets}
    dates = [d for d in all_sessions if entry_session <= d <= end_session]
    closes = dict(market.sessions.select("session", "close_at").iter_rows())
    prices = {(d, a): p for d, a, p in market.prices.iter_rows()}
    impact_costs = isinstance(costs, (SquareRootImpactCosts, Mapping))
    bindings, executed_baskets = {}, {entry_session}
    entry_binding = None
    rates = None if impact_costs else costs.resolve(assets)
    if impact_costs:
        executions = {day: plan[2] for day, plan in schedule.items()} if schedule else {entry_session: None}
        bindings = _bind_cost_schedule(costs, executions, assets, prices, closes, market.metadata["currency"])
        entry_binding = bindings[entry_session]
        gross = 0. if signed else _entry_size(capital, exposure, resolved_weights, entry_binding)
    elif signed:
        gross = 0.
    else:
        fee_rate = math.fsum(resolved_weights[a] * math.fsum(rates[c][a] for c in rates) for a in assets)
        gross = capital * exposure / (1 + exposure * fee_rate)
    if not math.isfinite(gross):
        raise ValueError("initial notional is not representable")
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
        for day, (target_weights, target_leverage, decision) in schedule.items():
            if not impact_costs and not signed:
                _basket({a: 0. for a in assets}, capital, 0., target_weights,
                        target_leverage, rates, rebalance_policy.receivable_policy)
    quantity = {a: 0.0 for a in assets}
    cash = capital
    debt = 0.0
    collateral = 0.0
    liabilities = {}
    receivables = {}
    reinvestment_rows, reinvestment_budgets = {}, {}
    pending_pnl = defaultdict(list)
    cash_deltas, receivable_deltas, debt_deltas = [], [], []
    collateral_deltas, liability_deltas = [], []
    quantity_deltas = defaultdict(list)

    def event(day, kind, *, phase="before_close", asset=None, action_id=None,
              trade_id=None, dq=0.0, dc=0.0, dr=0.0, dd=0.0, dk=0.0, dl=0.0):
        seq = len(records["events"])
        records["events"].append({
            "event_id": f"E{seq:06d}", "date": day,
            "time": closes[day] if phase in {"entry_close", "rebalance_close", "reinvestment_close", "close"} else None,
            "sequence": seq, "phase": phase, "type": kind, "asset": asset,
            "action_id": action_id, "trade_id": trade_id,
            "quantity_delta": dq, "cash_delta": dc, "debt_delta": dd,
            "receivable_delta": dr, "collateral_delta": dk, "dividend_liability_delta": dl,
        })
        collateral_deltas.append(dk)
        liability_deltas.append(dl)
        cash_deltas.append(dc)
        receivable_deltas.append(dr)
        debt_deltas.append(dd)
        if asset is not None:
            quantity_deltas[asset].append(dq)

    def balances(day, phase):
        values = {a: quantity[a] * prices[day, a] for a in assets}
        mv = math.fsum(values.values())
        receivable = math.fsum(item["outstanding"] for item in receivables.values())
        liability = math.fsum(item["outstanding"] for item in liabilities.values())
        equity = math.fsum([mv, cash, collateral, receivable, -liability, -debt])
        if (not all(math.isfinite(v) for v in (mv, cash, collateral, receivable, liability, debt, equity))
                or min(cash, collateral, receivable, liability, debt) < 0):
            raise ArithmeticError("ledger produced invalid balances")
        if phase in {"pre_entry", "post_entry"} and equity <= 0:
            raise ValueError("initial equity after costs must be positive")
        records["valuations"].append({"session": day, "time": closes[day], "phase": phase,
                                      "market_value": mv, "cash": cash, "debt": debt,
                                      "dividend_receivable": receivable, "dividend_liability": liability,
                                      "restricted_collateral": collateral, "short_liability": sum(max(-v, 0.) for v in values.values()), "equity": equity})
        checks = {
            "cash_reconciliation": (cash, math.fsum([capital, *cash_deltas])),
            "receivable_reconciliation": (receivable, math.fsum(receivable_deltas)),
            "debt_reconciliation": (debt, math.fsum(debt_deltas)),
            "collateral_reconciliation": (collateral, math.fsum(collateral_deltas)),
            "dividend_liability_reconciliation": (liability, math.fsum(liability_deltas)),
            "balance_sheet": (equity, math.fsum([*values.values(), cash, collateral, receivable, -liability, -debt])),
        }
        if signed:
            checks["marked_collateral"] = (collateral, long_short.collateral_multiple *
                math.fsum(max(-v, 0.) for v in values.values()))
        for a in assets:
            if not math.isclose(quantity[a], math.fsum(quantity_deltas[a]), rel_tol=1e-12, abs_tol=1e-12):
                raise ArithmeticError(f"quantity reconciliation failed for {a}")
        for code, (actual, expected) in checks.items():
            residual, tolerance = _check(actual, expected, max(capital, equity), code)
            records["diagnostics"].append({"session": day, "code": f"{phase}:{code}",
                                            "residual": residual, "tolerance": tolerance})
        return values, receivable, equity

    def fund(day, amount, phase="before_close"):
        nonlocal cash, debt
        if amount > 0:
            cash += amount
            debt += amount
            event(day, "borrowing", phase=phase, dc=amount, dd=amount)

    def available_cash():
        return max(0., cash-math.fsum(reinvestment_budgets.values()))

    def mark_collateral(day, values, phase="close"):
        nonlocal cash, collateral
        required = long_short.collateral_multiple * math.fsum(max(-v, 0.) for v in values.values())
        change = required-collateral
        fund(day, max(change-available_cash(), 0.), phase)
        cash -= change
        collateral = required
        if change:
            event(day, "collateral_transfer", phase=phase, dc=-change, dk=change)

    def margin_required(values):
        return (long_short.long_margin*math.fsum(max(v, 0.) for v in values.values()) +
                long_short.short_margin*math.fsum(max(-v, 0.) for v in values.values()))

    def breached(values, equity):
        if signed:
            return _below_margin(equity, margin_required(values))
        gross_value = math.fsum(values.values())
        return _below_margin(equity/gross_value if gross_value else None, threshold)

    def signed_basket(day, values, equity, targets, phase, decision=None):
        nonlocal cash, debt, collateral
        marks = {a: prices[day, a] for a in assets}
        old_quantities, fills = quantity.copy(), {}
        binding = bindings[day] if impact_costs else None
        def components(a, n):
            return binding.components(a, n) if binding else {c: abs(n)*rates[c][a] for c in rates}
        def marginal(a, n):
            return binding.marginal(a, n) if binding else math.fsum(rates[c][a] for c in rates)
        changes, fee = _signed_size(values, equity, targets, signed_kind, marks, components, marginal)
        new_values = {a: values[a]+changes[a] for a in assets}
        after = equity-fee
        if not all(math.isfinite(v) for v in [after, *new_values.values(), *changes.values()]) or after <= 0:
            raise ValueError("signed basket equity/positions are not representable")
        gross_value = math.fsum(abs(v) for v in new_values.values())
        if rebalance_policy and gross_value and any(abs(v)/gross_value > rebalance_policy.max_asset_weight+1e-12 for v in new_values.values()):
            raise ValueError("signed target exceeds max_asset_weight as a share of gross exposure")
        if phase == "entry_close" and breached(new_values, after):
            raise ValueError("signed entry violates long/short margin requirements")
        # Release collateral only within this atomic basket, then segregate the new
        # marked requirement before any debt sweep. Sale proceeds never enter a sweep.
        if collateral:
            cash += collateral
            event(day, "collateral_release_for_basket", phase=phase, dc=collateral, dk=-collateral)
            collateral = 0.
        required = long_short.collateral_multiple*math.fsum(max(-v, 0.) for v in new_values.values())
        fund(day, max(math.fsum([*changes.values(), fee, required, -cash]), 0.), phase)
        for a in sorted(assets, key=lambda a: (changes[a] >= 0, a)):
            n = changes[a]
            if not n:
                continue
            delta = -quantity[a] if new_values[a] == 0 else n/marks[a]
            quantity[a] += delta
            if not math.isfinite(quantity[a]):
                raise ValueError("signed quantity is not representable")
            trade_id = f"T{len(records['trades']):06d}"
            parts = components(a, n)
            charge = math.fsum(parts.values())
            fills[a] = (trade_id, delta, charge)
            cash -= n+charge
            event(day, "trade", phase=phase, asset=a, trade_id=trade_id, dq=delta, dc=-n)
            records["trades"].append(dict(trade_id=trade_id, session=day, time=closes[day], asset=a,
                signed_quantity=delta, reference_price=marks[a], signed_notional=n,
                execution="entry_close" if phase == "entry_close" else "scheduled_close", trade_cost=charge))
            if binding:
                records["execution_costs"].append(dict(binding.audit_row(a, n), trade_id=trade_id, session=day))
            for component, amount in parts.items():
                if amount:
                    records["costs"].append(dict(cost_id=f"C{len(records['costs']):06d}", date=day,
                        time=closes[day], trade_id=trade_id, asset=a, component=component, amount=amount, basis="modeled"))
                    event(day, component, phase=phase, asset=a, trade_id=trade_id, dc=-amount)
                    pending_pnl[component, a].append(-amount)
        mark_collateral(day, new_values, phase)
        # Roundoff is reconciled before normalization; all material deficits fail.
        if cash < 0:
            _check(cash, 0., capital, "signed_cash_roundoff")
            cash = 0.
        repayment = min(cash, debt)
        if repayment:
            cash -= repayment
            debt -= repayment
            event(day, "debt_repayment", phase=phase, dc=-repayment, dd=-repayment)
        actual_equity = math.fsum([*(quantity[a]*marks[a] for a in assets), cash, collateral,
            math.fsum(v["outstanding"] for v in receivables.values()),
            -math.fsum(v["outstanding"] for v in liabilities.values()), -debt])
        residual, tolerance = _check(actual_equity, after, max(capital, equity), "signed_basket_cost_equity")
        records["diagnostics"].append(dict(session=day, code="signed_basket_cost_equity", residual=residual, tolerance=tolerance))
        if signed_kind == "target_notional":
            actual_gross = math.fsum(abs(quantity[a]*marks[a]) for a in assets)
            for a in assets:
                actual = quantity[a]*marks[a]
                residual, tol = _check(actual, targets[a], max(capital, abs(targets[a])), "fixed_target_notional")
                trade_id, delta, charge = fills.get(a, (None, 0., 0.))
                records["target_executions"].append(dict(decision_session=decision, session=day, asset=a,
                    target_notional=targets[a], pre_trade_quantity=old_quantities[a], pre_trade_notional=values[a],
                    reference_price=marks[a], signed_quantity=delta, quantity=quantity[a], actual_notional=actual,
                    notional_residual=residual, tolerance=tol, trade_id=trade_id, trade_cost=charge))
            residual, tol = _check(actual_gross, math.fsum(abs(v) for v in targets.values()),
                max(capital, actual_gross), "fixed_target_gross")
            records["diagnostics"].append(dict(session=day, code="fixed_target_gross", residual=residual, tolerance=tol))
        executed_baskets.add(day)
        traded = math.fsum(abs(n) for n in changes.values())
        records["turnover"].append(dict(session=day, phase="entry" if phase == "entry_close" else "rebalance",
            gross_traded_notional=traded, equity_before=equity, turnover=traded/equity))
        if phase != "entry_close":
            records["rebalances"].append(dict(session=day, decision_session=decision, equity_before=equity,
                equity_after=after, target_gross_leverage=math.fsum(abs(v) for v in targets.values()) if signed_kind == "equity_exposure" else None,
                actual_gross_leverage=gross_value/after, trade_cost=fee, gross_traded_notional=traded, receivable_reserved=0.))
        return fee

    balances(entry_session, "pre_entry")
    if signed:
        entry_cost = signed_basket(entry_session, {a: 0. for a in assets}, capital, allocation,
            "entry_close", schedule[entry_session][2] if schedule else None)
    else:
        entry_plan = []
        for asset in assets:
            notional = gross * resolved_weights[asset]
            if notional == 0:
                continue
            mark = prices[entry_session, asset]
            qty = notional / mark
            if not math.isfinite(qty) or qty <= 0:
                raise ValueError(f"initial quantity is not representable for {asset}")
            components = entry_binding.components(asset, notional) if impact_costs else {c: notional * rates[c][asset] for c in rates}
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
            if impact_costs:
                records["execution_costs"].append(dict(entry_binding.audit_row(asset, notional), trade_id=trade_id, session=entry_session))
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
    if breached(previous_values, post_entry_equity):
        raise ValueError("funded entry violates maintenance_equity_ratio")

    def position_rows(day, values, equity):
        for asset in assets:
            records["positions"].append({"session": day, "asset": asset,
                                          "quantity": quantity[asset], "raw_mark": prices[day, asset],
                                          "market_value": values[asset],
                                          "weight": values[asset] / equity if equity > 0 else None})

    position_rows(entry_session, previous_values, post_entry_equity)
    if not signed:
        records["turnover"].append({"session": entry_session, "phase": "entry",
            "gross_traded_notional": entry_notional, "equity_before": capital,
            "turnover": entry_notional/capital})

    def rebalance(day, values, outstanding, equity):
        nonlocal cash, debt
        target_weights, target_leverage, decision = schedule[day]
        executed_baskets.add(day)
        if signed:
            signed_basket(day, values, equity, target_weights, "rebalance_close", decision)
            return
        binding = bindings[day] if impact_costs else None
        if impact_costs:
            changes, cost = _impact_basket(values, equity, outstanding, target_weights,
                target_leverage, binding, rebalance_policy.receivable_policy)
        else:
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
            component_costs = binding.components(asset, notional) if impact_costs else {c: abs(notional)*rates[c][asset] for c in rates}
            trade_cost = math.fsum(component_costs.values())
            trade_id = f"T{len(records['trades']):06d}"
            if impact_costs:
                records["execution_costs"].append(dict(binding.audit_row(asset, notional), trade_id=trade_id, session=day))
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
        notionals = []
        for action_id, budget in reinvestment_budgets.items():
            row = reinvestment_rows[action_id]
            asset = row["asset"]
            if quantity[asset] < 0:
                row.update(session=day, cash_released=budget, status="payer_now_short")
                continue
            # User-selected DRIP convention: no commission, spread or impact.
            # Ordinary entry and scheduled trades retain their configured costs.
            notional = budget
            delta = notional/prices[day, asset]
            updated_quantity = quantity[asset]+delta
            if (not all(math.isfinite(v) and v > 0 for v in (notional, delta, updated_quantity))
                    or updated_quantity <= quantity[asset]):
                raise ValueError("dividend reinvestment quantity is not representable")
            trade_id = f"T{len(records['trades']):06d}"
            quantity[asset] = updated_quantity
            cash -= budget
            event(day, "trade", phase="reinvestment_close", asset=asset, action_id=action_id,
                  trade_id=trade_id, dq=delta, dc=-notional)
            records["trades"].append({"trade_id": trade_id, "session": day, "time": closes[day],
                "asset": asset, "signed_quantity": delta, "reference_price": prices[day, asset],
                "signed_notional": notional, "execution": "dividend_reinvestment_close", "trade_cost": 0.})
            row.update(session=day, signed_notional=notional, trade_cost=0.,
                       trade_id=trade_id, status="reinvested")
            notionals.append(notional)
        if cash < 0:
            _check(cash, 0., max(capital, equity), "reinvestment_cash_roundoff")
            cash = 0.
        outstanding = math.fsum(item["outstanding"] for item in receivables.values())
        after = math.fsum([*(quantity[a]*prices[day, a] for a in assets), cash, collateral, outstanding,
            -math.fsum(v["outstanding"] for v in liabilities.values()), -debt])
        residual, tolerance = _check(after, equity, max(capital, equity), "reinvestment_equity_neutrality")
        records["diagnostics"].append({"session": day, "code": "reinvestment_equity_neutrality",
                                       "residual": residual, "tolerance": tolerance})
        traded = math.fsum(notionals)
        records["turnover"].append({"session": day, "phase": "dividend_reinvestment",
            "gross_traded_notional": traded, "equity_before": equity, "turnover": traded/equity})
        reinvestment_budgets.clear()
        if signed:
            sweep_cash(day)

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
        if signed:
            denominator = 360 if stock_borrow.day_count == "ACT/360" else 365
            for asset in short_assets:
                annual, available = borrow_plan[day, asset]
                short_value = max(-previous_values[asset], 0.)
                amount = short_value*annual/denominator
                records["stock_borrow_accruals"].append(dict(date=day, asset=asset, mark_session=previous_session,
                    cutoff_at=_cutoff(day), available_at=available, short_market_value=short_value,
                    annual_rate=annual, day_fraction=1/denominator, amount=amount))
                if amount:
                    fund(day, max(amount-available_cash(), 0.))
                    cash -= amount
                    event(day, "stock_borrow_fee", asset=asset, dc=-amount)
                    records["costs"].append(dict(cost_id=f"C{len(records['costs']):06d}", date=day,
                        time=None, trade_id=None, asset=asset, component="stock_borrow_fee", amount=amount,
                        basis=stock_borrow.metadata["basis"]))
                    pending_pnl["stock_borrow_fee", asset].append(-amount)
            denominator = 360 if long_short.rebate_day_count == "ACT/360" else 365
            rebate = collateral*long_short.rebate_rate/denominator
            records["short_financing_accruals"].append(dict(date=day, opening_collateral=collateral,
                rebate_rate=long_short.rebate_rate, day_fraction=1/denominator, rebate=rebate))
            if rebate:
                cash += rebate
                event(day, "short_collateral_rebate", dc=rebate)
                pending_pnl["short_collateral_rebate", None].append(rebate)
        for action in split_dates[day]:
            asset = action["asset"]
            old = quantity[asset]
            quantity[asset] *= action["ratio"]
            if old and (not math.isfinite(quantity[asset]) or quantity[asset] == 0):
                raise ValueError(f"split quantity is not representable for {asset} on {day}")
            if old:
                event(day, "split", asset=asset, action_id=action["action_id"], dq=quantity[asset] - old)
        for action in ex_dates[day]:
            asset = action["asset"]
            amount = quantity[asset] * action["cash_per_share"]
            if amount < 0:
                liabilities[action["action_id"]] = {**action, "entitled_quantity": quantity[asset],
                    "amount": -amount, "outstanding": -amount}
                event(day, "short_dividend_accrual", asset=asset, action_id=action["action_id"], dl=-amount)
                pending_pnl["short_dividend_expense", asset].append(amount)
            elif amount:
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
        for action_id, item in liabilities.items():
            if item["pay_date"] == day and item["outstanding"]:
                amount = item["outstanding"]
                fund(day, max(amount-available_cash(), 0.))
                cash -= amount
                item["outstanding"] = 0.
                event(day, "short_dividend_payment", asset=item["asset"], action_id=action_id, dc=-amount, dl=-amount)
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
            if signed:
                mark_collateral(day, before)
                sweep_cash(day)
            if schedule and day in schedule:
                before, outstanding, before_equity = balances(day, "pre_rebalance")
                peak = max(peak, before_equity)
                # A pre-trade failure cannot be hidden by a scheduled deleveraging.
                if before_equity > 0 and not breached(before, before_equity):
                    rebalance(day, before, outstanding, before_equity)
            elif reinvestment_budgets:
                before, outstanding, before_equity = balances(day, "pre_reinvestment")
                peak = max(peak, before_equity)
                if before_equity > 0 and not breached(before, before_equity):
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
            gross_value = math.fsum(abs(v) for v in values.values())
            equity_ratio = equity / gross_value if gross_value else None
            margin_breached = breached(values, equity)
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
                "restricted_collateral": collateral, "dividend_liability": math.fsum(v["outstanding"] for v in liabilities.values()),
                "long_exposure": math.fsum(max(v, 0.) for v in values.values()),
                "short_exposure": math.fsum(max(-v, 0.) for v in values.values()),
                "short_liability": math.fsum(max(-v, 0.) for v in values.values()),
                "margin_required": margin_required(values) if signed else (threshold or 0.)*gross_value,
                "gross_exposure": gross_value, "net_exposure": math.fsum(values.values()),
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
            for action_id, item in liabilities.items():
                records["dividend_liabilities"].append({"session": day, "action_id": action_id,
                    **{k: item[k] for k in SCHEMAS["dividend_liabilities"] if k not in {"session", "action_id"}}})
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
        "cost_model": entry_binding.metadata if impact_costs else {"model": "proportional"},
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
            reinvestment_budget="paid_dividend_principal_net_of_assigned_debt_repayment",
            reinvestment_cost_policy="zero_commission_spread_impact",
            margin_monitoring="before_scheduled_or_reinvestment_trade_and_after_session_close",
            turnover_denominator="pre_trade_equity_entry_separately_labeled")
    if signed:
        metadata.update(concentration_denominator="absolute_target_value_over_gross_exposure",
            target_weight_normalization="none_signed_inputs",
            long_short=asdict(long_short), stock_borrow=borrow_metadata,
            sizing=signed_kind, signed_targets=allocation, weights=None, resolved_weights=None,
            collateral_policy="marked_each_session_close_restricted_from_sweep",
            margin_convention="equity_ge_long_margin_times_long_plus_short_margin_times_absolute_short",
            funding="explicit_cash_loan_including_collateral_costs_and_unpaid_receivables",
            rebate_basis="opening_restricted_collateral_gross_before_separate_borrow_fees",
            reinvestment_if_payer_short="release_cash_no_cover",
            short_dividend_policy="ex_date_liability_pay_date_settlement_no_reinvestment")
    if impact_costs:
        metadata["cost_model_selection"] = "exact_decision_date" if isinstance(costs, Mapping) else "static"
        metadata["cost_models"] = [deepcopy(binding.metadata) for binding in bindings.values()]
        for day, binding in bindings.items():
            meta = binding.metadata
            liq = meta["liquidity"]
            records["cost_model_selections"].append(dict(session=day,
                decision_session=date.fromisoformat(meta["decision_session"]) if meta["decision_session"] else None,
                model_id=meta["model_id"], snapshot_id=meta["snapshot_id"],
                sample_start=date.fromisoformat(liq["sample_start"]) if liq.get("sample_start") else None,
                sample_end=date.fromisoformat(liq["sample_end"]),
                status="executed" if day in executed_baskets else "not_executed"))
    if signed_kind == "target_notional":
        metadata["target_notional_units"] = market.metadata["currency"]
        metadata["target_notional_sizing"] = "predetermined_currency_amount_divided_by_execution_raw_close_no_scaling"
    return BacktestResult(**{name: (target_table.clone() if name == "targets" and signed and target_table is not None
                             else pl.DataFrame(rows, schema=SCHEMAS[name]))
                             for name, rows in records.items()}, metadata=metadata,
                          status=status, stop_reason=stop_reason,
                          stop_session=stop_session, stop_time=stop_time)
