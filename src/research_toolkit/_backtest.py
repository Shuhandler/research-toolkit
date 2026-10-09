"""Shared daily ledger for buy-and-hold and explicitly scheduled target baskets."""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
import math
import platform

import polars as pl

from ._shorts import LongShortPolicy, StockBorrow, _signed, _signed_targets, _signed_size, _borrow_plan, _cutoff
from ._costs import TradeCosts
from ._execution import SquareRootImpactCosts, COST_SCHEMA, _entry_size, _impact_basket, _bind_cost_schedule
from ._data import UTC, _validated_market
from ._financing import Financing, _below_margin
from ._sofr import SOFRFinancing, _sofr_plan
from ._dividends import DividendReinvestment
from ._portfolio import BuyHoldPolicy, _number, _weights
from ._results import BacktestResult, SignalResult
from ._lifecycle import _market_assets
from ._signals import _checked_signals, _attach_signals, SIGNAL_AUDIT_SCHEMA
from ._rebalancing import RebalancePolicy, _targets, _basket
from ._actions import _ActionLedger, _compile, ACTION_SCHEMAS, CLAIM_COLUMNS
from ._lifecycle import CorporateActionPolicy


from ._version import __version__ as PACKAGE_VERSION


F, S, D, I = pl.Float64, pl.String, pl.Date, pl.Int64
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
              "includes_entry_costs": pl.Boolean, **{c: F for c in CLAIM_COLUMNS}},
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
               "receivable_delta": F, "collateral_delta": F, "dividend_liability_delta": F,
               "claim_cash_delta": F, "claim_quantity_delta": F},
    "valuations": {"session": D, "time": UTC, "phase": S, "market_value": F,
                   "cash": F, "debt": F, "dividend_receivable": F, "dividend_liability": F,
                   "restricted_collateral": F, "short_liability": F, "equity": F,
                   **{c: F for c in CLAIM_COLUMNS}},
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
    **ACTION_SCHEMAS,
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
                 dividend_reinvestment: DividendReinvestment | None = None,
                 corporate_actions: CorporateActionPolicy | None = None,
                 warrant_exercises: pl.DataFrame | None = None) -> BacktestResult:
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

    Markets with lifecycle inputs apply acquisitions, distributions, warrant expiry
    and explicit ``warrant_exercises`` through the same ledger; supply
    ``corporate_actions=CorporateActionPolicy(...)`` whenever they affect the run.
    A held position or claim without a market quote or supplied mark stops the run
    at the last valued close (``stop_reason='unvalued_position'``).

    All results are numerical Polars tables. See ``docs/api.md`` for the complete
    input and output contracts. This function performs no network or file I/O.
    """
    return _simulate(market, weights=weights, initial_capital=initial_capital,
        entry_session=entry_session, end_session=end_session, policy=policy, costs=costs,
        cash_rate=cash_rate, cash_day_count=cash_day_count, financing=financing,
        dividend_reinvestment=dividend_reinvestment, equity_exposures=equity_exposures,
        quantities=quantities, long_short=long_short, stock_borrow=stock_borrow,
        corporate_actions=corporate_actions, warrant_exercises=warrant_exercises)


def scheduled_rebalance(market, *, targets, initial_capital, entry_session, end_session,
                        policy: RebalancePolicy,
                        costs: TradeCosts | SquareRootImpactCosts | Mapping[date, SquareRootImpactCosts],
                        financing: Financing | SOFRFinancing,
                        dividend_reinvestment: DividendReinvestment | None = None,
                        long_short: LongShortPolicy | None = None, stock_borrow: StockBorrow | None = None,
                        corporate_actions: CorporateActionPolicy | None = None,
                        warrant_exercises: pl.DataFrame | None = None):
    """Execute dated long-only or explicitly signed targets through the shared ledger.

    Signed targets require long_short and stock_borrow and use equity_exposure,
    quantity or target_notional columns. Existing long-only target schemas remain unchanged.
    Dollar amounts divide by execution raw prices without rescaling for costs.
    Costs accept one static model or an exact decision-date square-root mapping.
    Target decisions precede execution by default; same-close research execution
    requires explicit policy opt-in. Every basket includes explicit zero exits,
    and no trade executes on the terminal session. See docs/api.md for schemas,
    receivable funding rules, turnover denominators, and research margin stops.
    Corporate actions use the same ``corporate_actions``/``warrant_exercises``
    contract as ``buy_and_hold``; a nonzero target for a security that is not
    quoted at its execution session is rejected as stale.
    """
    if not isinstance(policy, RebalancePolicy) or not isinstance(financing, (Financing, SOFRFinancing)):
        raise ValueError("scheduled_rebalance requires RebalancePolicy and Financing or SOFRFinancing")
    policy.__post_init__()
    market = _validated_market(market)
    if long_short is not None:
        if policy.receivable_policy != "require_target":
            raise ValueError("signed targets require receivable_policy='require_target'; declared loans fund unavailable receivables")
        table, plans, kind = _signed_targets(targets, market, entry_session, end_session, policy)
        first, _, _ = plans[entry_session]
        initial_policy = BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
            fractional_shares=True, initial_gross_leverage=1., terminal_action="mark_only")
        return _simulate(market, weights=None, initial_capital=initial_capital,
            entry_session=entry_session, end_session=end_session, policy=initial_policy, costs=costs,
            financing=financing, schedule=plans, target_table=table, rebalance_policy=policy,
            dividend_reinvestment=dividend_reinvestment, long_short=long_short, stock_borrow=stock_borrow,
            corporate_actions=corporate_actions, warrant_exercises=warrant_exercises,
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
        dividend_reinvestment=dividend_reinvestment, corporate_actions=corporate_actions,
        warrant_exercises=warrant_exercises)

    return _attach_signals(result, signals) if signals else result


def _simulate(market, *, weights, initial_capital, entry_session, end_session, policy,
              costs, cash_rate=None, cash_day_count=None, financing=None,
              schedule=None, target_table=None, rebalance_policy=None, dividend_reinvestment=None,
              equity_exposures=None, quantities=None, long_short=None, stock_borrow=None, target_notionals=None,
              corporate_actions=None, warrant_exercises=None):
    """Validate one run, book its entry, process every calendar date, and return the records."""
    ledger = _Ledger(market, weights=weights, initial_capital=initial_capital,
        entry_session=entry_session, end_session=end_session, policy=policy, costs=costs,
        cash_rate=cash_rate, cash_day_count=cash_day_count, financing=financing,
        schedule=schedule, target_table=target_table, rebalance_policy=rebalance_policy,
        dividend_reinvestment=dividend_reinvestment, equity_exposures=equity_exposures,
        quantities=quantities, long_short=long_short, stock_borrow=stock_borrow,
        target_notionals=target_notionals, corporate_actions=corporate_actions,
        warrant_exercises=warrant_exercises)
    ledger.enter()
    ledger.run()
    return ledger.result()


class _Ledger(_ActionLedger):
    """One simulation's account; every balance change is posted through ``event``.

    Construction validates the configuration in a fixed order and resolves funding,
    allocation, costs and corporate-action schedules. ``enter`` books the entry
    basket, ``run`` processes each calendar date through the final (or stopping)
    close, and ``result`` assembles the typed tables and run metadata.
    """

    def __init__(self, market, *, weights, initial_capital, entry_session, end_session, policy,
                 costs, cash_rate, cash_day_count, financing, schedule, target_table,
                 rebalance_policy, dividend_reinvestment, equity_exposures, quantities,
                 long_short, stock_borrow, target_notionals, corporate_actions=None, warrant_exercises=None):
        self.market = _validated_market(market)
        self.action_policy, self.exercise_input = corporate_actions, warrant_exercises
        self.policy, self.costs, self.financing = policy, costs, financing
        self.entry_session, self.end_session = entry_session, end_session
        self.schedule, self.target_table, self.rebalance_policy = schedule, target_table, rebalance_policy
        self.dividend_reinvestment = dividend_reinvestment
        self.long_short, self.stock_borrow = long_short, stock_borrow
        self.signed = long_short is not None
        self._validate_inputs(weights, equity_exposures, quantities, target_notionals)
        self._resolve_funding(initial_capital, cash_rate, cash_day_count)
        self._resolve_allocation(weights, equity_exposures, quantities, target_notionals)
        self._resolve_costs()
        self._prepare_records()
        self._open_account()

    # ------------------------------------------------------------------ setup

    def _validate_inputs(self, weights, equity_exposures, quantities, target_notionals):
        policy, long_short, financing = self.policy, self.long_short, self.financing
        if sum(v is not None for v in (weights, equity_exposures, quantities, target_notionals)) != 1:
            raise ValueError("supply exactly one of weights, equity_exposures, quantities or scheduled target_notionals")
        if self.signed:
            if not isinstance(long_short, LongShortPolicy) or weights is not None:
                raise ValueError("LongShortPolicy requires signed position targets, not long-allocation weights")
            long_short.__post_init__()
            if not isinstance(financing, (Financing, SOFRFinancing)) or not isinstance(policy, BuyHoldPolicy) or policy.initial_gross_leverage != 1.:
                raise ValueError("signed sizing requires financing and policy.initial_gross_leverage=1; signed targets set the size")
            if financing.maintenance_equity_ratio is not None:
                raise ValueError("signed margin uses LongShortPolicy; set financing.maintenance_equity_ratio=None")
        elif any(v is not None for v in (equity_exposures, quantities, target_notionals, self.stock_borrow)):
            raise ValueError("signed inputs require LongShortPolicy and StockBorrow")
        if self.market.metadata["price_basis"] != "raw":
            raise ValueError("buy_and_hold requires raw execution prices")
        if not isinstance(policy, BuyHoldPolicy) or not isinstance(self.costs, (TradeCosts, SquareRootImpactCosts, Mapping)):
            raise ValueError("explicit BuyHoldPolicy and supported cost model objects are required")
        policy.__post_init__()
        if isinstance(self.costs, Mapping) and self.schedule is None:
            raise ValueError("dated cost models require scheduled_rebalance with explicit decision dates")
        if self.dividend_reinvestment is not None:
            if not isinstance(self.dividend_reinvestment, DividendReinvestment):
                raise ValueError("dividend_reinvestment must be a DividendReinvestment object or None")
            self.dividend_reinvestment.__post_init__()

    def _resolve_funding(self, initial_capital, cash_rate, cash_day_count):
        financing = self.financing
        self.capital = _number(initial_capital, "initial_capital", positive=True)
        self.exposure = exposure = self.policy.initial_gross_leverage
        self.sofr_plan = None
        self.borrowing_denominator, self.cash_denominator = 365, 365
        if financing is not None:
            if not isinstance(financing, (Financing, SOFRFinancing)):
                raise ValueError("financing must be an explicit Financing object or SOFRFinancing object")
            if cash_rate is not None or cash_day_count is not None:
                raise ValueError("do not mix financing with cash_rate/cash_day_count")
            self.rate = financing.cash_rate
            if isinstance(financing, SOFRFinancing):
                # Validate/copy mutable rate inputs below, after checking run dates.
                self.borrowing_rate = None
                self.cash_day_count = financing.cash_day_count
                self.borrowing_denominator = 360 if financing.day_count == "ACT/360" else 365
                self.cash_denominator = 360 if self.cash_day_count == "ACT/360" else 365
            else:
                financing.__post_init__()
                self.borrowing_rate = financing.borrowing_rate
                self.cash_day_count = financing.day_count
            self.threshold = financing.maintenance_equity_ratio
        else:
            if exposure > 1:
                raise ValueError("initial leverage above 1 requires explicit Financing")
            self.rate = _number(cash_rate, "cash_rate")
            if cash_day_count != "ACT/365F":
                raise ValueError("only cash_day_count='ACT/365F' is supported")
            self.cash_day_count = cash_day_count
            self.borrowing_rate, self.threshold = 0.0, None
        if exposure > 1 and self.threshold is None:
            raise ValueError("borrowing requires maintenance_equity_ratio")
        if exposure and _below_margin(1 / exposure, self.threshold):
            raise ValueError("initial leverage violates maintenance_equity_ratio")
        entry_session, end_session = self.entry_session, self.end_session
        if type(entry_session) is not date or type(end_session) is not date:
            raise ValueError("entry_session and end_session must be datetime.date values")
        self.all_sessions = self.market.sessions["session"].to_list()
        if entry_session not in self.all_sessions or end_session not in self.all_sessions or entry_session >= end_session:
            raise ValueError("entry/end must be supplied sessions with at least one holding interval")
        if isinstance(financing, SOFRFinancing):
            self.sofr_plan, self.financing_metadata = _sofr_plan(financing, entry_session, end_session, self.market.metadata["currency"])
        else:
            self.financing_metadata = asdict(financing) if financing else None

    def _resolve_allocation(self, weights, equity_exposures, quantities, target_notionals):
        market, schedule = self.market, self.schedule
        if target_notionals is not None:
            self.signed_kind, signed_input = "target_notional", target_notionals
        elif quantities is not None:
            self.signed_kind, signed_input = "quantity", quantities
        else:
            self.signed_kind, signed_input = "equity_exposure", equity_exposures
        universe = (sorted(_market_assets(market)) if market.securities is not None
                    else market.prices["asset"].unique().to_list())
        self.allocation = (_signed(signed_input, universe, self.signed_kind) if self.signed
                           else _weights(weights, universe))
        self.universe = set(self.allocation)
        self.ca = _compile(market, self.universe, self.action_policy, self.exercise_input, self.entry_session,
                           self.end_session, set(self.all_sessions),
                           isinstance(self.costs, (SquareRootImpactCosts, Mapping)))
        self.borrow_plan, self.borrow_metadata, self.short_assets = {}, None, []
        if self.signed:
            self.short_assets = sorted({a for basket in ([p[0] for p in schedule.values()] if schedule else [self.allocation])
                                        for a, v in basket.items() if v < 0})
            ends = None
            if self.ca is not None:
                ends = self.ca.terminated
                if self.ca.policy is not None and self.ca.policy.short_obligations == "borrowed_short_position":
                    # Short holders of a source become short its delivered securities.
                    shorts = set(self.short_assets)
                    while True:
                        grown = shorts | {leg["asset"] for actions in self.ca.by_date.values() for action in actions
                                          if action["source_asset"] in shorts
                                          for leg in action["legs"] if leg["leg_type"] == "security"}
                        if grown == shorts:
                            break
                        shorts = grown
                    self.short_assets = sorted(shorts)
            self.borrow_plan, self.borrow_metadata = _borrow_plan(
                self.stock_borrow, self.short_assets, self.entry_session, self.end_session, ends)
        self.assets = sorted(self.allocation) if self.ca is None else sorted(self.ca.reachable)
        self.assets_set = set(self.assets)
        total_weight = math.fsum(self.allocation.values())
        self.resolved_weights = (self.allocation.copy() if self.signed else
                                 {a: self.allocation[a] / total_weight for a in self.allocation})
        self.dates = [d for d in self.all_sessions if self.entry_session <= d <= self.end_session]
        self.session_set = set(self.dates)
        self.closes = dict(market.sessions.select("session", "close_at").iter_rows())
        self.prices = {(d, a): p for d, a, p in market.prices.iter_rows()}

    def _resolve_costs(self):
        costs, assets = self.costs, self.assets
        self.impact_costs = isinstance(costs, (SquareRootImpactCosts, Mapping))
        self.bindings, self.executed_baskets = {}, {self.entry_session}
        self.entry_binding = None
        self.rates = None if self.impact_costs else costs.resolve(assets)
        if self.impact_costs:
            executions = ({day: plan[2] for day, plan in self.schedule.items()} if self.schedule
                          else {self.entry_session: None})
            # Lifecycle runs bind each basket to the strategy assets quoted at its execution.
            self.bindings = _bind_cost_schedule(costs, executions, assets, self.prices, self.closes,
                                                self.market.metadata["currency"],
                                                None if self.ca is None else {day: sorted(
                                                    a for a in self.universe if (day, a) in self.prices)
                                                    for day in executions})
            self.entry_binding = self.bindings[self.entry_session]
            entry_weights = (self.resolved_weights if self.ca is None else
                             {a: w for a, w in self.resolved_weights.items() if (self.entry_session, a) in self.prices})
            self.gross = 0. if self.signed else _entry_size(self.capital, self.exposure, entry_weights, self.entry_binding)
        elif self.signed:
            self.gross = 0.
        else:
            rates = self.rates
            fee_rate = math.fsum(self.resolved_weights[a] * math.fsum(rates[c][a] for c in rates) for a in self.resolved_weights)
            self.gross = self.capital * self.exposure / (1 + self.exposure * fee_rate)
        if not math.isfinite(self.gross):
            raise ValueError("initial notional is not representable")

    def _prepare_records(self):
        self.split_dates, self.ex_dates = defaultdict(list), defaultdict(list)
        for action in self.market.splits.iter_rows(named=True):
            if action["asset"] in self.assets_set:
                self.split_dates[action["effective_session"]].append(action)
        for action in self.market.dividends.iter_rows(named=True):
            if action["asset"] in self.assets_set:
                self.ex_dates[action["ex_session"]].append(action)
        self.records = {name: [] for name in SCHEMAS}
        if self.ca is not None:
            # A target cannot trade a security that is not quoted (terminated,
            # suspended, not yet listed) at its execution close.
            baskets = ({day: plan[0] for day, plan in self.schedule.items()} if self.schedule
                       else {self.entry_session: self.allocation})
            for day, basket in baskets.items():
                for asset, target in basket.items():
                    if target and (day, asset) not in self.prices:
                        status = ("terminated" if asset in self.ca.terminated and day >= self.ca.terminated[asset]
                                  else "not quoted")
                        raise ValueError(f"stale target: {asset} is {status} on {day} and cannot be traded")
        if self.target_table is not None:
            self.records["targets"] = self.target_table.to_dicts()
            # Validate all cost/leverage combinations before processing the first fill.
            for target_weights, target_leverage, _ in self.schedule.values():
                if not self.impact_costs and not self.signed:
                    _basket({a: 0. for a in self.assets}, self.capital, 0., target_weights,
                            target_leverage, self.rates, self.rebalance_policy.receivable_policy)

    def _open_account(self):
        self.quantity = {a: 0.0 for a in self.assets}
        self.cash = self.capital
        self.debt = 0.0
        self.collateral = 0.0
        self.liabilities = {}
        self.receivables = {}
        self.reinvestment_rows, self.reinvestment_budgets = {}, {}
        self.pending_pnl = defaultdict(list)
        self.cash_deltas, self.receivable_deltas, self.debt_deltas = [], [], []
        self.collateral_deltas, self.liability_deltas = [], []
        self.quantity_deltas = defaultdict(list)
        self.claims, self.deferred, self.action_held, self.applied = {}, [], set(), set()
        self.claim_counter, self.unvalued = 0, None
        self.claim_cash_deltas, self.claim_quantity_deltas = [], defaultdict(list)
        self._delta_lists = ("cash_deltas", "receivable_deltas", "debt_deltas", "collateral_deltas",
                             "liability_deltas", "claim_cash_deltas")
        self.status, self.stop_reason, self.stop_session, self.stop_time = "complete", None, None, None

    # ------------------------------------------------------------- posting

    def event(self, day, kind, *, phase="before_close", asset=None, action_id=None,
              trade_id=None, dq=0.0, dc=0.0, dr=0.0, dd=0.0, dk=0.0, dl=0.0, dcc=0.0, dcq=0.0):
        events = self.records["events"]
        seq = len(events)
        events.append({
            "event_id": f"E{seq:06d}", "date": day,
            "time": self.closes[day] if phase in {"entry_close", "rebalance_close", "reinvestment_close", "close"} else None,
            "sequence": seq, "phase": phase, "type": kind, "asset": asset,
            "action_id": action_id, "trade_id": trade_id,
            "quantity_delta": dq, "cash_delta": dc, "debt_delta": dd,
            "receivable_delta": dr, "collateral_delta": dk, "dividend_liability_delta": dl,
            "claim_cash_delta": dcc, "claim_quantity_delta": dcq,
        })
        self.claim_cash_deltas.append(dcc)
        if dcq:
            self.claim_quantity_deltas[asset].append(dcq)
        self.collateral_deltas.append(dk)
        self.liability_deltas.append(dl)
        self.cash_deltas.append(dc)
        self.receivable_deltas.append(dr)
        self.debt_deltas.append(dd)
        if asset is not None:
            self.quantity_deltas[asset].append(dq)

    def balances(self, day, phase):
        """Record a valuation and reconcile every balance with its posted events."""
        assets, quantity, capital = self.assets, self.quantity, self.capital
        cash, collateral, debt = self.cash, self.collateral, self.debt
        values = ({a: quantity[a] * self.prices[day, a] for a in assets} if self.ca is None
                  else {a: self.value_of(day, a) for a in assets})
        mv = math.fsum(values.values())
        receivable = math.fsum(item["outstanding"] for item in self.receivables.values())
        liability = math.fsum(item["outstanding"] for item in self.liabilities.values())
        claim_values = self.claim_totals() if self.ca is not None else (0., 0., 0., 0.)
        net_claims = math.fsum([claim_values[0], -claim_values[1], claim_values[2], -claim_values[3]])
        equity = (math.fsum([mv, cash, collateral, receivable, -liability, -debt]) if self.ca is None
                  else math.fsum([mv, cash, collateral, receivable, -liability, -debt, net_claims]))
        if (not all(math.isfinite(v) for v in (mv, cash, collateral, receivable, liability, debt, equity, *claim_values))
                or min(cash, collateral, receivable, liability, debt) < 0):
            raise ArithmeticError("ledger produced invalid balances")
        if phase in {"pre_entry", "post_entry"} and equity <= 0:
            raise ValueError("initial equity after costs must be positive")
        self.records["valuations"].append({"session": day, "time": self.closes[day], "phase": phase,
                                           "market_value": mv, "cash": cash, "debt": debt,
                                           "dividend_receivable": receivable, "dividend_liability": liability,
                                           "restricted_collateral": collateral, "short_liability": sum(max(-v, 0.) for v in values.values()), "equity": equity,
                                           **dict(zip(CLAIM_COLUMNS, claim_values))})
        checks = {
            "cash_reconciliation": (cash, math.fsum([capital, *self.cash_deltas])),
            "receivable_reconciliation": (receivable, math.fsum(self.receivable_deltas)),
            "debt_reconciliation": (debt, math.fsum(self.debt_deltas)),
            "collateral_reconciliation": (collateral, math.fsum(self.collateral_deltas)),
            "dividend_liability_reconciliation": (liability, math.fsum(self.liability_deltas)),
            "balance_sheet": (equity, math.fsum([*values.values(), cash, collateral, receivable, -liability, -debt])
                              if self.ca is None else
                              math.fsum([*values.values(), cash, collateral, receivable, -liability, -debt, net_claims])),
        }
        if self.ca is not None:
            checks["claim_cash_reconciliation"] = (
                math.fsum(c["amount"] for c in self.claims.values() if c["kind"] == "cash"),
                math.fsum(self.claim_cash_deltas))
            for a in assets:
                pending = math.fsum(c["quantity"] for c in self.claims.values() if c["kind"] == "security" and c["asset"] == a)
                if not math.isclose(pending, math.fsum(self.claim_quantity_deltas[a]), rel_tol=1e-12, abs_tol=1e-12):
                    raise ArithmeticError(f"claim quantity reconciliation failed for {a}")
        if self.signed:
            checks["marked_collateral"] = (collateral, self.long_short.collateral_multiple *
                (math.fsum(max(-v, 0.) for v in values.values()) if self.ca is None
                 else math.fsum([*(max(-v, 0.) for v in values.values()), self.obligations()])))
        for a in assets:
            if not math.isclose(quantity[a], math.fsum(self.quantity_deltas[a]), rel_tol=1e-12, abs_tol=1e-12):
                raise ArithmeticError(f"quantity reconciliation failed for {a}")
        for code, (actual, expected) in checks.items():
            residual, tolerance = _check(actual, expected, max(capital, equity), code)
            self.records["diagnostics"].append({"session": day, "code": f"{phase}:{code}",
                                                "residual": residual, "tolerance": tolerance})
        return values, receivable, equity

    def fund(self, day, amount, phase="before_close"):
        if amount > 0:
            self.cash += amount
            self.debt += amount
            self.event(day, "borrowing", phase=phase, dc=amount, dd=amount)

    def available_cash(self):
        return max(0., self.cash-math.fsum(self.reinvestment_budgets.values()))

    def mark_collateral(self, day, values, phase="close"):
        shorts = math.fsum(max(-v, 0.) for v in values.values())
        if self.ca is not None:
            # Pending delivery/cash obligations of short holders stay collateralized
            # until they are actually settled, not merely when a ticker disappears.
            shorts = math.fsum([shorts, self.obligations()])
        required = self.long_short.collateral_multiple * shorts
        change = required-self.collateral
        self.fund(day, max(change-self.available_cash(), 0.), phase)
        self.cash -= change
        self.collateral = required
        if change:
            self.event(day, "collateral_transfer", phase=phase, dc=-change, dk=change)

    def margin_required(self, values):
        long_short = self.long_short
        if self.ca is not None and self.ca.policy is not None:
            policy, warrants = self.ca.policy, self.ca.warrants
            pos_cash, neg_cash, pos_sec, neg_sec = self.claim_totals()
            return math.fsum([
                long_short.long_margin*math.fsum(max(v, 0.) for a, v in values.items() if a not in warrants),
                policy.warrant_margin*math.fsum(max(v, 0.) for a, v in values.items() if a in warrants),
                long_short.short_margin*math.fsum(max(-v, 0.) for v in values.values()),
                policy.pending_claim_margin*(pos_cash+pos_sec), policy.obligation_margin*(neg_cash+neg_sec)])
        return (long_short.long_margin*math.fsum(max(v, 0.) for v in values.values()) +
                long_short.short_margin*math.fsum(max(-v, 0.) for v in values.values()))

    def breached(self, values, equity):
        if self.signed:
            return _below_margin(equity, self.margin_required(values))
        gross_value = math.fsum(values.values())
        if self.ca is not None:
            # Undelivered securities are risky exposure; cash claims are like receivables.
            gross_value = math.fsum([gross_value, self.claim_totals()[2]])
        return _below_margin(equity/gross_value if gross_value else None, self.threshold)

    def position_rows(self, day, values, equity):
        for asset in self.assets:
            self.records["positions"].append({"session": day, "asset": asset,
                                              "quantity": self.quantity[asset],
                                              "raw_mark": self.prices[day, asset] if self.ca is None else self.mark(day, asset),
                                              "market_value": values[asset],
                                              "weight": values[asset] / equity if equity > 0 else None})

    # -------------------------------------------------------------- trading

    def signed_basket(self, day, values, equity, targets, phase, decision=None):
        """Execute one atomic signed basket: release collateral, trade, re-segregate, sweep."""
        quantity, records, capital = self.quantity, self.records, self.capital
        if self.ca is None:
            assets, base = self.assets, equity
        else:
            # Only quoted strategy assets trade; other holdings and claims stay outside.
            assets, outside = self._basket_scope(day, targets, values)
            # A long sleeve is excluded from the sizing base; a short sleeve never enlarges it.
            base = equity-max(outside, 0.) if self.signed_kind == "equity_exposure" else equity
        targets = {a: targets[a] for a in assets}
        marks = {a: self.prices[day, a] for a in assets}
        old_quantities, fills = quantity.copy(), {}
        binding = self.bindings[day] if self.impact_costs else None
        rates = self.rates

        def components(a, n):
            return binding.components(a, n) if binding else {c: abs(n)*rates[c][a] for c in rates}

        def marginal(a, n):
            return binding.marginal(a, n) if binding else math.fsum(rates[c][a] for c in rates)

        changes, fee = _signed_size({a: values[a] for a in assets}, base, targets, self.signed_kind, marks, components, marginal)
        new_values = {a: values[a]+changes.get(a, 0.) for a in values}
        after = equity-fee
        if not all(math.isfinite(v) for v in [after, *new_values.values(), *changes.values()]) or after <= 0:
            raise ValueError("signed basket equity/positions are not representable")
        gross_value = math.fsum(abs(new_values[a]) for a in assets)
        if self.rebalance_policy and gross_value and any(abs(new_values[a])/gross_value > self.rebalance_policy.max_asset_weight+1e-12 for a in assets):
            raise ValueError("signed target exceeds max_asset_weight as a share of gross exposure")
        if phase == "entry_close" and self.breached(new_values, after):
            raise ValueError("signed entry violates long/short margin requirements")
        # Release collateral only within this atomic basket, then segregate the new
        # marked requirement before any debt sweep. Sale proceeds never enter a sweep.
        if self.collateral:
            self.cash += self.collateral
            self.event(day, "collateral_release_for_basket", phase=phase, dc=self.collateral, dk=-self.collateral)
            self.collateral = 0.
        required = self.long_short.collateral_multiple*(math.fsum(max(-v, 0.) for v in new_values.values())
            if self.ca is None else math.fsum([*(max(-v, 0.) for v in new_values.values()), self.obligations()]))
        self.fund(day, max(math.fsum([*changes.values(), fee, required, -self.cash]), 0.), phase)
        for a in sorted(changes, key=lambda a: (changes[a] >= 0, a)):
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
            self.cash -= n+charge
            self.event(day, "trade", phase=phase, asset=a, trade_id=trade_id, dq=delta, dc=-n)
            records["trades"].append(dict(trade_id=trade_id, session=day, time=self.closes[day], asset=a,
                signed_quantity=delta, reference_price=marks[a], signed_notional=n,
                execution="entry_close" if phase == "entry_close" else "scheduled_close", trade_cost=charge))
            if binding:
                records["execution_costs"].append(dict(binding.audit_row(a, n), trade_id=trade_id, session=day))
            for component, amount in parts.items():
                if amount:
                    records["costs"].append(dict(cost_id=f"C{len(records['costs']):06d}", date=day,
                        time=self.closes[day], trade_id=trade_id, asset=a, component=component, amount=amount, basis="modeled"))
                    self.event(day, component, phase=phase, asset=a, trade_id=trade_id, dc=-amount)
                    self.pending_pnl[component, a].append(-amount)
        self.mark_collateral(day, new_values, phase)
        # Roundoff is reconciled before normalization; all material deficits fail.
        if self.cash < 0:
            _check(self.cash, 0., capital, "signed_cash_roundoff")
            self.cash = 0.
        repayment = min(self.cash, self.debt)
        if repayment:
            self.cash -= repayment
            self.debt -= repayment
            self.event(day, "debt_repayment", phase=phase, dc=-repayment, dd=-repayment)
        actual_equity = (math.fsum([*(quantity[a]*marks[a] for a in assets), self.cash, self.collateral,
            math.fsum(v["outstanding"] for v in self.receivables.values()),
            -math.fsum(v["outstanding"] for v in self.liabilities.values()), -self.debt]) if self.ca is None
            else self.current_equity({a: self.value_of(day, a) for a in self.assets}))
        residual, tolerance = _check(actual_equity, after, max(capital, equity), "signed_basket_cost_equity")
        records["diagnostics"].append(dict(session=day, code="signed_basket_cost_equity", residual=residual, tolerance=tolerance))
        if self.signed_kind == "target_notional":
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
        self.executed_baskets.add(day)
        traded = math.fsum(abs(n) for n in changes.values())
        records["turnover"].append(dict(session=day, phase="entry" if phase == "entry_close" else "rebalance",
            gross_traded_notional=traded, equity_before=equity, turnover=traded/equity))
        if phase != "entry_close":
            records["rebalances"].append(dict(session=day, decision_session=decision, equity_before=equity,
                equity_after=after, target_gross_leverage=math.fsum(abs(v) for v in targets.values()) if self.signed_kind == "equity_exposure" else None,
                actual_gross_leverage=gross_value/after, trade_cost=fee, gross_traded_notional=traded, receivable_reserved=0.))
        return fee

    def enter(self):
        """Book the entry basket at the entry close and record post-entry balances."""
        entry_session, capital = self.entry_session, self.capital
        self.balances(entry_session, "pre_entry")
        if self.signed:
            self.entry_cost = self.signed_basket(entry_session, {a: 0. for a in self.assets}, capital, self.allocation,
                "entry_close", self.schedule[entry_session][2] if self.schedule else None)
        else:
            self._enter_long_basket()
        values, _, post_entry_equity = self.balances(entry_session, "post_entry")
        _check(post_entry_equity, capital - self.entry_cost, capital, "entry_cost_equity")
        if self.breached(values, post_entry_equity):
            raise ValueError("funded entry violates maintenance_equity_ratio")
        self.position_rows(entry_session, values, post_entry_equity)
        if not self.signed:
            self.records["turnover"].append({"session": entry_session, "phase": "entry",
                "gross_traded_notional": self.entry_notional, "equity_before": capital,
                "turnover": self.entry_notional/capital})
        self.previous_values, self.opening_equity, self.previous_session = values, capital, entry_session
        self.cumulative_pnls, self.simple_returns = [], []
        self.wealth = 1.0
        self.peak = max(capital, post_entry_equity)
        if self.ca is not None:
            self._checkpoint()

    def _enter_long_basket(self):
        entry_session, capital, records = self.entry_session, self.capital, self.records
        rates = self.rates
        entry_plan = []
        for asset in self.resolved_weights:
            notional = self.gross * self.resolved_weights[asset]
            if notional == 0:
                continue
            mark = self.prices[entry_session, asset]
            qty = notional / mark
            if not math.isfinite(qty) or qty <= 0:
                raise ValueError(f"initial quantity is not representable for {asset}")
            components = self.entry_binding.components(asset, notional) if self.impact_costs else {c: notional * rates[c][asset] for c in rates}
            cost = math.fsum(components.values())
            entry_plan.append((asset, notional, mark, qty, components, cost))
        self.entry_cost = math.fsum(item[5] for item in entry_plan)
        self.entry_notional = math.fsum(item[1] for item in entry_plan)
        # Fund the whole order basket before any trade event; borrowing is not profit.
        if self.exposure > 1:
            self.debt = max(math.fsum([self.entry_notional, self.entry_cost, -capital]), 0.0)
            if not math.isfinite(self.debt) or self.debt <= 0:
                raise ValueError("initial borrowing is not representable")
            self.cash += self.debt
            self.event(entry_session, "borrowing", phase="entry_close", dc=self.debt, dd=self.debt)
        for asset, notional, mark, qty, components, cost in entry_plan:
            trade_id = f"T{len(records['trades']):06d}"
            if self.impact_costs:
                records["execution_costs"].append(dict(self.entry_binding.audit_row(asset, notional), trade_id=trade_id, session=entry_session))
            self.quantity[asset] = qty
            self.event(entry_session, "trade", phase="entry_close", asset=asset,
                       trade_id=trade_id, dq=qty, dc=-notional)
            records["trades"].append({"trade_id": trade_id, "session": entry_session,
                                      "time": self.closes[entry_session], "asset": asset,
                                      "signed_quantity": qty, "reference_price": mark,
                                      "signed_notional": notional, "execution": self.policy.execution,
                                      "trade_cost": cost})
            for component, amount in components.items():
                if amount == 0:
                    continue
                records["costs"].append({"cost_id": f"C{len(records['costs']):06d}",
                                         "date": entry_session, "time": self.closes[entry_session],
                                         "trade_id": trade_id, "asset": asset,
                                         "component": component, "amount": amount, "basis": "modeled"})
                self.event(entry_session, component, phase="entry_close", asset=asset,
                           trade_id=trade_id, dc=-amount)
                self.pending_pnl[component, asset].append(-amount)
        self.cash = math.fsum([capital, self.debt, -self.entry_notional, -self.entry_cost])
        # At full investment, only floating-point subtraction can leave residual cash.
        if self.exposure >= 1 or self.cash < 0:
            residual, tolerance = _check(self.cash, 0.0, capital, "entry_cash_roundoff")
            self.cash = 0.0
            records["diagnostics"].append({"session": entry_session, "code": "entry_cash_roundoff",
                                           "residual": residual, "tolerance": tolerance})

    def rebalance(self, day, values, outstanding, equity):
        """Trade a long-only scheduled basket toward its post-cost target."""
        target_weights, target_leverage, decision = self.schedule[day]
        self.executed_baskets.add(day)
        if self.signed:
            self.signed_basket(day, values, equity, target_weights, "rebalance_close", decision)
            return
        records, capital, policy = self.records, self.capital, self.rebalance_policy
        binding = self.bindings[day] if self.impact_costs else None
        rates = self.rates
        basket_values, basket_equity = values, equity
        if self.ca is not None:
            # Retained distributed securities and pending claims form an untraded sleeve.
            scope, outside = self._basket_scope(day, target_weights, values)
            target_weights = {a: target_weights[a] for a in scope}
            basket_values, basket_equity = {a: values[a] for a in scope}, equity-outside
        if self.impact_costs:
            changes, cost = _impact_basket(basket_values, basket_equity, outstanding, target_weights,
                target_leverage, binding, policy.receivable_policy)
        else:
            changes, cost = _basket(basket_values, basket_equity, outstanding, target_weights,
                target_leverage, rates, policy.receivable_policy)
        new_net_cash = math.fsum([self.cash, -self.debt, -math.fsum(changes.values()), -cost])
        new_debt = max(-new_net_cash, 0.)
        tolerance = min(.009, 1e-8 + 1e-12*max(capital, equity))
        if new_debt <= tolerance and target_leverage <= 1:
            new_debt = 0.
        if target_leverage <= 1 and new_debt > tolerance and (
                policy.receivable_policy == "require_target" or new_debt > self.debt + tolerance):
            raise ValueError(f"target on {day} requires financing unspendable receivables; use reserve or hold cash")
        if new_debt > 0 and self.threshold is None:
            raise ValueError("borrowing requires maintenance_equity_ratio")
        # Borrow only the funded basket's required net debt, execute sales before
        # purchases, then repay excess debt. No transient funding deficits.
        if new_debt > self.debt:
            borrowing = new_debt-self.debt
            self.cash += borrowing
            self.debt += borrowing
            self.event(day, "borrowing", phase="rebalance_close", dc=borrowing, dd=borrowing)
        for asset in sorted(changes, key=lambda a: (changes[a] >= 0, a)):
            notional = changes[asset]
            if notional == 0:
                continue
            mark = self.prices[day, asset]
            delta = notional/mark
            if values[asset]+notional == 0.:
                delta = -self.quantity[asset]
            next_quantity = self.quantity[asset]+delta
            if next_quantity < 0 and math.isclose(next_quantity, 0., abs_tol=1e-12):
                next_quantity = 0.
            if next_quantity < 0 or not math.isfinite(next_quantity):
                raise ValueError("scheduled quantity is not representable")
            component_costs = binding.components(asset, notional) if self.impact_costs else {c: abs(notional)*rates[c][asset] for c in rates}
            trade_cost = math.fsum(component_costs.values())
            trade_id = f"T{len(records['trades']):06d}"
            if self.impact_costs:
                records["execution_costs"].append(dict(binding.audit_row(asset, notional), trade_id=trade_id, session=day))
            self.quantity[asset] = next_quantity
            self.cash -= notional+trade_cost
            self.event(day, "trade", phase="rebalance_close", asset=asset, trade_id=trade_id,
                       dq=delta, dc=-notional)
            records["trades"].append({"trade_id": trade_id, "session": day, "time": self.closes[day],
                "asset": asset, "signed_quantity": delta, "reference_price": mark,
                "signed_notional": notional, "execution": "scheduled_close", "trade_cost": trade_cost})
            for component, amount in component_costs.items():
                if not amount:
                    continue
                records["costs"].append({"cost_id": f"C{len(records['costs']):06d}", "date": day,
                    "time": self.closes[day], "trade_id": trade_id, "asset": asset,
                    "component": component, "amount": amount, "basis": "modeled"})
                self.event(day, component, phase="rebalance_close", asset=asset, trade_id=trade_id, dc=-amount)
                self.pending_pnl[component, asset].append(-amount)
            if self.cash < -tolerance:
                raise ArithmeticError("scheduled trade basket has an unfunded purchase")
        if self.debt > new_debt:
            repayment = self.debt-new_debt
            self.debt = new_debt
            self.cash -= repayment
            self.event(day, "debt_repayment", phase="rebalance_close", dc=-repayment, dd=-repayment)
        expected_cash = max(new_net_cash, 0.)
        residual, tol = _check(self.cash, expected_cash, max(capital, equity), "rebalance_cash_roundoff")
        records["diagnostics"].append({"session": day, "code": "rebalance_cash_roundoff",
                                       "residual": residual, "tolerance": tol})
        self.cash = expected_cash
        if self.ca is None:
            gross = math.fsum(self.quantity[a]*self.prices[day, a] for a in self.assets)
            after = math.fsum([gross, self.cash, outstanding, -self.debt])
        else:
            gross = math.fsum(self.value_of(day, a) for a in self.assets)
            after = math.fsum([gross, self.cash, outstanding, -self.debt, self.claims_net()])
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
                if target_leverage <= 1 and policy.receivable_policy == "reserve" else 0.)})
        records["turnover"].append({"session": day, "phase": "rebalance", "gross_traded_notional": traded,
                                    "equity_before": equity, "turnover": traded/equity})

    # ---------------------------------------------------- dividend reinvestment

    def release_reinvestment(self, day, reason):
        # Release an earmark, not cash itself: it was already credited at payment.
        for action_id, budget in self.reinvestment_budgets.items():
            self.reinvestment_rows[action_id].update(session=day, cash_released=budget, status=reason)
        self.reinvestment_budgets.clear()

    def sweep_cash(self, day):
        budgets = self.reinvestment_budgets
        reserved = math.fsum(budgets.values())
        if self.dividend_reinvestment is None or self.dividend_reinvestment.funding == "after_debt_repayment":
            reserved = 0.
        if self.debt > 0 and self.cash > reserved:
            repayment = min(self.cash-reserved, self.debt)
            self.cash -= repayment
            self.debt -= repayment
            self.event(day, "debt_repayment", dc=-repayment, dd=-repayment)
            # Unreserved account cash pays first. If dividend cash is needed,
            # distribute the reduction pro rata across outstanding earmarks.
            total = math.fsum(budgets.values())
            used = min(total, max(0., total-self.cash))
            if used:
                for action_id, budget in list(budgets.items()):
                    reduction = budget*(used/total)
                    remaining = budget-reduction
                    self.reinvestment_rows[action_id]["debt_repaid"] += reduction
                    if remaining == 0.:
                        self.reinvestment_rows[action_id]["status"] = "debt_repaid"
                        del budgets[action_id]
                    else:
                        budgets[action_id] = remaining

    def reinvest(self, day, equity):
        records, capital = self.records, self.capital
        notionals = []
        for action_id, budget in self.reinvestment_budgets.items():
            row = self.reinvestment_rows[action_id]
            asset = row["asset"]
            if self.ca is not None and (day, asset) not in self.prices:
                # Never reinvest into a terminated, suspended or unquoted payer.
                row.update(session=day, cash_released=budget, status="payer_not_quoted")
                continue
            if self.quantity[asset] < 0:
                row.update(session=day, cash_released=budget, status="payer_now_short")
                continue
            # User-selected DRIP convention: no commission, spread or impact.
            # Ordinary entry and scheduled trades retain their configured costs.
            notional = budget
            delta = notional/self.prices[day, asset]
            updated_quantity = self.quantity[asset]+delta
            if (not all(math.isfinite(v) and v > 0 for v in (notional, delta, updated_quantity))
                    or updated_quantity <= self.quantity[asset]):
                raise ValueError("dividend reinvestment quantity is not representable")
            trade_id = f"T{len(records['trades']):06d}"
            self.quantity[asset] = updated_quantity
            self.cash -= budget
            self.event(day, "trade", phase="reinvestment_close", asset=asset, action_id=action_id,
                       trade_id=trade_id, dq=delta, dc=-notional)
            records["trades"].append({"trade_id": trade_id, "session": day, "time": self.closes[day],
                "asset": asset, "signed_quantity": delta, "reference_price": self.prices[day, asset],
                "signed_notional": notional, "execution": "dividend_reinvestment_close", "trade_cost": 0.})
            row.update(session=day, signed_notional=notional, trade_cost=0.,
                       trade_id=trade_id, status="reinvested")
            notionals.append(notional)
        if self.cash < 0:
            _check(self.cash, 0., max(capital, equity), "reinvestment_cash_roundoff")
            self.cash = 0.
        outstanding = math.fsum(item["outstanding"] for item in self.receivables.values())
        after = (math.fsum([*(self.quantity[a]*self.prices[day, a] for a in self.assets), self.cash, self.collateral, outstanding,
            -math.fsum(v["outstanding"] for v in self.liabilities.values()), -self.debt]) if self.ca is None
            else self.current_equity({a: self.value_of(day, a) for a in self.assets}))
        residual, tolerance = _check(after, equity, max(capital, equity), "reinvestment_equity_neutrality")
        records["diagnostics"].append({"session": day, "code": "reinvestment_equity_neutrality",
                                       "residual": residual, "tolerance": tolerance})
        traded = math.fsum(notionals)
        records["turnover"].append({"session": day, "phase": "dividend_reinvestment",
            "gross_traded_notional": traded, "equity_before": equity, "turnover": traded/equity})
        self.reinvestment_budgets.clear()
        if self.signed:
            self.sweep_cash(day)

    # ------------------------------------------------------------ daily loop

    def run(self):
        """Process every calendar date after entry through the end or the stopping close."""
        day = self.entry_session + timedelta(days=1)
        while day <= self.end_session:
            self._accrue_interest(day)
            if self.signed:
                self._accrue_stock_loan(day)
            self._apply_corporate_actions(day)
            if self.ca is not None:
                self._apply_lifecycle_actions(day)
            self._settle_payments(day)
            if self.ca is not None:
                self._settle_claims(day)
            if day == self.end_session:
                self.release_reinvestment(day, "terminal_cash")
            elif self.schedule and day in self.schedule:
                self.release_reinvestment(day, "scheduled_rebalance")
            self.sweep_cash(day)
            if day in self.session_set and self._close_session(day):
                break
            if day == self.end_session:
                break
            day += timedelta(days=1)
        for row in self.records["dividend_reinvestments"]:
            residual, tolerance = _check(row["paid_amount"], math.fsum(row[k] for k in (
                "debt_repaid", "cash_released", "signed_notional", "trade_cost")), self.capital, "dividend_budget_allocation")
            self.records["diagnostics"].append({"session": row["session"] or row["pay_date"],
                "code": "dividend_budget_allocation", "residual": residual, "tolerance": tolerance})

    def _accrue_interest(self, day):
        """Credit opening cash and capitalize opening debt for one calendar date."""
        daily_borrowing_rate = self.sofr_plan[day]["borrowing_rate"] if self.sofr_plan is not None else self.borrowing_rate
        interest = self.cash * (self.rate / self.cash_denominator)
        borrowing_charge = self.debt * (daily_borrowing_rate / self.borrowing_denominator)
        if self.sofr_plan is not None:
            self.records["financing_accruals"].append({"date": day, **self.sofr_plan[day],
                "cash_rate": self.rate, "borrowing_day_fraction": 1/self.borrowing_denominator,
                "cash_day_fraction": 1/self.cash_denominator, "opening_debt": self.debt, "opening_cash": self.cash,
                "borrowing_interest": borrowing_charge, "cash_interest": interest})
        if interest:
            self.cash += interest
            self.event(day, "cash_interest", dc=interest)
            self.pending_pnl["cash_interest", None].append(interest)
        if borrowing_charge:
            self.debt += borrowing_charge
            self.event(day, "borrowing_interest", dd=borrowing_charge)
            self.records["costs"].append({"cost_id": f"C{len(self.records['costs']):06d}",
                                          "date": day, "time": None, "trade_id": None, "asset": None,
                                          "component": "borrowing_interest", "amount": borrowing_charge,
                                          "basis": "modeled"})
            self.pending_pnl["borrowing_interest", None].append(-borrowing_charge)

    def _accrue_stock_loan(self, day):
        """Charge borrow fees on prior closing short values, then pay the gross collateral rebate."""
        records, stock_borrow, long_short = self.records, self.stock_borrow, self.long_short
        denominator = 360 if stock_borrow.day_count == "ACT/360" else 365
        for asset in self.short_assets:
            if self.ca is not None and asset in self.ca.terminated and day >= self.ca.terminated[asset]:
                continue  # An extinguished security is no longer borrowed; obligations are not loans.
            annual, available = self.borrow_plan[day, asset]
            short_value = max(-self.previous_values[asset], 0.)
            amount = short_value*annual/denominator
            records["stock_borrow_accruals"].append(dict(date=day, asset=asset, mark_session=self.previous_session,
                cutoff_at=_cutoff(day), available_at=available, short_market_value=short_value,
                annual_rate=annual, day_fraction=1/denominator, amount=amount))
            if amount:
                self.fund(day, max(amount-self.available_cash(), 0.))
                self.cash -= amount
                self.event(day, "stock_borrow_fee", asset=asset, dc=-amount)
                records["costs"].append(dict(cost_id=f"C{len(records['costs']):06d}", date=day,
                    time=None, trade_id=None, asset=asset, component="stock_borrow_fee", amount=amount,
                    basis=stock_borrow.metadata["basis"]))
                self.pending_pnl["stock_borrow_fee", asset].append(-amount)
        denominator = 360 if long_short.rebate_day_count == "ACT/360" else 365
        rebate = self.collateral*long_short.rebate_rate/denominator
        records["short_financing_accruals"].append(dict(date=day, opening_collateral=self.collateral,
            rebate_rate=long_short.rebate_rate, day_fraction=1/denominator, rebate=rebate))
        if rebate:
            self.cash += rebate
            self.event(day, "short_collateral_rebate", dc=rebate)
            self.pending_pnl["short_collateral_rebate", None].append(rebate)

    def _apply_corporate_actions(self, day):
        """Apply splits, then accrue ex-date entitlements and short obligations."""
        quantity = self.quantity
        if self.ca is not None:
            for action in [*self.split_dates[day], *self.ex_dates[day]]:
                if any(c["kind"] == "security" and c["asset"] == action["asset"] for c in self.claims.values()):
                    raise ValueError(f"{action['action_id']} on {day} affects {action['asset']} while a delivery "
                                     "claim on it is pending; pending claims are not re-denominated")
        for action in self.split_dates[day]:
            asset = action["asset"]
            old = quantity[asset]
            quantity[asset] *= action["ratio"]
            if old and (not math.isfinite(quantity[asset]) or quantity[asset] == 0):
                raise ValueError(f"split quantity is not representable for {asset} on {day}")
            if old:
                self.event(day, "split", asset=asset, action_id=action["action_id"], dq=quantity[asset] - old)
        for action in self.ex_dates[day]:
            asset = action["asset"]
            amount = quantity[asset] * action["cash_per_share"]
            if amount < 0:
                self.liabilities[action["action_id"]] = {**action, "entitled_quantity": quantity[asset],
                    "amount": -amount, "outstanding": -amount}
                self.event(day, "short_dividend_accrual", asset=asset, action_id=action["action_id"], dl=-amount)
                self.pending_pnl["short_dividend_expense", asset].append(amount)
            elif amount:
                self.receivables[action["action_id"]] = {
                    **action, "entitled_quantity": quantity[asset], "amount": amount, "outstanding": amount,
                }
                self.event(day, "dividend_accrual", asset=asset, action_id=action["action_id"], dr=amount)
                self.pending_pnl["dividend_income", asset].append(amount)

    def _settle_payments(self, day):
        """Collect long dividends (earmarking reinvestment budgets) and pay short dividends."""
        for action_id, item in self.receivables.items():
            if item["pay_date"] == day and item["outstanding"]:
                amount = item["outstanding"]
                self.cash += amount
                item["outstanding"] = 0.0
                self.event(day, "dividend_payment", asset=item["asset"], action_id=action_id, dc=amount, dr=-amount)
                if self.dividend_reinvestment is not None:
                    row = {"action_id": action_id, "asset": item["asset"], "pay_date": day,
                        "session": None, "paid_amount": amount, "debt_repaid": 0., "cash_released": 0.,
                        "signed_notional": 0., "trade_cost": 0., "trade_id": None, "status": "pending"}
                    self.reinvestment_rows[action_id] = row
                    self.records["dividend_reinvestments"].append(row)
                    self.reinvestment_budgets[action_id] = amount
        for action_id, item in self.liabilities.items():
            if item["pay_date"] == day and item["outstanding"]:
                amount = item["outstanding"]
                self.fund(day, max(amount-self.available_cash(), 0.))
                self.cash -= amount
                item["outstanding"] = 0.
                self.event(day, "short_dividend_payment", asset=item["asset"], action_id=action_id, dc=-amount, dl=-amount)

    def _close_session(self, day):
        """Mark, trade any scheduled or reinvestment basket, close and record the session.

        Returns True when the close breaches margin or equity is nonpositive.
        """
        assets, records, capital = self.assets, self.records, self.capital
        if self.ca is not None:
            missing = self._unvalued(day)
            if missing:
                return self._stop_unvalued(day, missing)
            self._recognize_deferred(day)
        # Old holdings earn the interval's price move. Rebalancing at this
        # close affects only future price P&L, with trade costs booked today.
        before = ({a: self.quantity[a]*self.prices[day, a] for a in assets} if self.ca is None
                  else {a: self.value_of(day, a) for a in assets})
        for asset in assets:
            self.pending_pnl["price_pnl", asset].append(before[asset] - self.previous_values[asset])
        if self.ca is not None:
            self._value_claims(day)
            self._settle_obligations_at_close(day)
        if self.signed:
            self.mark_collateral(day, before)
            self.sweep_cash(day)
        if self.ca is not None:
            self._action_trades(day)
        if self.schedule and day in self.schedule:
            before, outstanding, before_equity = self.balances(day, "pre_rebalance")
            self.peak = max(self.peak, before_equity)
            # A pre-trade failure cannot be hidden by a scheduled deleveraging.
            if before_equity > 0 and not self.breached(before, before_equity):
                self.rebalance(day, before, outstanding, before_equity)
        elif self.reinvestment_budgets:
            before, outstanding, before_equity = self.balances(day, "pre_reinvestment")
            self.peak = max(self.peak, before_equity)
            if before_equity > 0 and not self.breached(before, before_equity):
                self.reinvest(day, before_equity)
            else:
                self.release_reinvestment(day, "stopped_before_trade")
        values, outstanding, equity = self.balances(day, "close")
        components = {key: math.fsum(amounts) for key, amounts in self.pending_pnl.items()}
        opening_equity = self.opening_equity
        pnl = equity - opening_equity
        self.cumulative_pnls.append(pnl)
        simple = pnl / opening_equity
        if not math.isfinite(simple) or (equity > 0 and simple <= -1):
            raise ValueError(f"daily return outside representable positive wealth on {day}")
        self.simple_returns.append(simple)
        self.wealth *= 1 + simple
        if not math.isfinite(self.wealth) or (equity > 0 and self.wealth <= 0):
            raise ValueError(f"cumulative wealth is not representable on {day}")
        self.peak = max(self.peak, equity)
        gross_value = math.fsum(abs(v) for v in values.values())
        equity_ratio = equity / gross_value if gross_value else None
        margin_breached = self.breached(values, equity)
        if equity <= 0 or margin_breached:
            self.status = "stopped"
            self.stop_reason = "nonpositive_equity" if equity <= 0 else "maintenance_margin_breach"
            self.stop_session, self.stop_time = day, self.closes[day]
        for code, actual, expected in (
            ("pnl_attribution", math.fsum(components.values()), pnl),
            ("cumulative_pnl", math.fsum(self.cumulative_pnls), equity - capital),
            ("compounded_equity", capital * self.wealth, equity),
        ):
            residual, tolerance = _check(actual, expected, max(capital, equity), code)
            records["diagnostics"].append({"session": day, "code": code,
                                           "residual": residual, "tolerance": tolerance})
        records["daily"].append({
            "session": day, "period_start": self.previous_session, "opening_equity": opening_equity,
            "equity": equity, "pnl": pnl, "simple_return": simple,
            "log_return": math.log(equity) - math.log(opening_equity) if equity > 0 else None,
            "cumulative_pnl": math.fsum(self.cumulative_pnls),
            "cumulative_simple_return": math.fsum(self.simple_returns), "compounded_return": self.wealth - 1,
            "cash": self.cash, "debt": self.debt, "dividend_receivable": outstanding,
            "restricted_collateral": self.collateral, "dividend_liability": math.fsum(v["outstanding"] for v in self.liabilities.values()),
            "long_exposure": math.fsum(max(v, 0.) for v in values.values()),
            "short_exposure": math.fsum(max(-v, 0.) for v in values.values()),
            "short_liability": math.fsum(max(-v, 0.) for v in values.values()),
            "margin_required": self.margin_required(values) if self.signed else (self.threshold or 0.)*(
                gross_value if self.ca is None else math.fsum([gross_value, self.claim_totals()[2]])),
            "gross_exposure": gross_value, "net_exposure": math.fsum(values.values()),
            "gross_leverage": gross_value / equity if equity > 0 else None,
            "equity_ratio": equity_ratio, "margin_breached": margin_breached,
            "drawdown": equity / self.peak - 1, "includes_entry_costs": self.previous_session == self.entry_session,
            **dict(zip(CLAIM_COLUMNS, self.claim_totals() if self.ca is not None else (0., 0., 0., 0.))),
        })
        for (component, asset), amount in sorted(components.items(), key=lambda x: (x[0][0], x[0][1] or "")):
            records["attribution"].append({"session": day, "component": component,
                                           "asset": asset, "pnl": amount})
        self.position_rows(day, values, equity)
        for action_id, item in self.receivables.items():
            records["receivables"].append({"session": day, "action_id": action_id,
                                           **{k: item[k] for k in SCHEMAS["receivables"]
                                              if k not in {"session", "action_id"}}})
        for action_id, item in self.liabilities.items():
            records["dividend_liabilities"].append({"session": day, "action_id": action_id,
                **{k: item[k] for k in SCHEMAS["dividend_liabilities"] if k not in {"session", "action_id"}}})
        if self.ca is not None:
            self.claim_rows(day)
        self.pending_pnl.clear()
        self.previous_values, self.opening_equity, self.previous_session = values, equity, day
        if self.status == "stopped":
            self.event(day, self.stop_reason, phase="close")
            return True
        if self.ca is not None:
            self._checkpoint()
        return False

    # --------------------------------------------------------------- results

    def _metadata(self):
        market, policy, financing = self.market, self.policy, self.financing
        stop_session, stop_time = self.stop_session, self.stop_time
        metadata = {
            "snapshot_id": market.snapshot_id, "source": deepcopy(market.metadata),
            "currency": market.metadata["currency"], "frequency": "1d", "return_basis": "net_equity",
            "initial_capital": self.capital, "entry_session": self.entry_session.isoformat(),
            "end_session": self.end_session.isoformat(), "weights": self.allocation, "resolved_weights": self.resolved_weights,
            "actual_end_session": self.previous_session.isoformat(), "status": self.status,
            "stop_reason": self.stop_reason, "stop_session": stop_session.isoformat() if stop_session else None,
            "stop_time": stop_time.isoformat() if stop_time else None,
            "policy": asdict(policy), "cost_rates": self.rates, "cost_basis": "modeled",
            "cost_model": self.entry_binding.metadata if self.impact_costs else {"model": "proportional"},
            "cash_rate": self.rate, "cash_day_count": self.cash_day_count, "cash_capitalization": "daily_calendar_date",
            "financing": self.financing_metadata,
            "borrowing_rate": self.borrowing_rate, "maintenance_equity_ratio": self.threshold,
            "margin_monitoring": "session_close", "margin_comparison_relative_tolerance": 1e-12,
            "return_denominator": "first_interval_pre_entry_capital_then_prior_closing_equity",
            "dividend_policy": "ex_date_receivable_pay_date_cash_no_reinvestment",
            "cash_sweep": financing.cash_sweep if financing else "hold_cash_no_debt",
            "settlement": "immediate", "taxes": "excluded",
            "package_version": PACKAGE_VERSION, "python_version": platform.python_version(),
            "polars_version": pl.__version__,
        }
        if self.schedule is not None:
            metadata.update(strategy="scheduled_rebalance", rebalance_policy=asdict(self.rebalance_policy),
                dividend_policy="ex_date_receivable_pay_date_cash_reinvest_only_via_scheduled_trades",
                decision_timing=("decision_session_strictly_before_execution"
                    if self.rebalance_policy.decision_timing == "prior_session"
                    else "same_session_close_execution_assumed"),
                concentration_denominator="target_risky_asset_proportion",
                target_weight_normalization="divide_by_basket_sum_within_1e-12_roundoff",
                margin_monitoring="before_scheduled_trade_and_after_session_close",
                turnover_denominator="pre_trade_equity_entry_separately_labeled",
                no_trade_relative_tolerance=1e-13)
        if self.dividend_reinvestment is not None:
            metadata.update(dividend_reinvestment=asdict(self.dividend_reinvestment),
                dividend_policy="ex_date_receivable_pay_date_cash_explicit_reinvestment",
                reinvestment_payment_availability="before_close_on_pay_date_assumed",
                reinvestment_asset_policy="paying_asset_even_if_previously_sold",
                reinvestment_budget="paid_dividend_principal_net_of_assigned_debt_repayment",
                reinvestment_cost_policy="zero_commission_spread_impact",
                margin_monitoring="before_scheduled_or_reinvestment_trade_and_after_session_close",
                turnover_denominator="pre_trade_equity_entry_separately_labeled")
        if self.signed:
            metadata.update(concentration_denominator="absolute_target_value_over_gross_exposure",
                target_weight_normalization="none_signed_inputs",
                long_short=asdict(self.long_short), stock_borrow=self.borrow_metadata,
                sizing=self.signed_kind, signed_targets=self.allocation, weights=None, resolved_weights=None,
                collateral_policy="marked_each_session_close_restricted_from_sweep",
                margin_convention="equity_ge_long_margin_times_long_plus_short_margin_times_absolute_short",
                funding="explicit_cash_loan_including_collateral_costs_and_unpaid_receivables",
                rebate_basis="opening_restricted_collateral_gross_before_separate_borrow_fees",
                reinvestment_if_payer_short="release_cash_no_cover",
                short_dividend_policy="ex_date_liability_pay_date_settlement_no_reinvestment")
        if self.impact_costs:
            metadata["cost_model_selection"] = "exact_decision_date" if isinstance(self.costs, Mapping) else "static"
            metadata["cost_models"] = [deepcopy(binding.metadata) for binding in self.bindings.values()]
            for day, binding in self.bindings.items():
                meta = binding.metadata
                liq = meta["liquidity"]
                self.records["cost_model_selections"].append(dict(session=day,
                    decision_session=date.fromisoformat(meta["decision_session"]) if meta["decision_session"] else None,
                    model_id=meta["model_id"], snapshot_id=meta["snapshot_id"],
                    sample_start=date.fromisoformat(liq["sample_start"]) if liq.get("sample_start") else None,
                    sample_end=date.fromisoformat(liq["sample_end"]),
                    status="executed" if day in self.executed_baskets else "not_executed"))
        if self.ca is not None:
            policy = self.ca.policy
            metadata.update(lifecycle={
                "strategy_universe": sorted(self.universe), "reachable_assets": self.assets,
                "corporate_action_policy": asdict(policy) if policy is not None else None,
                "applied_actions": sorted(self.applied),
                "valuation": "market_close_else_supplied_mark_else_unvalued_stop",
                "pending_cash_claim_valuation": "face_value_undiscounted_no_credit_risk_no_interest",
                "pending_security_claim_valuation": "quantity_times_market_close_or_supplied_mark",
                "conversion_recognition": "previous_close_marks_else_first_available_mark",
                "entitlement": "holdings_at_start_of_effective_or_ex_date_before_that_session",
                "event_order": ["financing_accrual", "stock_borrow_fee_unless_terminated", "splits",
                    "dividend_ex_dates", "warrant_expiry", "acquisitions_and_distributions",
                    "dividend_payments", "claim_settlements_and_deliveries", "reinvestment_release_and_sweep",
                    "close: valuation_check", "close: price_and_claim_marks",
                    "close: obligation_cash_settlement", "close: collateral_and_sweep",
                    "close: warrant_exercise_and_liquidation", "close: scheduled_basket_or_reinvestment",
                    "close: margin_check_and_reporting"],
                "mandatory_conversion_costs": "none_except_reorganization_fee",
                "short_obligation_collateral": "collateral_multiple_times_obligation_until_settlement",
                "unvalued": self.unvalued})
            metadata["settlement"] = "immediate_trades_dated_corporate_action_settlement"
        if self.signed_kind == "target_notional":
            metadata["target_notional_units"] = market.metadata["currency"]
            metadata["target_notional_sizing"] = "predetermined_currency_amount_divided_by_execution_raw_close_no_scaling"
        return metadata

    def result(self):
        """Assemble typed result tables and run metadata (call once, after ``run``)."""
        metadata = self._metadata()
        return BacktestResult(**{name: (self.target_table.clone() if name == "targets" and self.signed and self.target_table is not None
                                 else pl.DataFrame(rows, schema=SCHEMAS[name]))
                                 for name, rows in self.records.items()}, metadata=metadata,
                              status=self.status, stop_reason=self.stop_reason,
                              stop_session=self.stop_session, stop_time=self.stop_time)
