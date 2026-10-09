"""Corporate-action, claim and lifecycle postings for the shared daily ledger.

``_ActionLedger`` is mixed into ``_backtest._Ledger``; it never runs on its own.
Every balance change still goes through ``_Ledger.event``. Pending consideration
and undelivered securities are *claims*: signed cash amounts (carried at face
value) or signed security quantities (marked at market quotes or explicitly
supplied valuation marks). Positive claims are assets, negative claims are
obligations of short holders. Markets without lifecycle inputs never reach here.
"""

from collections import defaultdict
from copy import deepcopy
import math

import polars as pl

from ._data import _table
from ._lifecycle import CorporateActionPolicy, EXERCISE_SCHEMA, _cash_amount, _lifecycle

F, S, D = pl.Float64, pl.String, pl.Date
ACTION_SCHEMAS = {
    "corporate_actions": {"date": D, "action_id": S, "action_type": S, "stage": S, "source_asset": S,
        "asset": S, "entitled_quantity": F, "quantity_delta": F, "cash_amount": F, "claim_id": S,
        "value": F, "valuation": S, "pnl": F, "fee": F, "trade_id": S, "provenance": S},
    "security_claims": {"session": D, "claim_id": S, "action_id": S, "kind": S, "source_asset": S,
        "asset": S, "quantity": F, "amount": F, "mark": F, "value": F, "due_date": D, "settlement": S},
}
CLAIM_COLUMNS = ("pending_cash_receivable", "pending_cash_payable",
                 "pending_security_receivable", "pending_security_obligation")


class _Schedule:
    """Compiled lifecycle/action inputs restricted to the assets a run can hold."""

    def __init__(self, market, universe, policy, exercises, entry, end, sessions, impact_costs):
        self.marks = {(r["session"], r["asset"]): (r["mark"], r["method"], r["source"])
                      for r in market.valuation_marks.iter_rows(named=True)}
        self.security = {r["asset"]: r for r in market.securities.iter_rows(named=True)}
        self.terminated = {a: r["terminated_date"] for a, r in self.security.items() if r["terminated_date"]}
        self.warrants = {r["asset"]: r for r in market.warrants.iter_rows(named=True)}
        legs = defaultdict(list)
        for r in market.action_legs.iter_rows(named=True):
            legs[r["action_id"]].append(r)
        actions = [dict(r, legs=sorted(legs[r["action_id"]], key=lambda x: x["leg_id"]))
                   for r in market.corporate_actions.iter_rows(named=True)]
        reachable = set(universe)
        while True:
            grown = set(reachable)
            grown |= {w["underlying_asset"] for a, w in self.warrants.items() if a in grown}
            for action in actions:
                if action["source_asset"] in grown:
                    grown |= {leg["asset"] for leg in action["legs"] if leg["leg_type"] == "security"}
            if grown == reachable:
                break
            reachable = grown
        self.reachable = reachable
        self.by_date = defaultdict(list)
        for action in sorted(actions, key=lambda x: (x["effective_date"], x["action_id"])):
            if action["source_asset"] in reachable and entry < action["effective_date"] <= end:
                self.by_date[action["effective_date"]].append(action)
        self.expiries = defaultdict(list)
        for a in sorted(reachable):
            if a in self.warrants and entry < self.terminated[a] <= end:
                self.expiries[self.terminated[a]].append(a)
        self.exercises = self._exercises(exercises, reachable, entry, end, sessions)
        involved = bool(self.by_date or self.expiries or self.exercises
                        or any(a in self.warrants for a in reachable))
        if involved and not isinstance(policy, CorporateActionPolicy):
            raise ValueError("corporate actions or warrants affect this portfolio; supply "
                             "corporate_actions=CorporateActionPolicy(...)")
        if policy is not None:
            if not isinstance(policy, CorporateActionPolicy):
                raise ValueError("corporate_actions must be a CorporateActionPolicy")
            policy.__post_init__()
        self.policy = policy
        if policy is not None and policy.distributed_securities != "retain" and impact_costs:
            raise ValueError("liquidating distributed securities needs proportional TradeCosts; "
                             "square-root models have no liquidity snapshot for those sessions")

    def _exercises(self, table, reachable, entry, end, sessions):
        if table is None:
            return defaultdict(list)
        frame = _table(table, EXERCISE_SCHEMA, "warrant_exercises", ["exercise_id"], nonempty=True)
        out = defaultdict(list)
        for r in frame.sort("session", "exercise_id").iter_rows(named=True):
            w = self.warrants.get(r["warrant"])
            where = f"warrant exercise {r['exercise_id']}"
            if w is None or r["warrant"] not in reachable:
                raise ValueError(f"{where}: unknown warrant or warrant outside this portfolio")
            if r["session"] not in sessions or r["decision_session"] not in sessions or not entry < r["session"] < end:
                raise ValueError(f"{where}: sessions must be supplied and execute in [entry, end); the terminal close is mark-only")
            if r["decision_session"] >= r["session"]:
                raise ValueError(f"{where}: decision_session must precede the exercise session")
            allowed = (w["exercisable_from"] <= r["session"] <= w["expiry_date"] if w["exercise_style"] == "american"
                       else r["session"] == w["expiry_date"])
            if not allowed:
                raise ValueError(f"{where}: exercise outside the warrant's {w['exercise_style']} exercise window")
            if r["warrants"] <= 0 or r["fee"] < 0 or r["delivery_date"] < r["session"]:
                raise ValueError(f"{where}: warrants must be positive, fee nonnegative and delivery on/after exercise")
            out[r["session"]].append(r)
        return out


def _compile(market, universe, policy, exercises, entry, end, sessions, impact_costs):
    if not _lifecycle(market):
        if policy is not None or exercises is not None:
            raise ValueError("corporate_actions/warrant_exercises require lifecycle market inputs")
        return None
    return _Schedule(market, universe, policy, exercises, entry, end, sessions, impact_costs)


class _ActionLedger:
    """Claims, conversions, deliveries, warrants and lifecycle-aware valuation."""

    # ------------------------------------------------------------- valuation

    def quoted(self, day, asset):
        return (day, asset) in self.prices

    def mark(self, day, asset):
        """Market close, else an explicitly supplied valuation mark, else None."""
        if (day, asset) in self.prices:
            return self.prices[day, asset]
        if self.ca is not None and (day, asset) in self.ca.marks:
            return self.ca.marks[day, asset][0]
        return None

    def value_of(self, day, asset):
        quantity = self.quantity[asset]
        if self.ca is None:
            return quantity*self.prices[day, asset]
        if quantity == 0:
            return 0.
        return quantity*self.mark(day, asset)

    def claim_totals(self):
        """Positive cash, payable cash, positive security value, obligation value (carrying)."""
        pos_cash = math.fsum(c["amount"] for c in self.claims.values() if c["kind"] == "cash" and c["amount"] > 0)
        neg_cash = math.fsum(-c["amount"] for c in self.claims.values() if c["kind"] == "cash" and c["amount"] < 0)
        pos_sec = math.fsum(c["carrying"] for c in self.claims.values() if c["kind"] == "security" and c["quantity"] > 0)
        neg_sec = math.fsum(-c["carrying"] for c in self.claims.values() if c["kind"] == "security" and c["quantity"] < 0)
        return pos_cash, neg_cash, pos_sec, neg_sec

    def claims_net(self):
        pos_cash, neg_cash, pos_sec, neg_sec = self.claim_totals()
        return math.fsum([pos_cash, -neg_cash, pos_sec, -neg_sec])

    def obligations(self):
        _, neg_cash, _, neg_sec = self.claim_totals()
        return math.fsum([neg_cash, neg_sec])

    def _unvalued(self, day):
        missing = sorted({a for a in self.assets if self.quantity[a] and self.mark(day, a) is None} |
                         {c["asset"] for c in self.claims.values() if c["kind"] == "security"
                          and self.mark(day, c["asset"]) is None})
        return missing

    def _basket_scope(self, day, targets, values):
        """Quoted strategy assets that trade, and the value held outside the basket.

        Universe assets without a quote must have zero target and zero holdings;
        action-created holdings and pending claims are held outside the basket.
        """
        scope = []
        for asset in sorted(targets):
            if (day, asset) in self.prices:
                scope.append(asset)
            elif targets[asset] or self.quantity[asset]:
                raise ValueError(f"target on {day} requires trading {asset}, which is not quoted "
                                 "(stale, suspended or terminated security)")
        outside = math.fsum([*(values[a] for a in values if a not in scope), self.claims_net()])
        return scope, outside

    # ---------------------------------------------------------- checkpoints

    def _checkpoint(self):
        self._saved = {
            "scalars": (self.cash, self.debt, self.collateral, self.peak, self.wealth,
                        self.opening_equity, self.previous_session),
            "quantity": dict(self.quantity), "previous_values": dict(self.previous_values),
            "receivables": deepcopy(self.receivables), "liabilities": deepcopy(self.liabilities),
            "claims": deepcopy(self.claims), "deferred": deepcopy(self.deferred),
            "action_held": set(self.action_held), "applied": set(self.applied),
            "budgets": dict(self.reinvestment_budgets),
            "reinvestments": deepcopy(self.records["dividend_reinvestments"]),
            "executed": set(self.executed_baskets),
            "lengths": {k: len(v) for k, v in self.records.items()},
            "deltas": {k: len(getattr(self, k)) for k in self._delta_lists},
            "qdeltas": {a: len(v) for a, v in self.quantity_deltas.items()},
            "cqdeltas": {a: len(v) for a, v in self.claim_quantity_deltas.items()},
            "series": (len(self.cumulative_pnls), len(self.simple_returns)),
            "claim_counter": self.claim_counter,
        }

    def _restore(self):
        s = self._saved
        (self.cash, self.debt, self.collateral, self.peak, self.wealth,
         self.opening_equity, self.previous_session) = s["scalars"]
        self.quantity, self.previous_values = dict(s["quantity"]), dict(s["previous_values"])
        self.receivables, self.liabilities = deepcopy(s["receivables"]), deepcopy(s["liabilities"])
        self.claims, self.deferred = deepcopy(s["claims"]), deepcopy(s["deferred"])
        self.action_held, self.applied = set(s["action_held"]), set(s["applied"])
        self.reinvestment_budgets = dict(s["budgets"])
        self.executed_baskets = set(s["executed"])
        self.claim_counter = s["claim_counter"]
        for name, length in s["lengths"].items():
            del self.records[name][length:]
        self.records["dividend_reinvestments"][:] = deepcopy(s["reinvestments"])
        self.reinvestment_rows = {r["action_id"]: r for r in self.records["dividend_reinvestments"]}
        for name, length in s["deltas"].items():
            del getattr(self, name)[length:]
        for store, saved in ((self.quantity_deltas, s["qdeltas"]), (self.claim_quantity_deltas, s["cqdeltas"])):
            for a in list(store):
                del store[a][saved.get(a, 0):]
        del self.cumulative_pnls[s["series"][0]:]
        del self.simple_returns[s["series"][1]:]
        self.pending_pnl = defaultdict(list)

    def _stop_unvalued(self, day, missing):
        """Roll back to the last valued close and stop; never invent a NAV."""
        self._restore()
        self.status, self.stop_reason = "stopped", "unvalued_position"
        self.stop_session, self.stop_time = self.previous_session, self.closes[self.previous_session]
        self.unvalued = {"session": day.isoformat(), "assets": missing,
                         "last_valued_session": self.previous_session.isoformat()}
        self.event(self.previous_session, "unvalued_position", phase="close")
        return True

    # -------------------------------------------------------------- records

    def audit(self, day, action, stage, **values):
        row = {k: None for k in ACTION_SCHEMAS["corporate_actions"]}
        row.update(date=day, action_id=action.get("action_id"), action_type=action.get("action_type"),
                   stage=stage, source_asset=action.get("source_asset"),
                   provenance=action.get("classification_source"), **values)
        self.records["corporate_actions"].append(row)

    def new_claim(self, day, action, kind, asset, quantity, amount, due, carrying, settlement, source):
        claim_id = f"K{self.claim_counter:06d}"
        self.claim_counter += 1
        self.claims[claim_id] = {"claim_id": claim_id, "action_id": action["action_id"], "kind": kind,
            "source_asset": source, "asset": asset, "quantity": quantity, "amount": amount,
            "due_date": due, "carrying": carrying, "settlement": settlement}
        if kind == "cash":
            self.event(day, "action_cash_claim", asset=source, action_id=action["action_id"], dcc=amount)
        else:
            self.event(day, "action_security_claim", asset=asset, action_id=action["action_id"], dcq=quantity)
        return claim_id

    def claim_rows(self, day):
        for c in sorted(self.claims.values(), key=lambda c: c["claim_id"]):
            mark = None if c["kind"] == "cash" else self.mark(day, c["asset"])
            self.records["security_claims"].append({"session": day, "claim_id": c["claim_id"],
                "action_id": c["action_id"], "kind": c["kind"], "source_asset": c["source_asset"],
                "asset": c["asset"], "quantity": c["quantity"], "amount": c["amount"], "mark": mark,
                "value": c["amount"] if c["kind"] == "cash" else c["carrying"],
                "due_date": c["due_date"], "settlement": c["settlement"]})

    def book_cost(self, day, component, amount, asset, trade_id=None, phase="before_close"):
        """Record an expense already deducted from cash by the caller."""
        self.records["costs"].append(dict(cost_id=f"C{len(self.records['costs']):06d}", date=day,
            time=self.closes.get(day) if phase != "before_close" else None, trade_id=trade_id, asset=asset,
            component=component, amount=amount, basis="modeled"))
        self.event(day, component, phase=phase, asset=asset, trade_id=trade_id, dc=-amount)
        self.pending_pnl[component, asset].append(-amount)

    def pay(self, day, amount, phase="before_close"):
        """Fund a mandatory cash outflow from free cash, then the explicit loan."""
        shortfall = max(amount-self.available_cash(), 0.)
        if shortfall and not self.signed and self.threshold is None:
            raise ValueError(f"a corporate-action payment on {day} needs borrowing; supply Financing "
                             "with maintenance_equity_ratio or hold sufficient cash")
        self.fund(day, shortfall, phase)
        self.cash -= amount

    # --------------------------------------------------------- date events

    def _apply_lifecycle_actions(self, day):
        """Warrant expiry, then acquisitions and distributions in action_id order."""
        ca = self.ca
        for w in ca.expiries.get(day, []):
            held = self.quantity[w]
            if held:
                carried = self.previous_values[w]
                self.quantity[w] = 0.
                self.previous_values[w] = 0.
                self.event(day, "warrant_expiry", asset=w, dq=-held)
                self.pending_pnl["warrant_expiry", w].append(-carried)
                self.audit(day, {"action_id": None, "source_asset": w}, "warrant_expiry", asset=w,
                           entitled_quantity=held, quantity_delta=-held, value=0., pnl=-carried,
                           valuation="lapsed_unexercised_zero")
            for claim_id, c in sorted(self.claims.items()):
                if c["kind"] == "security" and c["asset"] == w:
                    del self.claims[claim_id]
                    self.event(day, "warrant_expiry", asset=w, action_id=c["action_id"], dcq=-c["quantity"])
                    carried = c["carrying"] or 0.
                    self.pending_pnl["warrant_expiry", w].append(-carried)
                    self.audit(day, {"action_id": c["action_id"], "source_asset": c["source_asset"]},
                               "warrant_expiry", asset=w, claim_id=claim_id, quantity_delta=-c["quantity"],
                               value=0., pnl=-carried, valuation="undelivered_lapsed_zero")
        for action in ca.by_date.get(day, []):
            self._apply_action(day, action)

    def _apply_action(self, day, action):
        aid, src, kind = action["action_id"], action["source_asset"], action["action_type"]
        if aid in self.applied:
            raise ArithmeticError(f"corporate action {aid} applied twice")
        self.applied.add(aid)
        if any(c["kind"] == "security" and c["asset"] == src for c in self.claims.values()):
            raise ValueError(f"action {aid}: an undelivered entitlement in {src} is pending at its "
                             "corporate action; this sequence is unsupported")
        held = self.quantity[src]
        self.audit(day, action, "entitlement", asset=src, entitled_quantity=held,
                   valuation="no_position" if held == 0 else f"holdings_at_start_of_{action['entitlement_basis']}")
        if held == 0:
            return
        recognized = []
        if kind != "distribution":
            carried = self.previous_values[src]
            self.quantity[src] = 0.
            self.previous_values[src] = 0.
            self.event(day, "acquisition_extinguishment", asset=src, action_id=aid, dq=-held)
            self.audit(day, action, "extinguishment", asset=src, entitled_quantity=held,
                       quantity_delta=-held, value=carried, valuation="previous_close_mark")
            recognized.append(-carried)
        settlement = self.ca.policy.short_obligations if held < 0 else "on_due_date"
        for leg in action["legs"]:
            if leg["leg_type"] == "cash":
                per_share = _cash_amount(leg)
                amount = held*per_share
                claim = self.new_claim(day, action, "cash", None, 0., amount, leg["due_date"], amount, "on_due_date", src)
                recognized.append(amount)
                self.audit(day, action, "cash_claim", asset=None, entitled_quantity=held, cash_amount=amount,
                           claim_id=claim, value=amount, valuation="face_value_undiscounted")
                continue
            units = held*leg["units_per_source_share"]
            whole, fraction = units, 0.
            if leg["fraction_policy"] == "cash_in_lieu":
                magnitude = abs(units)
                nearest = round(magnitude)
                snapped = math.isclose(magnitude, nearest, rel_tol=1e-12, abs_tol=1e-9)
                whole = math.copysign(nearest if snapped else math.floor(magnitude), units)
                # A representation-level residual is not a fraction: never a claim opposite to the entitlement.
                fraction = 0. if snapped else units-whole
            asset = leg["asset"]
            if held < 0 and asset in self.ca.warrants and settlement == "borrowed_short_position":
                raise ValueError(f"action {aid}: short warrant obligations cannot become borrowed short positions")
            if whole:
                previous_mark = self.mark(self.previous_session, asset)
                carrying = whole*previous_mark if previous_mark is not None else None
                claim = self.new_claim(day, action, "security", asset, whole, 0., leg["due_date"], carrying, settlement, src)
                if carrying is not None:
                    recognized.append(carrying)
                self.audit(day, action, "security_claim", asset=asset, entitled_quantity=held,
                           quantity_delta=whole, claim_id=claim, value=carrying,
                           valuation=("previous_close_mark" if carrying is not None else "recognized_at_first_mark")
                           + (f";{leg['fraction_policy']}"))
            if fraction:
                amount = fraction*leg["cash_in_lieu_price"]
                claim = self.new_claim(day, action, "cash", None, 0., amount, leg["due_date"], amount, "on_due_date", src)
                recognized.append(amount)
                self.audit(day, action, "cash_in_lieu_claim", asset=asset, entitled_quantity=held,
                           quantity_delta=fraction, cash_amount=amount, claim_id=claim, value=amount,
                           valuation="supplied_cash_in_lieu_price")
        pnl = math.fsum(recognized)
        self.pending_pnl["corporate_action", src].append(pnl)
        self.audit(day, action, "conversion_pnl", asset=src, pnl=pnl,
                   valuation="recognized_claims_minus_derecognized_previous_mark")
        fee = self.ca.policy.reorganization_fee
        if fee:
            self.pay(day, fee)
            self.records["costs"].append(dict(cost_id=f"C{len(self.records['costs']):06d}", date=day, time=None,
                trade_id=None, asset=src, component="reorganization_fee", amount=fee, basis="modeled"))
            self.event(day, "reorganization_fee", asset=src, action_id=aid, dc=-fee)
            self.pending_pnl["reorganization_fee", src].append(-fee)
            self.audit(day, action, "reorganization_fee", asset=src, fee=fee)

    def _settle_claims(self, day):
        """Cash settlements and security deliveries due on this calendar date."""
        for claim_id, c in sorted(self.claims.items()):
            if c["due_date"] != day or c["settlement"] == "cash_at_delivery_close":
                continue
            action = {"action_id": c["action_id"], "source_asset": c["source_asset"]}
            if c["kind"] == "cash":
                amount = c["amount"]
                if amount > 0:
                    self.cash += amount
                else:
                    self.pay(day, -amount)
                self.event(day, "action_cash_settlement", asset=c["source_asset"], action_id=c["action_id"],
                           dc=amount, dcc=-amount)
                self.audit(day, action, "cash_settlement", claim_id=claim_id, cash_amount=amount, value=amount)
            else:
                asset, quantity = c["asset"], c["quantity"]
                self.quantity[asset] += quantity
                if not math.isfinite(self.quantity[asset]):
                    raise ValueError(f"delivered quantity of {asset} is not representable")
                stage = "delivery" if quantity > 0 else "obligation_conversion"
                self.event(day, "security_delivery" if quantity > 0 else "short_obligation_conversion",
                           asset=asset, action_id=c["action_id"], dq=quantity, dcq=-quantity)
                if c["carrying"] is None:
                    self.deferred.append((asset, quantity, c["source_asset"]))
                else:
                    self.previous_values[asset] += c["carrying"]
                if asset not in self.universe:
                    self.action_held.add(asset)
                self.audit(day, action, stage, asset=asset, claim_id=claim_id, quantity_delta=quantity,
                           value=c["carrying"], valuation="carried_claim_value")
            del self.claims[claim_id]

    # ------------------------------------------------------------ at close

    def _recognize_deferred(self, day):
        for asset, quantity, source in self.deferred:
            value = quantity*self.mark(day, asset)
            self.previous_values[asset] += value
            self.pending_pnl["corporate_action", source].append(value)
        self.deferred.clear()

    def _value_claims(self, day):
        for c in self.claims.values():
            if c["kind"] != "security":
                continue
            value = c["quantity"]*self.mark(day, c["asset"])
            if c["carrying"] is None:
                self.pending_pnl["corporate_action", c["source_asset"]].append(value)
                self.audit(day, {"action_id": c["action_id"], "source_asset": c["source_asset"]},
                           "recognition_at_first_mark", asset=c["asset"], claim_id=c["claim_id"],
                           value=value, pnl=value, valuation=self._mark_kind(day, c["asset"]))
            else:
                self.pending_pnl["claim_valuation", c["asset"]].append(value-c["carrying"])
            c["carrying"] = value

    def _mark_kind(self, day, asset):
        if (day, asset) in self.prices:
            return "market_close"
        mark = self.ca.marks.get((day, asset))
        return f"supplied_mark:{mark[1]}" if mark else None

    def _settle_obligations_at_close(self, day):
        for claim_id, c in sorted(self.claims.items()):
            if c["settlement"] != "cash_at_delivery_close" or c["due_date"] > day or not self.quoted(day, c["asset"]):
                continue
            amount = c["carrying"]  # Already marked at this close; settling it creates no P&L.
            if c["kind"] == "security" and c["quantity"] > 0:
                raise ArithmeticError("only short obligations settle in cash at the delivery close")
            self.pay(day, -amount, phase="close")
            self.event(day, "obligation_cash_settlement", phase="close", asset=c["asset"],
                       action_id=c["action_id"], dc=amount, dcq=-c["quantity"])
            self.audit(day, {"action_id": c["action_id"], "source_asset": c["source_asset"]},
                       "obligation_cash_settlement", asset=c["asset"], claim_id=claim_id,
                       quantity_delta=-c["quantity"], cash_amount=amount, value=amount, valuation="market_close")
            del self.claims[claim_id]

    def _action_trades(self, day):
        """Explicit warrant exercises, then the selected distributed-security liquidation."""
        exercises = self.ca.exercises.get(day, [])
        liquidate = (self.ca.policy is not None and self.ca.policy.distributed_securities != "retain"
                     and day != self.end_session)
        candidates = [a for a in sorted(self.action_held) if self.quantity[a] and self.quoted(day, a)] if liquidate else []
        if not exercises and not candidates:
            return
        values = {a: self.value_of(day, a) for a in self.assets}
        equity = self.current_equity(values)
        if equity <= 0 or self.breached(values, equity):
            return  # The session-close check stops the run; no discretionary trade conceals it.
        for row in exercises:
            self._exercise(day, row)
        for asset in candidates:
            self._liquidate(day, asset)
        values = {a: self.value_of(day, a) for a in self.assets}
        if self.signed:
            self.mark_collateral(day, values)
        self.sweep_cash(day)

    def current_equity(self, values):
        return math.fsum([*values.values(), self.cash, self.collateral,
            math.fsum(v["outstanding"] for v in self.receivables.values()),
            -math.fsum(v["outstanding"] for v in self.liabilities.values()), -self.debt, self.claims_net()])

    def _exercise(self, day, row):
        w, n = row["warrant"], row["warrants"]
        terms = self.ca.warrants[w]
        held = self.quantity[w]
        if held < n and not math.isclose(held, n, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"warrant exercise {row['exercise_id']} exceeds the {held} delivered warrants held")
        underlying = terms["underlying_asset"]
        underlying_mark, warrant_mark = self.mark(day, underlying), self.mark(day, w)
        if underlying_mark is None:
            raise ValueError(f"warrant exercise {row['exercise_id']} needs a valuation of {underlying} on {day}")
        strike = n*terms["exercise_price"]
        units = n*terms["units_per_warrant"]
        action = {"action_id": row["exercise_id"], "action_type": "warrant_exercise", "source_asset": w,
                  "classification_source": terms["source"]}
        self.pay(day, strike, phase="close")
        self.quantity[w] = 0. if math.isclose(held, n, rel_tol=1e-12, abs_tol=1e-12) else held-n
        self.event(day, "warrant_exercise", phase="close", asset=w, action_id=row["exercise_id"],
                   dq=self.quantity[w]-held, dc=-strike)
        carrying = units*underlying_mark
        pnl = math.fsum([carrying, -n*warrant_mark, -strike])
        self.pending_pnl["warrant_exercise", w].append(pnl)
        if row["delivery_date"] == day:
            self.quantity[underlying] += units
            self.event(day, "security_delivery", phase="close", asset=underlying, action_id=row["exercise_id"], dq=units)
            claim = None
            if underlying not in self.universe:
                self.action_held.add(underlying)
        else:
            claim = self.new_claim(day, action, "security", underlying, units, 0., row["delivery_date"],
                                   carrying, "on_due_date", w)
        self.audit(day, action, "warrant_exercise", asset=underlying, entitled_quantity=n, quantity_delta=units,
                   cash_amount=-strike, claim_id=claim, value=carrying, pnl=pnl, fee=row["fee"],
                   valuation="exercise_close_marks")
        if row["fee"]:
            self.pay(day, row["fee"], phase="close")
            self.records["costs"].append(dict(cost_id=f"C{len(self.records['costs']):06d}", date=day,
                time=self.closes[day], trade_id=None, asset=w, component="warrant_exercise_fee",
                amount=row["fee"], basis="modeled"))
            self.event(day, "warrant_exercise_fee", phase="close", asset=w, action_id=row["exercise_id"], dc=-row["fee"])
            self.pending_pnl["warrant_exercise_fee", w].append(-row["fee"])

    def _liquidate(self, day, asset):
        """Sell (or cover) an action-created holding at this close with ordinary costs."""
        held = self.quantity[asset]
        mark = self.prices[day, asset]
        equity_before = self.current_equity({a: self.value_of(day, a) for a in self.assets})
        notional = -held*mark
        components = {c: abs(notional)*self.rates[c][asset] for c in self.rates}
        charge = math.fsum(components.values())
        trade_id = f"T{len(self.records['trades']):06d}"
        outflow = notional+charge
        if outflow > 0:
            self.pay(day, outflow, phase="close")
        else:
            self.cash -= outflow
        self.quantity[asset] = 0.
        self.event(day, "trade", phase="close", asset=asset, trade_id=trade_id, dq=-held, dc=-notional)
        self.records["trades"].append(dict(trade_id=trade_id, session=day, time=self.closes[day], asset=asset,
            signed_quantity=-held, reference_price=mark, signed_notional=notional,
            execution="corporate_action_liquidation_close", trade_cost=charge))
        for component, amount in components.items():
            if amount:
                self.book_cost(day, component, amount, asset, trade_id, phase="close")
        self.action_held.discard(asset)
        self.audit(day, {"action_id": None, "source_asset": asset}, "liquidation", asset=asset,
                   quantity_delta=-held, cash_amount=-notional, value=-notional, fee=charge, trade_id=trade_id,
                   valuation="market_close")
        self.records["turnover"].append(dict(session=day, phase="corporate_action_liquidation",
            gross_traded_notional=abs(notional), equity_before=equity_before, turnover=abs(notional)/equity_before))
