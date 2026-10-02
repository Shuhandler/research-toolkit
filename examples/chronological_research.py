"""Offline milestone 3: lagged features, purged splits, validation and later trades.

All prices/actions come from the repository's synthetic snapshot. Two tiny models
are fitted here (not in the toolkit); no optional ML package or market download.
Run python examples/chronological_research.py to explicitly export artifacts.
"""
from pathlib import Path
from statistics import mean
import math

import polars as pl
import research_toolkit as rt

ROOT = Path(__file__).resolve().parents[1]


def run_example(*, train_sessions=80, validation_sessions=60, test_sessions=80,
                lag=1, gap_sessions=1, initial_capital=100_000., annualization=252,
                rebalance="on_change"):
    market = rt.load_snapshot(ROOT/"examples/snapshots/v1/equities")
    returns = rt.returns(market, method="simple", basis="total_return", dividend_policy="reinvest_ex_close")
    # Only the known structural first return is removed, with its calendar row.
    calendar = market.sessions.slice(1)
    dates = calendar["session"].to_list()
    closes = dict(market.sessions.iter_rows())
    train_end = dates[train_sessions-1]
    validation_end = dates[train_sessions+validation_sessions-1]
    test_end = dates[train_sessions+validation_sessions+test_sessions-1]
    observations = returns.values.filter(pl.col("period_start").is_not_null()).join(calendar, on="session", how="left").select(
        "session", "asset", pl.col("close_at").alias("available_at"), pl.col("simple_return").alias("return"))
    feature_metadata = market.metadata | {"source": "synthetic analytical total returns",
        "feature_units": {"return": "fraction"}, "availability_basis": "synthetic source closes"}
    features = rt.lagged_features(observations, sessions=calendar, columns=["return"], lags=[lag],
        decision_timing="session_close", metadata=feature_metadata)
    assets = sorted(market.prices["asset"].unique())
    by_key = {(r["session"], r["asset"]): r["simple_return"] for r in returns.values.iter_rows(named=True)}
    # Label is the analytical return AFTER next-close execution. It is not the
    # return into that fill, nor the net realized portfolio return including costs.
    labels = pl.DataFrame([(d, a, dates[i+2], closes[dates[i+2]], by_key[dates[i+2], a])
        for i, d in enumerate(dates) if d <= test_end for a in assets],
        schema={"session": pl.Date, "asset": pl.String, "label_end": pl.Date,
                "available_at": pl.Datetime("us", "UTC"), "target": pl.Float64}, orient="row")
    split = rt.chronological_split(features, labels, train_end=train_end, validation_end=validation_end,
        test_end=test_end, gap_sessions=gap_sessions, label_metadata=dict(source="synthetic analytical total returns",
            unit="fraction", definition="one close interval after next-session-close fill; ex-date entitlement index"))
    split = rt.standardize(split)
    xcol = split.metadata["feature_columns"][0]
    x, y = split.train[xcol].to_list(), split.train["target"].to_list()
    # Example-specific training only: intercept baseline and one-feature least squares.
    mx, my = mean(x), mean(y)
    slope = math.fsum((a-mx)*(b-my) for a, b in zip(x, y))/math.fsum((a-mx)**2 for a in x)
    parameters = {"training_mean": {"intercept": my, "slope": 0.},
                  "lag_linear": {"intercept": my-slope*mx, "slope": slope}}

    def predict(partition, configs):
        return pl.concat([partition.select("session", "asset", pl.lit(name).alias("candidate"),
            (pl.lit(p["intercept"])+pl.col(xcol)*p["slope"]).alias("prediction"),
            pl.col("decision_at").alias("available_at")) for name, p in configs.items()])

    study = rt.ResearchStudy(split)
    selection = study.select(predict(split.validation, parameters), parameters=parameters, metric="mean_squared_error")
    forecasts = predict(split.test, {selection.selected_candidate: selection.metadata["selected_parameters"]})
    # Fixed, declared allocation rule: positive forecasts receive equal capital
    # slots; inactive slots stay cash. Model parameters are not portfolio weights.
    joined = forecasts.join(split.test.select("session", "asset", pl.col("available_at").alias("observed_at")),
                            on=["session", "asset"], how="left")
    signal_rows = []
    for day in joined["session"].unique().sort():
        basket = joined.filter(pl.col("session") == day).sort("asset")
        active = (basket["prediction"] > 0).sum()
        for row in basket.iter_rows(named=True):
            weight = float(row["prediction"] > 0)/active if active else 1/len(assets)
            signal_rows.append((day, row["asset"], weight, active/len(assets), row["observed_at"], row["available_at"]))
    signals = pl.DataFrame(signal_rows, schema={"decision_session": pl.Date, "asset": pl.String, "weight": pl.Float64,
        "gross_leverage": pl.Float64, "observed_at": pl.Datetime("us", "UTC"), "available_at": pl.Datetime("us", "UTC")}, orient="row")
    targets = rt.signal_targets(signals, sessions=market.sessions, execution="next_session_close", rebalance=rebalance,
        metadata=market.metadata | {"signal_definition": "positive forecasts: equal capital slots; other slots remain cash",
            "research_split_id": split.snapshot_id, "selected_candidate": selection.selected_candidate,
            "selected_parameters": selection.metadata["selected_parameters"], "selection_partition": "validation"})
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=initial_capital,
        entry_session=targets.targets["session"][0], end_session=test_end,
        policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
            terminal_action="mark_only", non_session="raise", receivable_policy="reserve", max_asset_weight=1.),
        costs=rt.TradeCosts(commission_bps=2., half_spread_bps=3., impact_bps=1.),
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
            on_breach="stop", cash_sweep="repay_debt")).require_complete()
    report = rt.performance(result, periods_per_year=annualization, risk_free_annual_effective=0.,
                            minimum_acceptable_return_annual_effective=0.)
    # Only after fixing the selected model and allocation rule, record final loss.
    evaluation = study.evaluate_test(forecasts)
    return dict(features=features, split=split, selection=selection, evaluation=evaluation,
                forecasts=forecasts, targets=targets, result=result, report=report)


def main():
    outputs = run_example()
    destination = ROOT/"artifacts/milestone3"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in {"exclusions": outputs["split"].excluded, "transforms": outputs["split"].transforms,
        "validation": outputs["selection"].scores, "test": outputs["evaluation"].scores,
        "research-audit": outputs["evaluation"].audit, "signal-audit": outputs["result"].signal_audit,
        "daily": outputs["result"].daily, "trades": outputs["result"].trades}.items():
        table.write_csv(destination/f"{name}.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("equity", "allocation", "drawdown"):
        fig, ax = getattr(rt.plots, name)(outputs["report"])
        fig.savefig(destination/f"{name}.png", dpi=130)
        plt.close(fig)
    print(outputs["selection"].scores)
    print(outputs["evaluation"].scores)
    print(outputs["report"].summary)
    print(destination)


if __name__ == "__main__":
    main()
