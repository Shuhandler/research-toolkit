from dataclasses import replace
from datetime import date, timedelta
import math

import polars as pl
import pytest
import research_toolkit as rt


def panel(inputs, data=None):
    # A: +10%, -10%, +10%; B twice those returns. Two usable observations
    # strictly before Jan 5; that day's return must not enter its decision.
    data = data or {"A": [100., 110., 99., 108.9, 120.], "B": [100., 120., 96., 115.2, 130.]}
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(len(next(iter(data.values()))))]
    return rt.returns(rt.prepare_market_data(**inputs(series=data, dates=dates)), method="simple", basis="price")


OPTIONS = dict(decision_session=date(2024, 1, 5), lookback=2, periods_per_year=4)


def test_inverse_volatility_independent_oracle(inputs):
    result = rt.inverse_volatility_weights(panel(inputs), max_asset_weight=.7, **OPTIONS)
    assert result.weights["weight"].to_list() == pytest.approx([2/3, 1/3])
    assert result.estimates["annualized_volatility"].to_list() == pytest.approx([math.sqrt(.08), math.sqrt(.32)])
    assert result.metadata["sample_end"] == "2024-01-04"
    assert result.metadata["sample_period_start"] == "2024-01-02"


def test_decision_and_future_returns_do_not_change_weights_or_risk(inputs):
    before = panel(inputs)
    after = replace(before, values=before.values.with_columns(
        pl.when(pl.col("session") >= date(2024, 1, 5)).then(pl.lit(.8))
        .otherwise(pl.col("simple_return")).alias("simple_return")))
    one = rt.inverse_volatility_weights(before, max_asset_weight=1., **OPTIONS)
    two = rt.inverse_volatility_weights(after, max_asset_weight=1., **OPTIONS)
    assert one.weights.equals(two.weights)
    assert one.estimates.equals(two.estimates)
    kwargs = dict(weights={"A": .5, "B": .5}, gross_leverage=1., **OPTIONS)
    assert rt.risk_contributions(before, **kwargs).values.equals(rt.risk_contributions(after, **kwargs).values)


def test_limits_are_rejections_not_clipping(inputs):
    with pytest.raises(ValueError, match="no clipping"):
        rt.inverse_volatility_weights(panel(inputs), max_asset_weight=.6, **OPTIONS)


@pytest.mark.parametrize("update", [{"lookback": 3}, {"lookback": True}, {"lookback": 1},
    {"periods_per_year": 0}, {"decision_session": date(2024, 1, 1)}])
def test_invalid_or_insufficient_window(inputs, update):
    with pytest.raises(ValueError):
        rt.inverse_volatility_weights(panel(inputs), max_asset_weight=1., **(OPTIONS | update))


def test_flat_asset_rejected_for_inverse_vol_but_valid_covariance(inputs):
    data = panel(inputs, {"A": [100.]*5, "B": [100., 120., 96., 115.2, 130.]})
    with pytest.raises(ValueError, match="positive asset volatility"):
        rt.inverse_volatility_weights(data, max_asset_weight=1., **OPTIONS)
    risk = rt.risk_contributions(data, weights={"A": .5, "B": .5}, gross_leverage=1., **OPTIONS)
    assert risk.values["volatility_contribution"].to_list() == pytest.approx([0, math.sqrt(.08)])


def test_covariance_euler_contributions(inputs):
    risk = rt.risk_contributions(panel(inputs), weights={"A": .5, "B": .5}, gross_leverage=2., **OPTIONS)
    # Perfect positive correlation: 1*A + 1*B = 3*A; contributions are sigma_A, 2*sigma_A.
    assert risk.values["volatility_contribution"].to_list() == pytest.approx([math.sqrt(.08), 2*math.sqrt(.08)])
    assert risk.values["fraction_of_total"].to_list() == pytest.approx([1/3, 2/3])
    assert risk.metadata["portfolio_volatility"] == pytest.approx(3*math.sqrt(.08))
    zero = rt.risk_contributions(panel(inputs), weights={"A": .5, "B": .5}, gross_leverage=0., **OPTIONS)
    assert zero.values["volatility_contribution"].null_count() == 2
    assert zero.values["status"].to_list() == ["zero_portfolio_volatility"]*2


def test_negative_risk_contribution_is_retained(inputs):
    data = panel(inputs, {"A": [100., 110., 99., 108.9, 120.], "B": [100., 80., 96., 76.8, 100.]})
    risk = rt.risk_contributions(data, weights={"A": .8, "B": .2}, gross_leverage=1., **OPTIONS)
    assert risk.values["volatility_contribution"][1] < 0
    assert risk.values["volatility_contribution"].sum() == pytest.approx(risk.metadata["portfolio_volatility"])


def test_raw_total_return_split_and_dividend(inputs):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 49., 49.]},
        splits=[("s", "A", date(2024, 1, 3), 2.)],
        dividends=[("d", "A", date(2024, 1, 3), date(2024, 1, 4), 1.)]))
    result = rt.returns(market, method="simple", basis="total_return", dividend_policy="reinvest_ex_close")
    assert result.values["simple_return"].to_list() == [None, 0., 0.]
    assert "raw_price_split_discontinuity" not in result.diagnostics["code"]
    assert result.metadata["dividend_policy"] == "reinvest_ex_close"
    with pytest.raises(ValueError):
        rt.returns(market, method="simple", basis="price", dividend_policy="reinvest_ex_close")


def test_analytical_reinvestment_differs_from_cash_held_ledger(inputs, run):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 90., 99.]},
        dividends=[("d", "A", date(2024, 1, 3), date(2024, 1, 4), 10.)]))
    analytical = rt.returns(market, method="simple", basis="total_return", dividend_policy="reinvest_ex_close")
    assert analytical.values["simple_return"][2] == pytest.approx(.1)
    assert run(market).daily["simple_return"][1] == pytest.approx(.09)


def test_rolling_risk_hand_oracle(market, run):
    result = rt.rolling_risk(run(market), window=2, periods_per_year=4, risk_free_annual_effective=0.)
    assert result.values["n_obs"].to_list() == [1, 2]
    assert result.values["annualized_volatility"][0] is None
    assert result.values["annualized_volatility"][1] == pytest.approx(math.sqrt(.08))
    assert result.values["sharpe"][1] == pytest.approx(0)
    assert result.values["period_start"][1] == date(2024, 1, 2)
    assert result.metadata["information_cutoff"] == "through_current_close_reporting_only"


def test_rolling_flat_and_insufficient(inputs, run):
    backtest = run(rt.prepare_market_data(**inputs(series={"A": [100.]*3})))
    result = rt.rolling_risk(backtest, window=2, periods_per_year=252, risk_free_annual_effective=0.)
    assert result.values["annualized_volatility"][1] == 0
    assert result.values["sharpe_status"][1] == "zero_volatility"
    assert rt.rolling_risk(backtest, window=10, periods_per_year=252, risk_free_annual_effective=0.).values["annualized_volatility"].null_count() == 2
