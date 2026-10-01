import subprocess
import sys

import pytest
import polars as pl

import research_toolkit as rt
from test_metrics import KW


def test_core_import_has_no_optional_stack():
    subprocess.run([sys.executable, "-c", "import research_toolkit, sys; assert not set(('matplotlib', 'numpy', 'pandas', 'sklearn', 'torch', 'requests')) & set(sys.modules)"], check=True)


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


def test_figures_use_prepared_numbers_without_recalculation(market, run, plt, monkeypatch, tmp_path):
    report = rt.performance(run(market), **KW)
    corr = rt.correlation(rt.returns(market, method="simple", basis="price"))
    def blocked(*args, **kwargs): raise AssertionError("plot attempted calculation")
    monkeypatch.setattr(rt, "buy_and_hold", blocked)
    monkeypatch.setattr(rt, "performance", blocked)
    monkeypatch.setattr(rt, "correlation", blocked)
    for name in ("pnl", "returns", "equity", "drawdown", "distribution", "allocation", "attribution"):
        fig, ax = plt.subplots()
        actual_fig, actual_ax = getattr(rt.plots, name)(report, ax=ax)
        assert actual_fig is fig and actual_ax is ax
        if name in ("pnl", "returns"):
            column = "pnl" if name == "pnl" else "simple_return"
            assert ax.lines[0].get_ydata().tolist() == report.daily[column].to_list()
            assert ax.lines[0].get_xdata().tolist() == report.daily["session"].to_list()
        if name == "equity": assert len(ax.lines[0].get_ydata()) == report.daily.height+2
        fig.savefig(tmp_path/f"{name}.png")
    rt.plots.prices(market)[0].savefig(tmp_path/"prices.png")
    rt.plots.correlation(corr)[0].savefig(tmp_path/"correlation.png")


def test_partial_plot_label(inputs, policy, run, plt):
    result = run(rt.prepare_market_data(**inputs(series={"A": [100., 40., 45.]})),
        policy=policy(2), cash_rate=None, cash_day_count=None,
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
            maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt"))
    report = rt.performance(result, allow_partial=True, **KW)
    _, ax = rt.plots.equity(report)
    assert "STOPPED" in ax.get_title() and result.stop_reason in ax.get_title()
    assert result.stop_session.isoformat() in ax.get_title()
    assert min(ax.lines[0].get_ydata()) < 0


def test_rolling_turnover_and_exposure_plots(market, run, plt):
    backtest = run(market)
    rolling = rt.rolling_risk(backtest, window=2, periods_per_year=4, risk_free_annual_effective=0.)
    fig, ax = rt.plots.rolling_risk(rolling)
    assert ax.lines[0].get_ydata().tolist() == rolling.values["annualized_volatility"].to_list()
    fig, ax = rt.plots.turnover(backtest)
    assert len(ax.collections[0].get_offsets()) == 1
    report = rt.performance(backtest, **KW)
    fig, ax = rt.plots.exposures(report)
    assert ax.lines[0].get_ydata().tolist() == report.daily["gross_exposure"].to_list()
