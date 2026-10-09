"""The documented public surface exists, is exported, and plot views use prepared values only."""
from datetime import date
from importlib.metadata import version
import pathlib
import re

import pytest

import research_toolkit as rt
from test_allocation import panel, OPTIONS
from test_metrics import KW
from test_series_performance import KW as SERIES_KW, pnl_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLOTS = ["prices", "pnl", "returns", "cumulative_returns", "equity", "drawdown", "distribution", "allocation",
         "attribution", "correlation", "rolling_risk", "risk_contributions", "turnover", "exposures"]


def _resolve(dotted):
    obj = rt
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("document", ["README.md", "docs/api.md"])
def test_every_documented_call_exists(document):
    names = sorted(set(re.findall(r"`(rt\.[A-Za-z_][\w.]*)\(", (ROOT/document).read_text())))
    assert names, "documented calls were found"
    for name in names:
        assert callable(_resolve(name)), name
        if name.count(".") == 1:
            assert name.split(".")[1] in rt.__all__, f"{name} is documented but not exported"


def test_namespaces_and_version():
    assert all(hasattr(rt, name) for name in rt.__all__)
    assert rt.plots.__all__ == PLOTS and all(callable(getattr(rt.plots, n)) for n in PLOTS)
    assert rt.adapters.__all__ == ["yahoo_chart", "download_yahoo", "resolve_aliases"]
    # Helpers imported for internal use are not part of the public namespaces.
    assert not hasattr(rt.plots, "fill") and not hasattr(rt.adapters, "prepare_market_data")
    assert rt.__version__ == version("research-toolkit")


def test_run_metadata_records_the_package_version(market, run):
    assert run(market).metadata["package_version"] == rt.__version__


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


def test_risk_contributions_plot_uses_prepared_values(inputs, plt):
    risk = rt.risk_contributions(panel(inputs), weights={"A": .5, "B": .5}, gross_leverage=1.5, **OPTIONS)
    fig, ax = rt.plots.risk_contributions(risk)
    assert [p.get_height() for p in ax.patches] == pytest.approx(risk.values["volatility_contribution"].to_list())
    assert risk.metadata["sample_start"] in ax.get_title() and "undefined" not in ax.get_title()
    flat = rt.risk_contributions(panel(inputs, {"A": [100.]*5, "B": [50.]*5}), weights={"A": .5, "B": .5},
                                 gross_leverage=1., **OPTIONS)
    _, ax = rt.plots.risk_contributions(flat)
    assert "undefined: zero portfolio volatility" in " ".join(ax.get_title().split())  # Titles wrap.
    assert all(p.get_height() != p.get_height() for p in ax.patches)  # NaN bars, never invented zeros


def test_distribution_bins_prepared_returns(market, run, plt):
    report = rt.performance(run(market), **KW)
    _, ax = rt.plots.distribution(report, bins=4)
    assert len(ax.patches) == 4 and sum(p.get_height() for p in ax.patches) == report.daily.height


def test_prices_marks_each_split(inputs, plt):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 50., 55.], "B": [10., 11., 12.]},
                                             splits=[("s1", "A", date(2024, 1, 3), 2.)]))
    _, ax = rt.plots.prices(market)
    assert [line.get_ydata().tolist() for line in ax.lines[:2]] == [[100., 50., 55.], [10., 11., 12.]]
    markers = ax.lines[2:]
    assert len(markers) == 1 and markers[0].get_label() == "A split 2:1"
    assert list(markers[0].get_xdata()) == [date(2024, 1, 3)]*2
    assert "raw" in ax.get_title()


@pytest.mark.parametrize("name", ["allocation", "attribution", "exposures"])
def test_ledger_only_plots_reject_series_reports(name, plt):
    report = rt.series_performance(pnl_table([1., -2., 3.]), **SERIES_KW)
    with pytest.raises(ValueError, match=f"plots.{name} needs a backtest report"):
        getattr(rt.plots, name)(report)


def test_plot_input_guards(market, run, plt):
    backtest = run(market)
    report = rt.performance(backtest, **KW)
    rolling = rt.rolling_risk(backtest, window=2, periods_per_year=4, risk_free_annual_effective=0.)
    cases = [
        (lambda: rt.plots.equity(backtest), "expected PerformanceResult"),
        (lambda: rt.plots.cumulative_returns(report, method="log"), "method=sum"),
        (lambda: rt.plots.cumulative_returns(report, method="sum", benchmark="yes"), "boolean benchmark"),
        (lambda: rt.plots.rolling_risk(report), "expected RollingRiskResult"),
        (lambda: rt.plots.rolling_risk(rolling, metric="sortino"), "metric must be"),
        (lambda: rt.plots.rolling_risk(rolling, metric="beta"), "benchmark is required"),
        (lambda: rt.plots.risk_contributions(report), "expected RiskResult"),
        (lambda: rt.plots.turnover(report), "expected BacktestResult"),
        (lambda: rt.plots.turnover(backtest, allow_partial=1), "boolean allow_partial"),
    ]
    for call, match in cases:
        with pytest.raises(ValueError, match=match):
            call()
