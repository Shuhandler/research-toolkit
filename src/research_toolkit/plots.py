"""Optional Matplotlib views of prepared results; no simulations or metric calls.

Every function returns (Figure, Axes), accepts an existing ax, and never calls
show() or changes global styles. Matplotlib is imported only on a plotting call.
"""

from textwrap import fill

import polars as pl

from ._results import PerformanceResult


def _axes(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires the optional extra: pip install -e '.[plot]'") from exc
    return plt.subplots(figsize=(9, 4), layout="constrained") if ax is None else (ax.figure, ax)


def _finish(ax, title, ylabel, *, percent=False, dates=True):
    from matplotlib.ticker import PercentFormatter
    ax.set(title=fill(title, width=88), ylabel=ylabel, xlabel="Session" if dates else "")
    ax.grid(alpha=0.2)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    if dates:
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    return ax.figure, ax


def _context(report):
    if not isinstance(report, PerformanceResult):
        raise ValueError("expected PerformanceResult from performance()")
    m = report.metadata
    label = f"{m['entry_session']} to {m['actual_end_session']}"
    if m["status"] != "complete":
        label += f" | STOPPED: {m['stop_reason']} (partial)"
    return label


def _daily(report, column, label, ax, percent=False):
    context = _context(report)
    _, ax = _axes(ax)
    ax.plot(report.daily["session"].to_list(), report.daily[column].to_list(), linewidth=1)
    return _finish(ax, f"{label} | {context}", "Return" if percent else report.metadata["currency"], percent=percent)


def pnl(report, *, ax=None):
    """Daily net dollar P&L, including entry costs in the first interval."""
    return _daily(report, "pnl", "Daily net P&L", ax)


def returns(report, *, ax=None):
    """Daily net simple returns; no cumulative transformation."""
    return _daily(report, "simple_return", "Daily net simple returns", ax, True)


def equity(report, *, ax=None):
    context = _context(report)
    _, ax = _axes(ax)
    ax.plot(report.equity["session"].to_list(), report.equity["equity"].to_list(), label="Portfolio (net)")
    if report.benchmark_series.height:
        ax.plot(report.benchmark_series["session"].to_list(), report.benchmark_series["equity"].to_list(),
                label=f"Benchmark ({report.metadata['benchmark']['basis']})", alpha=0.8)
    ax.legend(fontsize="small")
    return _finish(ax, f"Equity | {context}", report.metadata["currency"])


def drawdown(report, *, ax=None):
    context = _context(report)
    _, ax = _axes(ax)
    ax.plot(report.drawdowns["session"].to_list(), report.drawdowns["drawdown"].to_list())
    return _finish(ax, f"Net drawdown (entry costs included) | {context}", "Drawdown", percent=True)


def distribution(report, *, bins=30, ax=None):
    context = _context(report)
    _, ax = _axes(ax)
    ax.hist(report.daily["simple_return"].to_list(), bins=bins, edgecolor="white")
    _finish(ax, f"Net simple return distribution | {context}", "Observations", dates=False)
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Daily simple return")
    return ax.figure, ax


def allocation(report, *, ax=None):
    context = _context(report)
    _, ax = _axes(ax)
    for component in report.allocation["component"].unique().sort():
        rows = report.allocation.filter(pl.col("component") == component)
        label = component.removeprefix("asset:") if component.startswith("asset:") else {
            "account:cash": "Cash", "account:debt": "Debt (negative)",
            "account:dividend_receivable": "Dividend receivables",
        }[component]
        ax.plot(rows["session"].to_list(), rows["weight"].to_list(), label=label)
    ax.legend(fontsize="small", ncol=2)
    return _finish(ax, f"Drifting allocation | {context}", "Weight / net equity", percent=True)


def attribution(report, *, ax=None):
    context = _context(report)
    _, ax = _axes(ax)
    labels = ["Price P&L" if value == "price_pnl" else value.replace("_", " ").capitalize()
              for value in report.attribution["component"]]
    ax.barh(labels, report.attribution["pnl"].to_list())
    _finish(ax, f"Cumulative net dollar attribution | {context}", "Component", dates=False)
    ax.set_xlabel(report.metadata["currency"])
    return ax.figure, ax


def prices(market, *, ax=None):
    """Unnormalized supplied prices; split dates marked, price basis always shown."""
    _, ax = _axes(ax)
    for asset in market.prices["asset"].unique().sort():
        rows = market.prices.filter(pl.col("asset") == asset).sort("session")
        ax.plot(rows["session"].to_list(), rows["close"].to_list(), label=asset)
    for row in market.splits.iter_rows(named=True):
        ax.axvline(row["effective_session"], linestyle=":", alpha=0.5, label=f"{row['asset']} split {row['ratio']:g}:1")
    ax.legend(fontsize="small", ncol=2)
    return _finish(ax, f"Prices ({market.metadata['price_basis']})", market.metadata["currency"])


def correlation(result, *, ax=None):
    """Prepared asset correlation matrix; undefined cells stay visibly blank."""
    new_axes = ax is None
    fig, ax = _axes(ax)
    if new_axes:
        fig.set_size_inches(7, 5)
    assets = result.metadata["assets"]
    lookup = {(r['asset'], r['other_asset']): r['correlation'] for r in result.values.iter_rows(named=True)}
    matrix = [[lookup[a, b] if lookup[a, b] is not None else float("nan") for b in assets] for a in assets]
    artist = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(assets)), assets, rotation=30, ha="right")
    ax.set_yticks(range(len(assets)), assets)
    ax.set_title(fill(f"Simple-return correlation ({result.metadata['basis']})", width=65) + "\n"
                 f"{result.metadata['sessions'][0]} to {result.metadata['sessions'][-1]} | "
                 f"n={result.values['n_obs'][0]}", fontsize=10)
    ax.figure.colorbar(artist, ax=ax, label="Pearson correlation")
    return ax.figure, ax
