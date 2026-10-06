# research-toolkit

A Python library for quantitative trading research in Jupyter notebooks. It handles
common tasks like preparing market data, building portfolios, running backtests,
measuring performance, and making plots. Calculations return Polars tables
with assumptions and diagnostics so you can check how the numbers were produced.

```python
import research_toolkit as rt
```

The toolkit currently supports daily long-only and long/short portfolios in one currency,
including leverage, dividends, splits, trading costs, and financing. You can run
buy-and-hold, scheduled rebalancing, or strategies driven by dated signals.

## Functions

The [API documentation](docs/api.md) has arguments, table schemas, and
complete examples.

### Data and returns

| Function | What it does |
| --- | --- |
| `rt.prepare_market_data(...)` | Validates prices, trading sessions, splits, dividends, and source information. |
| `rt.returns(...)` | Calculates simple or log returns using an explicit price or total-return basis. |
| `rt.cumulative_returns(...)` | Calculates summed returns, compounded returns, or wealth multiples. |
| `rt.save_snapshot(...)` | Saves validated market inputs locally for repeatable research. |
| `rt.load_snapshot(...)` | Loads a saved snapshot and checks its integrity. |
| `rt.adapters.yahoo_chart(...)` | Converts saved Yahoo chart responses, or a `download_yahoo` result, into market data and an audit of the conversion. No downloads. |
| `rt.adapters.download_yahoo(...)` | Downloads daily Yahoo Close, Adj Close, and volume for an inclusive date range, with retries, timeouts, and retrieval details. The only function that uses the network. Optional exchange calendars detect missing sessions and return each calendar's completed sessions. |
| `rt.trading_calendar(...)` | Builds explicit exchange trading sessions, or a declared union/intersection of exchanges, for calendar validation (optional `calendar` extra). |

The Yahoo adapter requires explicit price-adjustment information, calendars, and
dividend payment dates where applicable. It does not assume that a provider's
“Close” is a raw execution price. A download keeps its decoded responses, so
`yahoo_chart(download, ...)` can rebuild raw prices for a backtest from it.

### Allocation and backtesting

| Function | What it does |
| --- | --- |
| `rt.equal_weights(...)` | Assigns an equal portfolio weight to each asset. |
| `rt.inverse_volatility_weights(...)` | Gives lower-volatility assets more weight, using only data before the decision. Can reject or explicitly redistribute weights above a cap. |
| `rt.buy_and_hold(...)` | Opens a portfolio from long weights, signed equity exposures, or exact signed quantities, then tracks its holdings and accounts. |
| `rt.scheduled_rebalance(...)` | Trades toward dated weights, signed equity exposures, shares, or fixed dollar targets. |
| `rt.signal_targets(...)` | Converts dated allocation signals into targets for the next supplied session's close. |
| `rt.risk_contributions(...)` | Estimates each asset's contribution to portfolio volatility. |
| `result.require_complete()` | Raises an error if a backtest stopped before its requested end. |

Buy-and-hold lets weights drift without trading. Splits and explicitly enabled
dividend reinvestment can change quantities. Neither simulator automatically sells
the portfolio at the end.

### Trading costs and sizing

| Function | What it does |
| --- | --- |
| `rt.estimate_liquidity(...)` | Estimates daily volatility and average daily dollar volume from data available before the decision. |
| `rt.estimate_trade_costs(...)` | Estimates commission, spread, and square-root impact costs for supplied orders. |
| `rt.size_entry_orders(...)` | Sizes entry orders to pay trading costs and reach the requested leverage relative to equity after costs. |

Order size divided by daily dollar volume is an order/ADV ratio, not intraday
participation. Cost estimates are previews; the backtest charges costs once, on
actual trades. Automatic dividend reinvestment has zero trading costs.
For scheduled runs, pass one static cost model or a model for each decision date.
The [fixed-dollar notebook](examples/fixed_dollar_rebalancing.ipynb) shows a
long/short basket with a hedge and updated liquidity estimates.

### Performance and comparisons

| Function | What it does |
| --- | --- |
| `rt.performance(...)` | Reports P&L, returns, drawdowns, risk, and benchmark comparisons, including tracking error and information ratio. |
| `rt.series_performance(...)` | Builds the same report from a daily net dollar P&L series and a starting capital, without a backtest. |
| `rt.tail_risk(...)` | Calculates historical VaR and expected tail loss for returns and dollar P&L from a report. |
| `rt.correlation(...)` | Calculates correlations between asset returns. |
| `rt.rolling_risk(...)` | Calculates rolling volatility and Sharpe ratios, plus beta and correlation when a benchmark is supplied. |
| `rt.risk_free_returns(...)` | Converts dated annual rate observations into holding-period risk-free returns using explicit day-count and compounding rules. |
| `rt.compare_performance(...)` | Combines scenario reports into a numerical comparison table, with checks for compatible assumptions and coverage. |

Reporting accepts either a constant risk-free rate or dated risk-free returns.
The risk-free benchmark is separate from borrowing costs and cash interest.
Summed simple returns, compounded returns, and wealth multiples are kept distinct.
Stopped runs require explicit partial reporting and retain their stop status.

### Research workflows

| Function | What it does |
| --- | --- |
| `rt.lagged_features(...)` | Creates features lagged by supplied trading sessions, keeping availability and warm-up records. |
| `rt.chronological_split(...)` | Splits data into training, validation, and test periods, excluding labels that cross boundaries or arrive too late. |
| `rt.standardize(...)` | Fits feature means and standard deviations on retained training rows, then applies them to the splits. |
| `rt.ResearchStudy(...)` | Creates a study that records candidate selection and final-test use. |
| `study.select(...)` | Selects among supplied candidate predictions using validation loss. |
| `study.evaluate_test(...)` | Evaluates the selected candidate on the final test sample. |

These tools do not train models. `study.audit` records evaluations; repeated test
use must be marked exploratory. Carry the audit into later sessions—the toolkit
cannot detect test data you inspected elsewhere.

## Backtest settings

Pass these objects to the relevant functions to make your assumptions explicit.

| Object | What it controls |
| --- | --- |
| `rt.BuyHoldPolicy(...)` | Entry timing, share sizing, and initial leverage for buy-and-hold. |
| `rt.RebalancePolicy(...)` | Scheduled execution, dividend-receivable funding, and target concentration limits. |
| `rt.TradeCosts(...)` | Commission, spread, and impact costs expressed in basis points. |
| `rt.SquareRootImpactCosts(...)` | Size-dependent impact based on daily volatility and dollar ADV, plus commissions and spreads. |
| `rt.Financing(...)` | Fixed borrowing and cash-interest rates, day counts, debt repayment, and margin limits. |
| `rt.SOFRFinancing(...)` | Historical published SOFR plus a borrowing spread, with separately configured cash interest. |
| `rt.LongShortPolicy(...)` | Sets restricted collateral, separate long/short margin requirements, and gross collateral interest. |
| `rt.StockBorrow(...)` | Supplies annual borrow fees by asset, using fixed assumptions or dated rates. |
| `rt.DividendReinvestment(...)` | Reinvestment of paid dividends into the paying asset, including debt-repayment priority. |

Use exact `quantities=` to add a hedge without resizing existing long positions.
See the [long/short guide](docs/long-short.md) and [runnable example](examples/long_short_hedge.py).

SOFR financing needs supplied rates and a publication calendar. It accrues and
capitalizes interest daily using information available at New York midnight;
it is a research model, not an exact broker loan contract.

## Plots

All plotting functions return Matplotlib `(fig, ax)` objects. They do not show or
save figures automatically, and they do not rerun calculations. The allocation,
attribution and exposure views need a backtest report; they raise a clear error for
a `series_performance` report, which has no ledger.

| Function | What it plots |
| --- | --- |
| `rt.plots.prices(...)` | Asset prices with their adjustment basis labeled. |
| `rt.plots.pnl(...)` | Daily dollar profit and loss. |
| `rt.plots.returns(...)` | Daily net simple returns. |
| `rt.plots.cumulative_returns(...)` | Summed simple returns, compounded returns, or wealth, with an optional benchmark. |
| `rt.plots.equity(...)` | Portfolio equity and a benchmark when included in the report. |
| `rt.plots.drawdown(...)` | Declines from the portfolio's previous equity peak. |
| `rt.plots.distribution(...)` | A histogram of daily simple returns. |
| `rt.plots.correlation(...)` | A return-correlation heatmap. |
| `rt.plots.rolling_risk(...)` | A chosen rolling volatility, Sharpe, beta, or correlation series. |
| `rt.plots.allocation(...)` | Asset, cash, debt, and receivable weights over time. |
| `rt.plots.attribution(...)` | Dollar P&L broken down by component. |
| `rt.plots.risk_contributions(...)` | Estimated contributions to portfolio volatility. |
| `rt.plots.turnover(...)` | Trading volume relative to equity, separating entry, rebalancing, and dividend purchases. |
| `rt.plots.exposures(...)` | Risky asset value, cash, debt, and receivables in currency units. |