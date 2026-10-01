# Examples

[unlevered_buy_and_hold.py](unlevered_buy_and_hold.py) is a runnable offline 1A/1B
example with synthetic prices, a split, a dividend, entry costs, and numerical
reconciliation outputs. After installing this checkout, run
`python examples/unlevered_buy_and_hold.py` from the repository root. It does not
download data or imply market-data performance conclusions.

[financed_buy_and_hold.py](financed_buy_and_hold.py) reuses the same synthetic
snapshot for 1×/2× financed comparisons and a separate price-shock scenario that
stops on margin breach. Run `python examples/financed_buy_and_hold.py` from the
repository root. It shows `require_complete()` for complete comparisons and
explicitly labels the stopped scenario's requested/actual coverage.

## Acceptance notebook specification — not implemented

Create `buy_and_hold_equities.ipynb` during milestone 1D, after the tested library
exists. Notebook cells should configure inputs, call the library, inspect tables,
plot results, then explain observed findings in separate Markdown cells.

1. Load a versioned local snapshot for five stocks and a benchmark. Show data
   provenance, dates, price/return basis, calendar, missing-data report, and actions.
   Provide an offline synthetic example if real data cannot be redistributed.
2. Set starting equity to $100,000,000 and choose equal or explicit weights,
   entry/end sessions spanning one calendar year, entry cost assumptions, financing
   rates, and margin policy. Include the entry valuation and all subsequent expected
   sessions; do not equate a year mechanically with exactly 252 observations.
3. Run the unlevered case and optionally one or more financed initial-leverage
   cases on identical inputs. Show all resolved assumptions, drift in leverage,
   financing totals, and completion status. Never multiply an unlevered return
   series to manufacture a leveraged backtest.
4. Inspect trades, positions, dividend receivables, cash, debt, costs, and daily
   P&L. Show reconciliation residuals, not just a final performance number.
5. Report daily dollar P&L, daily simple returns, cumulative simple-return sum,
   compounded return, cumulative dollar P&L, ending equity, correlation, beta,
   volatility, drawdown, Sharpe, and Sortino, with units and sample counts.
6. Plot normalized raw prices (mark splits), daily P&L/returns, net equity against a
   normalized benchmark, drawdowns, return distributions, and drifting allocation.
   Clearly separate cumulative sums from compounded returns. Add asset-return
   correlation and dollar attribution views once their calculations are implemented.
7. Interpret actual outputs and limitations in Markdown, including benchmark
   basis, modeled costs, dividend handling, and the lack of terminal liquidation.

Re-running from the same saved inputs must require no network and reproduce the
same numerical tables within declared tolerances. No tickers, dates, five-asset
assumption, or $100 million constant belongs in library code.
