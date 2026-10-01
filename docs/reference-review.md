# Local reference review

Inspected read-only during repository setup: `video1.ipynb`, `research.py`,
`models.py`, `binance.py`, `README.md`, and `LICENSE`, located at:

```text
/Users/jordanshuhandler/Library/Mobile Documents/com~apple~CloudDocs/Test Environment/build-a-quant-trading-strategy-main
```

Review used notebook source cells and complete Python/document source inspection,
not notebook execution. Python files parse as ASTs; this is not proof of runtime
correctness. No market data was downloaded and no model was trained. Notebook cell
numbers below are **zero-based positions**, not execution counts. Line references
describe the local files at the hashes below, not an upstream version.

## What is useful

The notebook demonstrates the convenience of one `research` import, readable
feature-building calls, reusable charts, and chronological rather than random
splits. Preserve that convenience while separating data access, metrics, accounts,
plots, and optional model work. Do not reuse its trading outputs as ground truth.

## Verified findings

| Finding | Local evidence | Design consequence |
| --- | --- | --- |
| `add_tx_fees_log()` adds `log(fee)` | `research.py:886–893` | A small fee becomes a huge negative log change. Even a single wealth haircut would use `log(1-fee)`, not `log(fee)`; actual trade costs belong in the account ledger with the correct notional and timing. |
| Correlation defaults to closing price levels and horizontal concatenation | `research.py:650–656` | Row positions can mismatch timestamps. Compare explicitly joined return intervals and disclose sample size. |
| Wealth multiple/log quantities have ambiguous metric names | `research.py:857–873,895–928`; notebook cells 27–43 | `compound_return` is `exp(sum(log_returns))` without subtracting 1; `equity_curve` is a log sum; extrema/drawdown are log-space values. Use names, units, and conversions that match economics. |
| Annualization defaults are crypto-oriented and conflict with docstrings | `research.py:289–324` | Actual defaults are 365 days/24 hours; docstrings say 252/6.5. Require an explicit convention rather than inheriting either. |
| Sampling frequency changes without refreshing annualization | Notebook cell 4 computes the 1-hour factor; cell 52 switches to 8 hours; cells 54/56 reuse the earlier factor | The supplied Sharpe multiplier is too large by sqrt(8) under the same 365×24 convention. Bind the declared frequency to metric settings. |
| Model selection reuses the test sample | `research.py:941–956,1016–1032`; notebook cells 45–47 and 54–57 | Configurations are ranked by test Sharpe, then the winning configuration is showcased. Chronological train/test splitting alone does not prevent selection leakage; use validation and an untouched test. |
| P&L helper work is unfinished | `research.py:972–982,1051–1084` | `learn_model_trade_pnl()` builds results but returns nothing. `add_trade_log_returns()` returns nothing and has adjacent expression calls where commas are missing; these parse but are not a valid intended Polars argument list. Its annotated list/array input also does not support the assumed `.alias()` operation. |
| Leverage/trading simulation lacks a portfolio account | `research.py:857–873,1051–1112` | Signed log-return proxies, row-by-row entry/exit fees, and leveraged notional compounding do not track multi-asset shares, cash, receivables, debt, financing, or margin. Build an explicit ledger. |

Notebook cell 49 separately uses `log(1-2*fee)`. That avoids the specific
`log(fee)` error but still assumes a round trip on every observation and a
particular fee base; it is not a substitute for actual trade cash flows.

Additional observations:

- `research.py:24–39` imports Polars, PyTorch, NumPy, Altair, and Matplotlib together.
  Simple analysis therefore depends on an ML/plotting stack. `models.py` contains
  simple linear and nonlinear PyTorch models and should inform only optional work.
- `binance.py` combines Binance USD-M futures download paths, local caches, maker/
  taker constants, and aggregation. Range download helpers catch errors and keep
  going; a partial series can look usable without a completeness contract.
  `research.py:157–285` similarly skips missing/error daily files. Offline snapshots
  and strict expected-session validation must be independent of provider clients.
- `research.py:477–486,489–547` creates/shows Matplotlib charts without returning
  the figure; other helpers return Altair objects. Choose a consistent plotting
  return contract and separate display from calculation.
- `README.md` only gives environment installation instructions (Python 3.12 and a
  combined data/ML/plotting stack). It provides no accounting or metric contract.

## License and provenance

The local `LICENSE` is MIT, copyright **2025 memlabs-research**. It requires the
copyright and permission notice in copies or substantial portions of the software.
No source code or notebook content was copied into this repository; this document
records independent findings. If future agents copy or substantially adapt code,
preserve the full applicable notice with that reuse and record its source/version.
The reference license is not a license selection for this new project; the owner
must choose that before external distribution.

The initial repository contained only `.gitignore` and a title-only `README.md`,
and had no uncommitted changes. These fingerprints identify the inspected reference
and allow verification that it remains unchanged:

| File | SHA-256 |
| --- | --- |
| `video1.ipynb` | `a3c4584b60b83e586819c8bf222bd7e6af11aae2cf19908af48c9a0487f38175` |
| `research.py` | `16dc72ae00ab6c2c4c0559bf990649e43c98563f3afa722281efcfd5f6a7b9e1` |
| `models.py` | `7d035a7f945fd8bb68d1ca77ae4115375ad5959d2f746c9ab5d5052484e308bf` |
| `binance.py` | `bccc8a6693d33007a629659bf82f97440eebf1f4ede00c63fee18ea2ba0b303f` |
| `README.md` | `3837d2c3b46ce4db7e4bd4641b3cbd633ec7ece1336318de2b2a2ee9a6530244` |
| `LICENSE` | `67ce32348655a6815c5bfebe34ecb5eda27daefdeba86dc128ab785dadef3a0b` |
