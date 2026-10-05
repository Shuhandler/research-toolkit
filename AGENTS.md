# Working instructions

## Start here

- Inspect repository status, existing code, tests, and documentation before editing.
  Preserve user changes; never reset or overwrite unrelated work. Read
  `docs/project-context.md`, `docs/architecture.md`, and
  `docs/financial-conventions.md` before financial implementation.
- Milestones 1A–1D, 2 and 3 are implemented; read `docs/api.md` for supported behavior.
  Do not implement later functions merely because a proposed signature appears in
  a document. Follow the current user request. Do not create placeholder functions
  or empty class hierarchies for future phases.
- Work only in this repository. The local reference project is read-only. Never
  push to GitHub or publish a package without explicit user authorization.

## Design and notebook experience

- Use `import research_toolkit as rt`; Polars is the confirmed public table
  interface. Keep the public API small, consistent, documented, and convenient in
  notebooks. Put implementation in focused internal modules; expose only useful
  supported functions and result types.
- Keep calculation, visualization, data access, and narrative interpretation
  separate. Calculations return numerical data with units and diagnostics. Plots
  consume those results and return customizable figure objects. Neither should
  perform hidden network calls or rerun simulations.
- Never generate narrative conclusions through if/else branches in calculation
  code. Analysis belongs in separate Markdown cells, written after inspecting the
  actual outputs and plots. Structured validation errors and metric status codes
  are appropriate; automated investment commentary is not.
- Keep machine learning optional. Core imports and plotting/backtesting must not
  import PyTorch, scikit-learn, provider clients, or interactive plotting engines.
- Add dependencies only when implemented behavior needs them. Prefer a direct,
  readable implementation to speculative frameworks or interchangeable backends.
- Document proposed and implemented behavior separately. Keep README, API docs,
  examples, schemas, and tests aligned whenever behavior changes.

## Financial correctness

- Never silently forward-fill missing asset prices, drop mismatched dates, alter
  return conventions, change annualization, or assume an execution policy.
  Timestamp joins must be explicit and checked. Reject unsupported inputs rather
  than silently approximating their economics.
- State assumptions explicitly, make them configurable where supported, and save
  their resolved values in results. Do not infer market calendars, adjustment
  basis, risk-free rates, borrowing rates, or asset classes from ticker strings.
- Distinguish prices from returns, log returns from simple returns, cumulative sums
  from compounding, wealth multiples from percentage returns, model coefficients
  from portfolio weights, and measured data from cost assumptions.
- Raw execution prices plus explicit split/dividend accounting and initial
  leverage/financing are confirmed first-release requirements. Never treat an
  adjusted series as executable share prices or count its dividends twice.
- Use only information available before the specified decision/execution time.
  Buy-and-hold means fixed quantities apart from splits and explicitly enabled
  payment-funded dividend purchases; drifting weights and leverage must not trigger
  implicit trades. Preserve the selected dividend funding priority, scheduled-trade
  collision policy, final-session mark-only rule, and per-payment audit records.
- Keep buy-and-hold and scheduled strategies on the shared accounting engine.
  Scheduled decisions must precede execution; trailing allocation/risk inputs end
  strictly before the decision session. Missing/non-session execution dates raise.
  Preserve explicit reserve-versus-require-target receivable funding and pre-trade
  margin checks. Do not silently turn a target concentration limit into a rule that
  trades whenever actual weights drift. Analytical ex-date-reinvested returns are
  distinct from the executable receivable/cash ledger.
- Signed portfolios use explicit equity exposures or exact quantities, never negative
  long-allocation weights. Preserve daily marked segregated collateral and separate
  long/short margin fractions (user-confirmed choices). Restricted cash cannot silently
  fund longs or repay debt. Equity includes signed values and collateral once.
  Borrow fees use prior supplied closing short values and explicit calendar-day rates;
  gross collateral rebate is separate, never a net fee quote. Short dividends accrue
  liabilities and settle once; long DRIP must not auto-cover a now-short payer.
- Signed scheduled dollar targets are predetermined currency amounts. Convert inside
  the ledger at execution raw closes; never shrink them to pay costs or normalize by
  equity. Preserve explicit zero exits, target residual audits and terminal mark-only.
  Dated square-root cost mappings match exact decision dates and the complete basket
  universe, including exits/hedges. Bind once per basket, retain model identities and
  prior-session cutoffs, and keep one continuous accounting state.
- Keep fixed-rate and historical SOFR financing distinct. SOFR rates require a
  complete supplied publication calendar, point-in-time availability timestamps,
  declared coverage/units and an age limit. Never apply an observation before its
  publication, silently invent holiday dates, infer percent-to-decimal conversion,
  or substitute a current quote for missing history. Preserve ACT/360 versus
  ACT/365F, separate cash/loan rates, and the daily accrual audit. The current SOFR
  model uses rates known at New York midnight and daily capitalization; do not
  describe it as the official SOFR Index or a broker's exact loan contract.
- Automatic dividend reinvestment has zero commission, spread and impact by user
  decision. Preserve its zero-cost trade/audit records; normal entry and scheduled
  rebalancing retain configured costs, including when funded by released dividends.
- Preserve explicit cap redistribution versus rejection, dated RF versus scalar
  inputs, and portfolio-return versus excess-return Sharpe denominators. Match
  both holding-interval endpoints. Rolling beta/correlation are descriptive only.
- Square-root costs use daily volatility and dollar ADV; order/ADV is not intraday
  participation. Bind the frozen pre-decision liquidity snapshot to actual orders,
  size against post-cost equity, and never debit a preview in addition to the ledger.
- Provider Close adjustment semantics must be checked. Yahoo raw reconstruction
  requires explicit cumulative split factors through retrieval; supplied calendars,
  payment dates and declared volume units remain mandatory. Reconstructed close-time
  availability is not a point-in-time vintage. Keep adapters offline by default.
- Charge transaction costs on applicable actual trades. Reconcile positions, trades, cash,
  receivables, debt, costs, P&L, and equity. State the equity denominator used for
  returns, including entry costs and any external flows.
- Preserve stopped-run status and actual coverage in all downstream consumers.
  Full-period reports must call `result.require_complete()`; partial reports
  need an explicit opt-in and must show the stop reason and actual ending session.
- Validate calculations with small hand-checkable examples and accounting
  reconciliations before relying on large or visually plausible examples.
- Test meaningful cases: zero trades, changing/drifting weights, flat prices,
  entry costs, dividends, splits, leverage, financing over weekends, inadequate
  equity, insufficient samples, zero volatility, and mismatched dates. Do not
  create tests that merely reproduce an implementation's formulas as its oracle.
- Unit tests must run offline with synthetic or permitted saved fixtures. Keep
  credentials, private data, provider caches, and large snapshots out of Git.
- Research features must retain source availability and lag by supplied sessions.
  Purge label horizons and late publications at chronological boundaries; retain
  exclusions explicitly. Standardization fits only retained training rows.
- Signal instructions declare next-session-close execution and each-signal versus
  on-change rebalancing. Preserve the signal audit, raw-price ledger, terminal
  mark-only rule and stop status. No same-close signal may earn a preceding move.
- Keep ResearchStudy audits across sessions. Repeated final-test evaluation or
  selection after test inspection must stay exploratory; a new object does not
  make a previously inspected holdout independent. External access is not detected.
- Preserve an untouched final test sample. Fit transformations only on training
  data; choose models and strategies on validation data. Record any reuse of a
  final test as exploratory, not as independent final-test evidence.

## Decisions, provenance, and completion

- Ask concise multiple-choice questions in ordinary chat only when an unresolved
  choice materially affects design. Include a recommendation and its trade-off.
  Do not use disappearing questionnaire widgets. Honor prior choices; make routine
  scaffolding choices independently.
- Update `docs/project-context.md` after consequential user decisions. Distinguish
  user-confirmed decisions from engineering proposals and deferred choices.
- Check the reference project's license before any code reuse. Preserve required
  attribution and notices for copied or substantially adapted code. The recorded
  reference license is MIT, copyright 2025 memlabs-research; this project's own
  license is undecided. See `docs/reference-review.md` for exact provenance.
- Run checks appropriate to the change. Report what was verified, what remains
  unimplemented, and material limitations. Do not present planned APIs or financial
  outputs as tested implementation.
