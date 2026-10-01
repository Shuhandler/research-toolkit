# Working instructions

## Start here

- Inspect repository status, existing code, tests, and documentation before editing.
  Preserve user changes; never reset or overwrite unrelated work. Read
  `docs/project-context.md`, `docs/architecture.md`, and
  `docs/financial-conventions.md` before financial implementation.
- This repository is currently in its setup phase. Do not implement functions
  merely because a proposed signature appears in a document. Follow the scope of
  the current user request. Do not create placeholder functions or empty class
  hierarchies for future phases.
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
  Buy-and-hold means fixed quantities apart from corporate actions; drifting
  weights and leverage must not trigger implicit trades.
- Charge transaction costs on actual trades. Reconcile positions, trades, cash,
  receivables, debt, costs, P&L, and equity. State the equity denominator used for
  returns, including entry costs and any external flows.
- Validate calculations with small hand-checkable examples and accounting
  reconciliations before relying on large or visually plausible examples.
- Test meaningful cases: zero trades, changing/drifting weights, flat prices,
  entry costs, dividends, splits, leverage, financing over weekends, inadequate
  equity, insufficient samples, zero volatility, and mismatched dates. Do not
  create tests that merely reproduce an implementation's formulas as its oracle.
- Unit tests must run offline with synthetic or permitted saved fixtures. Keep
  credentials, private data, provider caches, and large snapshots out of Git.
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
