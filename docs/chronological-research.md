# Milestone 3 — signals and chronological research (implemented)

Use `import research_toolkit as rt`. This release adds daily lagged features,
chronological research samples with label-horizon/availability purging, a
training-only standardizer, audited validation selection and final-test evaluation,
and dated long-only instructions executed through the existing scheduled ledger.
No ML dependency or separate trading engine is introduced.

The [Python workflow](../examples/chronological_research.py) and
[notebook](../examples/chronological_research.ipynb) use only the committed synthetic
snapshot. Example model fitting and the forecast-to-allocation rule are visible
research choices outside library calculations. They are not investment evidence.

## Lagged features

```python
features = rt.lagged_features(
    observations, sessions=sessions, columns=["return", "volume"], lags=[1, 5],
    decision_timing="session_close", metadata=feature_metadata,
)
features.values
features.availability
features.diagnostics
```

Inputs are exact-schema eager Polars tables. `sessions` has the existing
`session: Date`, `close_at: Datetime("us", "UTC")` contract. `observations` has
`session: Date`, `asset: String`, `available_at: Datetime("us", "UTC")` and one
Float64 column for every requested feature. Source values must be finite and
nonnull; the complete asset-by-calendar panel is required. Source observations
represent the stated session close, so availability cannot precede that close.
No missing-date fill, frequency inference, interpolation or implicit return-basis
conversion occurs. If using `rt.returns`, explicitly exclude only its structural
first null and the corresponding calendar row before preparing features.

Metadata requires `source`, `calendar`, `calendar_version`, `timezone` (IANA),
`frequency="1d"`, and `feature_units` mapping exactly the selected columns to
nonblank unit strings. Supplied calendars assert completeness; the library cannot
independently verify holidays or provider publication vintages. Retain adjustment
basis, source snapshot identity and any reconstructed-availability caveat in metadata.

Lags are unique positive integer counts of **supplied trading sessions**, not
calendar days. Features are named `column__lagN`; lag zero/negative values raise.
A Friday observation is the one-session lag on the supplied Monday. Asset values
never cross between assets, and future observations cannot affect earlier values.

`FeatureResult` contains:

| Table | Fields |
| --- | --- |
| `values` | `session`, `asset`, `decision_at`, nullable `available_at`, generated Float64 features |
| `availability` | `session`, `asset`, `feature`, nullable `source_session`, `observed_at`, `available_at` for each source value |
| `diagnostics` | `session`, `asset`, `status`, `n_missing` |
| `sessions` | Owned, sorted supplied calendar |

Decision time is the supplied session close. A complete feature row is ready only
when all its source values are available by that decision. Leading rows retain
nulls with `warmup`; delayed publication produces `unavailable`, preserving the
actual lagged value in the audit rather than substituting an older value. Ready
rows have status `ready`. Row availability is the maximum input availability, or
null when the feature vector is incomplete. Metadata and a deterministic identity
retain definitions, lags and source provenance. Treat result tables as immutable;
consumers reject changes after preparation.

## Chronological boundaries and label horizons

```python
split = rt.chronological_split(
    features, labels, train_end=train_end, validation_end=validation_end,
    test_end=test_end, gap_sessions=1, label_metadata=label_metadata,
)
split.train
split.validation
split.test
split.excluded
```

`labels` is exactly `session: Date`, `asset: String`, `label_end: Date`,
`available_at: Datetime("us", "UTC")`, `target: Float64`. Each sample's label
starts from its feature decision session and finishes on a **later** supplied
session. Availability cannot precede the outcome close. Every feature key through
`test_end` needs a label record, including warm-up rows; extras, omissions,
duplicates and nonfinite values raise. Supply enough calendar history beyond the
requested test cutoff to describe the horizons of trailing rows; those crossing
the boundary are explicitly excluded. Label metadata requires `source`, `unit`
and `definition`. Label construction is caller-controlled and not a trade.

For a next-close strategy, a useful forecast label can describe the return from
the next close to the following close, with `label_end` two sessions after the
decision. A return ending at the execution close is not a return earned by the new
position. The example uses analytical corporate-action-aware labels, explicitly
distinct from the net portfolio's cash/receivable/cost accounting.

All three end dates are inclusive, strictly increasing supplied sessions:

- Training decision dates run from the first supplied session through `train_end`.
- Validation starts at the next supplied session and ends at `validation_end`.
- Final test starts at the next supplied session and ends at `test_end`.

Each partition's end is also its outcome and publication cutoff. Remove samples
whose actual `label_end` crosses it, or whose label is unavailable by that closing
time. Thus a label ending exactly on the training cutoff is permitted, but one
extending into validation is purged. Different assets/samples can have different
horizons; no fixed horizon is guessed. Test trailing labels are purged too.

`gap_sessions` explicitly excludes the first N supplied decision sessions after
each train/validation boundary; default zero adds no gap. This optional boundary
gap is separate from the horizon purge. Using past training history to construct
validation/test lagged features is permitted; fitting transforms on future data
is not. No shuffled split or cross-validation fold is introduced.

`ResearchSplit.train`, `.validation`, `.test` retain feature keys, decision and
feature availability, feature values, `label_end`, `label_available_at`, and
`target`. `.excluded` retains original rows plus `partition` and one primary
`reason`: `warmup`, `unavailable`, `boundary_gap`,
`label_horizon_crosses_boundary`, or `label_unavailable_at_boundary`, in that
priority order. Nothing is silently dropped. Each partition must retain at least
one sample. Different asset-specific exclusions may produce unequal asset counts;
these are visible in the excluded table, not filled to a rectangular sample.
Metadata retains boundary/cutoff times, source IDs, units, counts and policy.

## Training-only standardization

```python
scaled = rt.standardize(split, zero_variance="raise")
scaled.transforms
```

This small concrete transform fits each feature's mean and sample standard
deviation (`ddof=1`) using only retained **training** rows, pooled across assets.
It never uses labels or validation/test values to estimate parameters. At least
two training rows are required. The saved training mean and scale transform all
three included partitions. `zero_variance="raise"` rejects a constant training
feature; explicit `"unit_scale"` uses scale one, subtracting only its training
mean. An already standardized split is rejected; retain the original for changes.

`transforms` has `feature`, `mean`, `scale`, `n_obs`, `status`, with status `ok` or
`zero_variance_unit_scale`. Excluded rows remain **original, untransformed** audit
values. Metadata records fitting partition, pooling, source split identity and
parameters. This is not an interchangeable transformer framework. Fit any external
model exclusively on `scaled.train`; validation chooses configurations, and test
remains reserved. Model fitting is not performed by the toolkit.

## Validation selection and final-test audit

```python
study = rt.ResearchStudy(scaled)
selection = study.select(
    validation_predictions, parameters=candidate_parameters,
    metric="mean_squared_error", tie_break="raise",
)
selection.selected_candidate
selection.scores
final = study.evaluate_test(selected_test_predictions)
final.scores
final.audit
```

Predictions have exactly `session: Date`, `asset: String`, `candidate: String`,
`prediction: Float64`, `available_at: Datetime("us", "UTC")`. Each candidate
must cover exactly the retained validation keys; final-test predictions must
cover exactly the retained test keys and contain **only the chosen candidate**.
Predictions cannot precede the feature availability or training cutoff and must
be available by their decision closes. Test predictions also cannot precede
validation selection. Timestamps are caller assertions; the library cannot prove
how externally supplied predictions or universes were produced.

`parameters` maps each candidate identifier to its finite JSON model/strategy
parameter dictionary. These are recorded separately from any portfolio allocation.
`metric` explicitly selects `mean_squared_error` or `mean_absolute_error`, both
minimized using the partition's own labels. No caller-supplied score or label
column can substitute test outcomes for validation. Every candidate is scored on
the same validation observations. Exact ties raise by default; explicitly choosing
`tie_break="candidate_id"` selects the lexicographically first tied identifier.

Scores are numerical Polars tables with `candidate`, `metric`, `value`, `unit`,
`n_obs` (asset-sample rows, not trading days), `partition`. MSE uses squared target
units; MAE uses target units. Neither is a Sharpe ratio or portfolio performance
measure. `ResearchSelection` includes the winner and source/parameter metadata;
selection time is the declared validation boundary close.

The study holds a small ordered audit, with `sequence`, `event`, `candidate`,
`metric`, `value`, `n_obs`, `status`, `split_id`, and JSON-string `parameters`.
The first evaluation following validation selection is `final_test`. Another
final-test evaluation, or new selection after viewing that test, raises unless
`reuse="exploratory"` is explicit. That status is retained in returned
`ResearchEvaluation` metadata and the audit. Once exploratory, reselection cannot
restore a final-test claim. If a holdout was inspected outside this study, explicitly
start selection/evaluation with `reuse="exploratory"`.

```python
# A deliberate repeat is exploratory, even with unchanged parameters.
repeat = study.evaluate_test(selected_test_predictions, reuse="exploratory")
# Persist this table yourself, e.g. as Parquet; no hidden file writes occur.
audit = study.audit
resumed = rt.ResearchStudy(scaled, prior_audit=audit)
```

Restored audits must match the split, sequence, selected parameters, metric and
sample counts, and may not relabel repeated tests as independent. A restored study
requires explicit validation selection again; prior test use still forces
exploratory status. Returned tables are copies, and earlier evaluation results
keep their own audit snapshots. The audit is a research discipline aid, not access
control: creating a fresh study without prior history, reading `.test` directly,
or changing external code cannot be detected. Preserve the audit across sessions.

This milestone evaluates supplied numerical predictions; it does not add model
training, automated hyperparameter search, walk-forward execution, strategy
performance ranking or broker-specific execution.

## Dated signals and next supported execution

```python
instructions = rt.signal_targets(
    signals, sessions=market.sessions, execution="next_session_close",
    rebalance="on_change", metadata=signal_metadata,
)
result = rt.scheduled_rebalance(
    market, targets=instructions, initial_capital=capital,
    entry_session=instructions.targets["session"][0], end_session=end,
    policy=rebalance_policy, costs=costs, financing=financing,
)
```

`signals` has exactly `decision_session: Date`, `asset: String`,
`weight: Float64`, `gross_leverage: Float64`, `observed_at: Datetime("us", "UTC")`,
`available_at: Datetime("us", "UTC")`. Every basket covers the same asset universe,
including explicit zero weights. Nonnegative risky proportions sum to one under
the existing tolerance; leverage is finite and nonnegative, with one value per
basket. Cash uses leverage zero with a valid risky allocation template. There is
no implicit score/probability-to-weight conversion and no negative-position support.
Metadata requires the common source/calendar/timezone/frequency fields plus
`signal_definition`; retain research selection/source identities and parameters.

The decision time is the supplied close of `decision_session`.
`observed_at <= available_at <= decision_at` is mandatory. An order instruction is
recorded at that decision time and executes at the **next supplied session close**,
never the decision close or next open. Weekends/holidays follow the supplied
calendar. Non-session decisions and signals without a next session raise.
Missing market prices still fail market validation; they never cause a later fill.
The next raw close sizes a predetermined dollar target under the existing idealized
fractional fill assumption; this is not a promise of closing-auction liquidity.

`rebalance` is required, not guessed:

- `each_signal` issues a target at every supplied decision, which may trade to
  restore drifted weights/leverage even if the instruction is unchanged.
- `on_change` issues the first target and then only changes in the exact supplied
  weight/leverage basket. Otherwise quantities drift, apart from splits and any
  separately enabled dividend reinvestment. Tiny instruction changes are still
  changes; this is not a turnover threshold.

Absent signal dates create no new target, not a forward-filled trade. Source
instructions remain in `SignalResult.signals`; `.targets` uses the existing
scheduled-target schema. `.audit` records decision/execution sessions, observed,
available, decision, order and execution timestamps, weights/leverage, and `action`
(`target` or `unchanged_target`). `.sessions`, metadata and a snapshot ID retain
calendar and policy provenance. The helper does not read prices or simulate orders.

Pass the **SignalResult itself** to `scheduled_rebalance` to retain the signal
audit and validate its calendar against the market. Passing only `.targets` still
runs the explicit scheduled contract but does not retain signal provenance.
All requested signal execution dates must lie in `[entry_session, end_session)`;
the terminal session is mark-only, including when an unchanged instruction is
submitted. The first target defines entry. Concentration, funding, maintenance,
costs, corporate actions and financing are enforced by the shared engine.
No new fill earns the preceding price movement. Loan interest, spreads and impact
are booked by existing policies; automatic dividend reinvestment remains free.

`BacktestResult.signal_audit` adds `status`: `processed` (a basket processed, not
necessarily a nonzero trade), `unchanged_no_target`, `blocked_at_stop`, or
`not_reached`. Ordinary nonsignal runs have a typed empty table. Stop dates/reasons,
actual coverage and completion guards remain unchanged. Full requested signals
are retained even when a run stops; they are not mislabeled as fills.

Long/short positions, stock lending/collateral, next-open or intraday execution,
stop-order fills, label construction frameworks and walk-forward model workflows
remain separate future work.
