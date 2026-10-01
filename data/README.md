# Local research data

No market data has been downloaded. Future private or large snapshots belong in
`data/local/` (ignored by Git), with a manifest alongside each snapshot. Include
source, retrieval time, adjustment basis, asset identifiers/currency, calendar and
timezone, coverage, checksums, transformations, and redistribution restrictions.
Never overwrite an immutable snapshot in place. Never store credentials here.

Small synthetic test fixtures belong in `tests/fixtures/` when tests are added.
Public example data may be committed only after checking redistribution rights.
CSV plus JSON is sufficient for tiny fixtures; add Parquet support when its first
consumer exists. See [the data contract](../docs/architecture.md).
