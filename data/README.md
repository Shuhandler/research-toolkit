# Local research data

No market data has been downloaded. Future private or large snapshots belong in
`data/local/` (ignored by Git), with a manifest alongside each snapshot. Include
source, retrieval time, adjustment basis, asset identifiers/currency, calendar and
timezone, coverage, checksums, transformations, and redistribution restrictions.
Never overwrite an immutable snapshot in place. Never store credentials here.

Tiny synthetic test inputs are built in `tests/conftest.py`; the one-year
acceptance fixtures live in `examples/snapshots/v1`. Public example data may be
committed only after checking redistribution rights. Parquet preserves exact
types for saved market tables. See [the data contract](../docs/architecture.md).

Implemented snapshot I/O is `rt.save_snapshot(market, new_directory)` and
`rt.load_snapshot(directory)`: typed Parquet files plus a versioned JSON manifest
with SHA-256 file hashes and canonical input identity. See [the API](../docs/api.md).
Self-authored, small acceptance fixtures are committed under `examples/snapshots/v1`;
real/private research inputs still belong in ignored `data/local/`.
