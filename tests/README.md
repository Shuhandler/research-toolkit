# Tests

No financial functions or tests exist yet. Add pytest when implementing the first
slice in [the roadmap](../docs/roadmap.md); follow the
[testing plan](../docs/testing-plan.md).

Organize real tests by behavior: data validation, return arithmetic, ledger and
corporate actions, financing and costs, metrics, then plots. Keep tiny synthetic
fixtures here under `fixtures/` once needed. Unit tests must not use the network.
The five-stock acceptance notebook complements, but does not replace, accounting
tests. Do not add empty test modules or placeholder assertions.
