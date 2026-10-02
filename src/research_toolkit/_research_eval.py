"""Validation selection and an explicit, exportable final-test use audit."""
from copy import deepcopy
from datetime import datetime
import json
import math
from statistics import mean

import polars as pl

from ._data import _table, _identity
from ._research import _checked_split, UTC, KEYS
from ._results import ResearchSelection, ResearchEvaluation

PREDICTION_SCHEMA = {**KEYS, "candidate": pl.String, "prediction": pl.Float64, "available_at": UTC}
SCORE_SCHEMA = {"candidate": pl.String, "metric": pl.String, "value": pl.Float64,
                "unit": pl.String, "n_obs": pl.Int64, "partition": pl.String}
AUDIT_SCHEMA = {"sequence": pl.Int64, "event": pl.String, "candidate": pl.String,
    "metric": pl.String, "value": pl.Float64, "n_obs": pl.Int64, "status": pl.String,
    "split_id": pl.String, "parameters": pl.String}
LOSSES = {"mean_squared_error", "mean_absolute_error"}


def _losses(split, predictions, partition, metric, candidates=None):
    if metric not in LOSSES:
        raise ValueError("metric must be mean_squared_error or mean_absolute_error; smaller is better")
    table = _table(predictions, PREDICTION_SCHEMA, "predictions", ["session", "asset", "candidate"], nonempty=True).sort("candidate", "session", "asset")
    names = sorted(table["candidate"].unique())
    if candidates is not None and set(names) != set(candidates):
        raise ValueError("test predictions must contain only the selected candidate")
    sample = getattr(split, partition)
    expected = set(sample.select(*KEYS).iter_rows())
    lookup = {(r["session"], r["asset"]): r for r in sample.iter_rows(named=True)}
    training_cutoff = datetime.fromisoformat(split.metadata["boundaries"]["train"]["cutoff_at"])
    unit = split.metadata["label_metadata"]["unit"]
    unit = f"({unit})^2" if metric == "mean_squared_error" else unit
    rows = []
    for candidate in names:
        subset = table.filter(pl.col("candidate") == candidate)
        if set(subset.select(*KEYS).iter_rows()) != expected:
            raise ValueError(f"candidate {candidate!r} must match every {partition} key exactly")
        errors = []
        for r in subset.iter_rows(named=True):
            observation = lookup[r["session"], r["asset"]]
            if not max(training_cutoff, observation["available_at"]) <= r["available_at"] <= observation["decision_at"]:
                raise ValueError("prediction must be available by decision, after its features and training cutoff")
            difference = r["prediction"]-observation["target"]
            try:
                loss = difference**2 if metric == "mean_squared_error" else abs(difference)
            except OverflowError as exc:
                raise ValueError("prediction loss is not representable") from exc
            if not math.isfinite(loss):
                raise ValueError("prediction loss is not representable")
            errors.append(loss)
        rows.append((candidate, metric, mean(errors), unit, len(errors), partition))
    return pl.DataFrame(rows, schema=SCORE_SCHEMA, orient="row"), table


class ResearchStudy:
    """Small stateful evaluation audit, not a trainer or trading/accounting engine.

    Persist ``audit`` yourself and supply it as ``prior_audit`` when resuming.
    Starting a fresh study cannot detect test inspection outside this audit. Use
    reuse='exploratory' if the holdout has already influenced any research choice.
    """
    def __init__(self, split, *, prior_audit=None):
        self._split = deepcopy(_checked_split(split))
        self._selection = None
        self._audit = pl.DataFrame(schema=AUDIT_SCHEMA) if prior_audit is None else _table(
            prior_audit, AUDIT_SCHEMA, "prior_audit", ["sequence"]).sort("sequence")
        if self._audit["sequence"].to_list() != list(range(self._audit.height)):
            raise ValueError("audit sequence must be contiguous from zero")
        seen_test, selected, tainted = False, None, False
        for row in self._audit.iter_rows(named=True):
            if row["split_id"] != split.snapshot_id or row["metric"] not in LOSSES or row["value"] < 0:
                raise ValueError("prior_audit has incompatible split/metric/loss")
            try:
                params = json.loads(row["parameters"])
                json.dumps(params, allow_nan=False)
                if not isinstance(params, dict):
                    raise ValueError("parameters must be a dictionary")
            except (TypeError, ValueError) as exc:
                raise ValueError("audit parameters must encode a finite JSON dictionary") from exc
            if row["event"] == "validation_selection":
                if row["n_obs"] != split.validation.height or row["status"] not in {"validation", "exploratory"} or ((seen_test or tainted) and row["status"] != "exploratory"):
                    raise ValueError("invalid validation audit status/coverage")
                selected = row
            elif row["event"] == "test_evaluation":
                if selected is None or row["candidate"] != selected["candidate"] or row["metric"] != selected["metric"] or row["parameters"] != selected["parameters"]:
                    raise ValueError("test audit must follow the recorded candidate selection")
                if row["n_obs"] != split.test.height or row["status"] not in {"final_test", "exploratory"} or (
                        (seen_test or tainted or selected["status"] == "exploratory") and row["status"] != "exploratory"):
                    raise ValueError("test reuse must be recorded as exploratory")
                seen_test = True
            else:
                raise ValueError("unknown research audit event")
            tainted = tainted or row["status"] == "exploratory"
        # A restored audit intentionally requires a fresh, explicit selection;
        # prior holdout use remains visible and cannot be reset by selecting again.

    @property
    def audit(self):
        return self._audit.clone()

    @property
    def selection(self):
        return deepcopy(self._selection)

    def _status(self, reuse, *, final=False):
        if reuse not in {"raise", "exploratory"}:
            raise ValueError("reuse must be raise or exploratory")
        used = self._audit.filter(pl.col("event") == "test_evaluation").height > 0
        exploratory = used or "exploratory" in self._audit["status"] or (final and self._selection is not None and self._selection.metadata["status"] == "exploratory")
        if exploratory and reuse != "exploratory":
            raise ValueError("final test already used or research exploratory; explicit reuse='exploratory' is required")
        return "exploratory" if exploratory or reuse == "exploratory" else "final_test" if final else "validation"

    def _record(self, event, row, status, parameters):
        record = (self._audit.height, event, row["candidate"], row["metric"], row["value"], row["n_obs"],
                  status, self._split.snapshot_id, json.dumps(parameters, sort_keys=True, allow_nan=False))
        self._audit = pl.concat([self._audit, pl.DataFrame([record], schema=AUDIT_SCHEMA, orient="row")])

    def select(self, predictions, *, parameters, metric, tie_break="raise", reuse="raise") -> ResearchSelection:
        """Select the lowest validation loss; ties require a named deterministic rule.

        Model/strategy parameter dictionaries are provenance, never portfolio weights.
        Supplied predictions assert the caller fitted their model on training only.
        """
        split = _checked_split(self._split)
        status = self._status(reuse)
        if tie_break not in {"raise", "candidate_id"}:
            raise ValueError("tie_break must be raise or candidate_id")
        scores, table = _losses(split, predictions, "validation", metric)
        if not isinstance(parameters, dict) or set(parameters) != set(scores["candidate"]) or any(not isinstance(v, dict) for v in parameters.values()):
            raise ValueError("parameters must map every candidate to its model/strategy parameter dictionary")
        try:
            params = json.loads(json.dumps(parameters, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must be finite JSON") from exc
        best = scores.filter(pl.col("value") == scores["value"].min()).sort("candidate")
        if best.height > 1 and tie_break == "raise":
            raise ValueError("validation losses tie; explicitly select tie_break='candidate_id' or revise candidates on validation")
        chosen = best.row(0, named=True)
        meta = dict(split_id=split.snapshot_id, selection_partition="validation", status=status,
            selected_parameters=params[chosen["candidate"]], candidate_parameters=params,
            metric=metric, direction="minimize", tie_break=tie_break,
            selected_at=split.metadata["boundaries"]["validation"]["cutoff_at"],
            predictions_id=_identity({"predictions": table}, {}), model_fit_policy="caller_asserted_training_only")
        self._selection = ResearchSelection(scores, chosen["candidate"], meta)
        self._record("validation_selection", chosen, status, params[chosen["candidate"]])
        return self.selection

    def evaluate_test(self, predictions, *, reuse="raise") -> ResearchEvaluation:
        """Evaluate only the selected candidate; repeated evaluations are exploratory."""
        split = _checked_split(self._split)
        if self._selection is None:
            raise ValueError("select a candidate on validation before evaluating the final test")
        status = self._status(reuse, final=True)
        selected = self._selection
        scores, table = _losses(split, predictions, "test", selected.metadata["metric"], [selected.selected_candidate])
        selected_at = datetime.fromisoformat(selected.metadata["selected_at"])
        if (table["available_at"] < selected_at).any():
            raise ValueError("test predictions cannot precede validation selection")
        self._record("test_evaluation", scores.row(0, named=True), status, selected.metadata["selected_parameters"])
        return ResearchEvaluation(scores, self.audit, dict(split_id=split.snapshot_id, status=status,
            selected_candidate=selected.selected_candidate, selected_parameters=deepcopy(selected.metadata["selected_parameters"]),
            selected_at=selected.metadata["selected_at"], predictions_id=_identity({"predictions": table}, {}),
            test_boundary=deepcopy(split.metadata["boundaries"]["test"]),
            audit_scope="caller_preserved_history_not_external_test_access_detection"))
