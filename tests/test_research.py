from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
import math

import polars as pl
import pytest
import research_toolkit as rt
from research_toolkit._research import LABEL_SCHEMA
from research_toolkit._research_eval import PREDICTION_SCHEMA


@pytest.fixture
def research_inputs():
    # Fifteen supplied weekday sessions, not an inferred exchange calendar.
    days = [date(2024, 1, 2)+timedelta(days=i) for i in range(21) if (date(2024, 1, 2)+timedelta(days=i)).weekday() < 5]
    closes = [datetime.combine(d, time(21), timezone.utc) for d in days]
    sessions = pl.DataFrame({"session": days, "close_at": closes}, schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")})
    observations = pl.DataFrame([(d, a, t, float(i + offset)) for i, (d, t) in enumerate(zip(days, closes))
        for a, offset in [("A", 1), ("B", 3)]],
        schema={"session": pl.Date, "asset": pl.String, "available_at": pl.Datetime("us", "UTC"), "x": pl.Float64}, orient="row")
    labels = pl.DataFrame([(d, a, days[i+1], closes[i+1], float(i%2)) for i, d in enumerate(days[:13]) for a in ["A", "B"]], schema=LABEL_SCHEMA, orient="row")
    meta = dict(source="synthetic measurements", calendar="supplied weekday fixture", calendar_version="1", timezone="America/New_York",
                frequency="1d", feature_units={"x": "arbitrary_units"})
    return dict(days=days, sessions=sessions, observations=observations, labels=labels, metadata=meta)


def features(data, *, observations=None, lags=(1,)):
    return rt.lagged_features(data["observations"] if observations is None else observations, sessions=data["sessions"],
        columns=["x"], lags=list(lags), decision_timing="session_close", metadata=data["metadata"])


def split(data, *, f=None, labels=None, gap=0):
    return rt.chronological_split(features(data) if f is None else f, data["labels"] if labels is None else labels,
        train_end=data["days"][4], validation_end=data["days"][8], test_end=data["days"][12], gap_sessions=gap,
        label_metadata={"source": "synthetic future labels", "unit": "fraction", "definition": "one supplied session after decision"})


def predictions(part, values):
    # Every candidate provides one timestamped prediction per retained sample.
    return pl.concat([part.select("session", "asset", pl.lit(candidate).alias("candidate"),
        pl.lit(value).cast(pl.Float64).alias("prediction"), pl.col("decision_at").alias("available_at"))
        for candidate, value in values.items()]).select(*PREDICTION_SCHEMA)


def test_lags_follow_sessions_and_retain_lineage(research_inputs):
    data = research_inputs
    result = features(data, lags=(1, 2))
    a = result.values.filter(pl.col("asset") == "A")
    assert a["x__lag1"].to_list()[:4] == [None, 1., 2., 3.]
    assert a["x__lag2"].to_list()[:4] == [None, None, 1., 2.]
    monday = data["days"][4]
    assert monday.weekday() == 0
    row = result.availability.filter((pl.col("session") == monday) & (pl.col("asset") == "A") & (pl.col("feature") == "x__lag1")).row(0, named=True)
    assert row["source_session"].weekday() == 4
    assert row["available_at"] < a.filter(pl.col("session") == monday)["decision_at"][0]
    assert result.diagnostics.filter(pl.col("status") == "warmup").height == 4
    assert result.metadata["lag_unit"] == "supplied_trading_sessions"


def test_future_observations_cannot_change_earlier_features(research_inputs):
    data = research_inputs
    changed = data["observations"].with_columns(pl.when(pl.col("session") >= data["days"][8]).then(pl.col("x")*100)
        .otherwise(pl.col("x")).alias("x"))
    a, b = features(data), features(data, observations=changed)
    assert a.values.filter(pl.col("session") <= data["days"][8]).equals(b.values.filter(pl.col("session") <= data["days"][8]))
    assert a.availability.equals(b.availability)


def test_delayed_feature_is_excluded_not_filled(research_inputs):
    data = research_inputs
    observations = data["observations"].with_columns(pl.when(pl.col("session") == data["days"][1])
        .then(pl.col("available_at")+pl.duration(days=8)).otherwise(pl.col("available_at")).alias("available_at"))
    f = features(data, observations=observations)
    delayed = f.values.filter((pl.col("session") == data["days"][2]) & (pl.col("asset") == "A"))
    assert delayed["x__lag1"][0] == 2.  # Retained audit value, not an older usable observation.
    result = split(data, f=f)
    assert result.excluded.filter(pl.col("reason") == "unavailable").height == 2
    assert data["days"][2] not in result.train["session"]


@pytest.mark.parametrize("edit", ["missing", "duplicate", "nan", "null", "early_availability", "zero_lag", "negative_lag", "bad_units", "calendar_close"])
def test_feature_invalid_inputs(research_inputs, edit):
    data = deepcopy(research_inputs)
    if edit == "missing": data["observations"] = data["observations"].tail(-1)
    if edit == "duplicate": data["observations"] = pl.concat([data["observations"], data["observations"].head(1)])
    if edit == "nan": data["observations"] = data["observations"].with_columns(pl.lit(float("nan")).alias("x"))
    if edit == "null": data["observations"] = data["observations"].with_columns(pl.lit(None, dtype=pl.Float64).alias("x"))
    if edit == "early_availability": data["observations"] = data["observations"].with_columns(pl.col("available_at")-pl.duration(seconds=1))
    if edit == "bad_units": data["metadata"]["feature_units"] = {}
    if edit == "calendar_close": data["sessions"] = data["sessions"].with_columns(pl.col("close_at")+pl.duration(days=1))
    with pytest.raises(ValueError):
        features(data, lags=(0,) if edit == "zero_lag" else (-1,) if edit == "negative_lag" else (1,))


def test_split_boundaries_purge_horizons_and_gap(research_inputs):
    data = research_inputs
    result = split(data)
    assert result.train["session"].unique().sort().to_list() == data["days"][1:4]
    assert result.validation["session"].unique().sort().to_list() == data["days"][5:8]
    assert result.test["session"].unique().sort().to_list() == data["days"][9:12]
    assert result.excluded.filter(pl.col("reason") == "label_horizon_crosses_boundary").height == 6
    longer = data["labels"].with_columns(pl.when(pl.col("session") == data["days"][3]).then(pl.lit(data["days"][5])).otherwise(pl.col("label_end")).alias("label_end"),
        pl.when(pl.col("session") == data["days"][3]).then(pl.lit(data["sessions"]["close_at"][5])).otherwise(pl.col("available_at")).alias("available_at"))
    assert data["days"][3] not in split(data, labels=longer).train["session"]
    gapped = split(data, gap=1)
    assert gapped.validation["session"].min() == data["days"][6]
    assert gapped.test["session"].min() == data["days"][10]
    assert gapped.excluded.filter(pl.col("reason") == "boundary_gap").height == 4
    # Same target horizon, later publication: unavailable by the training cutoff.
    late = data["labels"].with_columns(pl.when(pl.col("session") == data["days"][2]).then(pl.lit(data["sessions"]["close_at"][5]))
        .otherwise(pl.col("available_at")).alias("available_at"))
    assert split(data, labels=late).excluded.filter(pl.col("reason") == "label_unavailable_at_boundary").height == 2


@pytest.mark.parametrize("edit", ["gap", "missing", "early_label", "backward", "boundary", "modified_features"])
def test_split_invalid_or_empty_samples(research_inputs, edit):
    data = deepcopy(research_inputs)
    if edit == "missing": data["labels"] = data["labels"].tail(-1)
    if edit == "early_label": data["labels"] = data["labels"].with_columns(pl.col("available_at")-pl.duration(days=30))
    if edit == "backward": data["labels"] = data["labels"].with_columns(pl.col("session").alias("label_end"))
    if edit == "boundary": data["days"][4] = data["days"][8]
    f = features(data)
    if edit == "modified_features": f = replace(f, values=f.values.with_columns(pl.lit(1.).alias("x__lag1")))
    with pytest.raises(ValueError): split(data, f=f, gap=10 if edit == "gap" else 0)


def test_training_only_standardization_hand_oracle(research_inputs):
    data = research_inputs
    raw = split(data)
    result = rt.standardize(raw)
    # Training features are A:1,2,3 and B:3,4,5. Mean 3, squared deviations sum 10.
    assert result.transforms["mean"][0] == 3.
    assert result.transforms["scale"][0] == pytest.approx(math.sqrt(2.))
    assert result.transforms["n_obs"][0] == 6
    assert result.validation["x__lag1"][0] == pytest.approx((5.-3.)/math.sqrt(2.))
    assert result.excluded.equals(raw.excluded)
    changed = data["observations"].with_columns(pl.when(pl.col("session") >= data["days"][4]).then(pl.col("x")*1000).otherwise(pl.col("x")).alias("x"))
    second = rt.standardize(split(data, f=features(data, observations=changed)))
    assert result.transforms.equals(second.transforms)
    assert result.train.equals(second.train)
    with pytest.raises(ValueError, match="already standardized"): rt.standardize(result)
    with pytest.raises(ValueError, match="changed"):
        rt.standardize(replace(raw, train=raw.train.with_columns(pl.lit(3.).alias("x__lag1"))))


def test_flat_training_requires_explicit_policy(research_inputs):
    data = research_inputs
    f = features(data, observations=data["observations"].with_columns(pl.lit(4.).alias("x")))
    with pytest.raises(ValueError, match="zero training variance"): rt.standardize(split(data, f=f))
    result = rt.standardize(split(data, f=f), zero_variance="unit_scale")
    assert result.transforms["scale"][0] == 1.
    assert result.train["x__lag1"].to_list() == [0.]*6


def test_selection_is_validation_only_and_test_reuse_is_audited(research_inputs):
    result = split(research_inputs)
    study = rt.ResearchStudy(result)
    pred = predictions(result.validation, {"half": .5, "zero": 0.})
    selection = study.select(pred, parameters={"half": {"constant": .5}, "zero": {"constant": 0.}}, metric="mean_squared_error")
    assert selection.selected_candidate == "half"
    assert selection.scores.filter(pl.col("candidate") == "half")["value"][0] == .25
    assert selection.metadata["selection_partition"] == "validation"
    assert selection.metadata["selected_parameters"] == {"constant": .5}
    final = predictions(result.test, {"half": .5})
    evaluated = study.evaluate_test(final)
    assert evaluated.metadata["status"] == "final_test"
    assert evaluated.scores["value"][0] == .25
    with pytest.raises(ValueError, match="exploratory"): study.evaluate_test(final)
    reused = study.evaluate_test(final, reuse="exploratory")
    assert reused.metadata["status"] == "exploratory"
    with pytest.raises(ValueError, match="exploratory"):
        study.select(pred, parameters={"half": {}, "zero": {}}, metric="mean_squared_error")
    study.select(pred, parameters={"half": {}, "zero": {}}, metric="mean_squared_error", reuse="exploratory")
    assert study.selection.metadata["status"] == "exploratory"
    assert study.audit["sequence"].to_list() == [0, 1, 2, 3]
    assert evaluated.audit.height == 2  # Earlier results retain the original audit snapshot.


def test_future_labels_do_not_change_validation_winner(research_inputs):
    data = research_inputs
    one = split(data)
    altered = data["labels"].with_columns(pl.when(pl.col("session") > data["days"][8]).then(100.).otherwise(pl.col("target")).alias("target"))
    two = split(data, labels=altered)
    results = [rt.ResearchStudy(s).select(predictions(s.validation, {"half": .5, "zero": 0.}),
        parameters={"half": {}, "zero": {}}, metric="mean_squared_error") for s in (one, two)]
    assert results[0].scores.equals(results[1].scores)
    assert results[0].selected_candidate == results[1].selected_candidate


def test_holdout_cannot_select_candidate_and_ties_are_explicit(research_inputs):
    s = split(research_inputs)
    study = rt.ResearchStudy(s)
    with pytest.raises(ValueError, match="validation key"):
        study.select(predictions(s.test, {"a": 0.}), parameters={"a": {}}, metric="mean_absolute_error")
    with pytest.raises(ValueError, match="select a candidate"): study.evaluate_test(predictions(s.test, {"a": 0.}))
    tied = predictions(s.validation, {"a": .5, "b": .5})
    with pytest.raises(ValueError, match="tie"): study.select(tied, parameters={"a": {}, "b": {}}, metric="mean_absolute_error")
    study.select(tied, parameters={"a": {}, "b": {}}, metric="mean_absolute_error", tie_break="candidate_id")
    with pytest.raises(ValueError, match="selected candidate"): study.evaluate_test(predictions(s.test, {"b": .5}))
    assert study.audit.height == 1


@pytest.mark.parametrize("edit", ["missing", "duplicate", "unavailable", "too_early", "infinite"])
def test_prediction_contract(research_inputs, edit):
    s = split(research_inputs)
    table = predictions(s.validation, {"a": .5})
    if edit == "missing": table = table.tail(-1)
    if edit == "duplicate": table = pl.concat([table, table.head(1)])
    if edit == "unavailable": table = table.with_columns(pl.col("available_at")+pl.duration(seconds=1))
    if edit == "too_early": table = table.with_columns(pl.col("available_at")-pl.duration(days=50))
    if edit == "infinite": table = table.with_columns(pl.lit(float("inf")).alias("prediction"))
    with pytest.raises(ValueError): rt.ResearchStudy(s).select(table, parameters={"a": {}}, metric="mean_squared_error")


def test_resumed_audit_cannot_reset_test_use(research_inputs):
    s = split(research_inputs)
    study = rt.ResearchStudy(s)
    val, test = predictions(s.validation, {"a": .5}), predictions(s.test, {"a": .5})
    study.select(val, parameters={"a": {}}, metric="mean_squared_error")
    study.evaluate_test(test)
    resumed = rt.ResearchStudy(s, prior_audit=study.audit)
    with pytest.raises(ValueError, match="exploratory"):
        resumed.select(val, parameters={"a": {}}, metric="mean_squared_error")
    resumed.select(val, parameters={"a": {}}, metric="mean_squared_error", reuse="exploratory")
    assert resumed.evaluate_test(test, reuse="exploratory").metadata["status"] == "exploratory"
    bad = resumed.audit.with_columns(pl.when(pl.col("sequence") == 3).then(pl.lit("final_test")).otherwise(pl.col("status")).alias("status"))
    with pytest.raises(ValueError, match="exploratory"): rt.ResearchStudy(s, prior_audit=bad)
    external = rt.ResearchStudy(s)
    external.select(val, parameters={"a": {}}, metric="mean_squared_error", reuse="exploratory")
    with pytest.raises(ValueError, match="exploratory"):
        external.select(val, parameters={"a": {}}, metric="mean_squared_error")


def test_audit_parquet_roundtrip_and_returned_selection_is_a_copy(research_inputs, tmp_path):
    s = split(research_inputs)
    study = rt.ResearchStudy(s)
    val = predictions(s.validation, {"a": .5})
    selection = study.select(val, parameters={"a": {"coefficient": .5}}, metric="mean_absolute_error")
    selection.metadata["selected_parameters"]["coefficient"] = 99.
    assert study.selection.metadata["selected_parameters"] == {"coefficient": .5}
    study.evaluate_test(predictions(s.test, {"a": .5}))
    file = tmp_path/"audit.parquet"
    study.audit.write_parquet(file)
    resumed = rt.ResearchStudy(s, prior_audit=pl.read_parquet(file))
    with pytest.raises(ValueError, match="exploratory"):
        resumed.select(val, parameters={"a": {}}, metric="mean_absolute_error")
