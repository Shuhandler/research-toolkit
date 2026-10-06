import json

import pytest

import research_toolkit as rt


def test_roundtrip_and_replay(market, run, tmp_path):
    path = rt.save_snapshot(market, tmp_path/"saved")
    loaded = rt.load_snapshot(path)
    assert loaded.snapshot_id == market.snapshot_id
    for name in ("prices", "sessions", "splits", "dividends"):
        assert getattr(loaded, name).equals(getattr(market, name))
    assert run(loaded).daily.equals(run(market).daily)
    with pytest.raises(FileExistsError): rt.save_snapshot(market, path)


def test_file_corruption(market, tmp_path):
    path = rt.save_snapshot(market, tmp_path/"saved")
    with (path/"prices.parquet").open("ab") as stream: stream.write(b"damage")
    with pytest.raises(ValueError, match="hash mismatch"): rt.load_snapshot(path)


@pytest.mark.parametrize("edit", ["identity", "metadata", "schema", "version", "files"])
def test_manifest_corruption(market, tmp_path, edit):
    path = rt.save_snapshot(market, tmp_path/"saved")
    target = path/"manifest.json"
    m = json.loads(target.read_text())
    if edit == "identity": m["snapshot_id"] = "bad"
    if edit == "metadata": m["metadata"]["source"] = "changed"
    if edit == "schema": m["files"]["prices"]["schema"]["close"] = "Int64"
    if edit == "version": m["format_version"] = 2
    if edit == "files": del m["files"]["splits"]
    target.write_text(json.dumps(m))
    with pytest.raises(ValueError): rt.load_snapshot(path)
