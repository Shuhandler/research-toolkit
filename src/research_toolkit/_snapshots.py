"""Local immutable snapshots; file integrity plus canonical data identity."""

import hashlib
import json
from pathlib import Path
import shutil

import polars as pl

from ._data import _validated_market, prepare_market_data

_TABLES = ("prices", "sessions", "splits", "dividends")


def save_snapshot(market, path) -> Path:
    """Write four Parquet tables and a versioned JSON manifest to a NEW directory.

    Existing paths are never overwritten. Usage restrictions and transformations
    belong in source metadata. File hashes detect corruption, not authenticity.
    """
    market = _validated_market(market)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        files = {}
        for name in _TABLES:
            table = getattr(market, name)
            target = path / f"{name}.parquet"
            table.write_parquet(target)
            files[name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                           "rows": table.height, "schema": {k: str(v) for k, v in table.schema.items()}}
        manifest = {"format_version": 1, "snapshot_id": market.snapshot_id,
                    "metadata": market.metadata, "files": files,
                    "diagnostics": market.diagnostics.to_dicts()}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    except BaseException:
        shutil.rmtree(path)
        raise
    return path


def load_snapshot(path):
    """Verify every file hash/schema and revalidate economic inputs before loading."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or set(manifest.get("files", {})) != set(_TABLES):
        raise ValueError("unsupported or incomplete snapshot manifest")
    tables = {}
    for name in _TABLES:
        target, entry = path / f"{name}.parquet", manifest["files"][name]
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError(f"snapshot hash mismatch: {name}")
        table = pl.read_parquet(target)
        if table.height != entry.get("rows") or {k: str(v) for k, v in table.schema.items()} != entry.get("schema"):
            raise ValueError(f"snapshot schema/row mismatch: {name}")
        tables[name] = table
    market = prepare_market_data(**tables, metadata=manifest["metadata"])
    if market.snapshot_id != manifest.get("snapshot_id"):
        raise ValueError("snapshot canonical identity mismatch")
    return market
