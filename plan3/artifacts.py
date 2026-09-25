"""Small, explicit artifact contracts; no cache is identified by its filename alone."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hashes(package: str) -> dict:
    return {p.name: sha256(p) for p in sorted((ROOT / package).glob("*.py"))}


def versions() -> dict:
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "polars", "scipy", "scikit-learn", "lightgbm",
             "rapidfuzz", "anyascii", "sparse-dot-topn")}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def contract(path: Path, expected: dict) -> None:
    """Create a contract, or reject changed input/config/code before reading any cache."""
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"Artifact contract changed: {path}. Choose a new output directory.")
    else:
        write_json(path, expected)


def parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.write_parquet(tmp)
    os.replace(tmp, path)


def records(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       null_values=None, empty_string_is_null=False)


def load_prepared(path: Path) -> dict:
    meta = read_json(path / "prepared.json")
    if meta["baseline_hashes"] != source_hashes("plan1"):
        raise ValueError("Imported Plan 1 code changed since preparation; prepare a new snapshot.")
    return meta


def partition_path(prepared: Path, split: str, country: str, source: str) -> Path:
    key = fingerprint(country)[:16]  # country is arbitrary data, never a filesystem path
    return prepared / "normalized" / split / f"{source}-{key}.parquet"
