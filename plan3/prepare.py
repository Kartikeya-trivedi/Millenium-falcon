"""Rebuild v2 preprocessing and its shared split without touching Plan 1 run paths."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import polars as pl

from plan1.normalize import NORMALIZE_VERSION, normalize_records
from plan1.splits import build_manifest, stable_unit, stratified_take
from plan1.translit import INDIC_RE, build_dictionary, make_converter
from .artifacts import (contract, fingerprint, parquet, partition_path, read_json,
                        records, sha256, source_hashes, versions, write_json)

FRESH_AUDIT_SALT = "amc26-plan3-fresh-audit-v1"


def fresh_audit(manifest: pl.DataFrame, n: int) -> pl.DataFrame:
    pool = manifest.filter((pl.col("role") == "audit") &
                           (pl.col("sample").fill_null("") != "audit_panel"))
    if not 0 < n <= pool.height:
        raise ValueError(f"Requested fresh Audit {n}; eligible unused Audit owners: {pool.height}")
    pool = pool.with_columns(sample_u=pl.Series(stable_unit(pool["s1_id"].to_list(), FRESH_AUDIT_SALT)))
    chosen = stratified_take(pool, n).drop("sample_u")
    if chosen.height != n or chosen["s1_id"].n_unique() != n:
        raise ValueError("Audit sample cardinality failed")
    return chosen


def prepare(dataset: Path, out: Path, audit_size: int = 50_000, include_test: bool = False) -> None:
    start = time.monotonic()
    dataset = dataset.resolve()
    baseline_hashes = source_hashes("plan1")
    paths = [dataset / "train" / f"train_source{s}.tsv" for s in (1, 2, 3)]
    paths.append(dataset / "train" / "train_ground_truth.tsv")
    if include_test:
        paths += [dataset / "test" / f"test_source{s}.tsv" for s in (1, 2, 3)]
    print("Fingerprinting raw inputs...", flush=True)
    inputs = {str(p.relative_to(dataset)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}
    identity = {"format": 1, "inputs": inputs, "baseline_hashes": baseline_hashes,
                "prepare_code": sha256(Path(__file__)), "normalize_version": NORMALIZE_VERSION,
                "fresh_audit_salt": FRESH_AUDIT_SALT, "audit_size": audit_size,
                "dependencies": versions(), "include_test": include_test}
    contract(out / "contract.json", identity)
    if (out / "prepared.json").exists():
        print(f"Prepared snapshot already complete: {out}", flush=True)
        return

    labels = records(dataset / "train/train_ground_truth.tsv").collect(engine="streaming")
    if labels.columns != ["source1_entity_id", "matched_entity_ids"]:
        raise ValueError("Unexpected label columns")
    if labels["source1_entity_id"].n_unique() != labels.height:
        raise ValueError("Duplicate ground truth S1")
    truth = (labels.select(s1_id="source1_entity_id", target_id=pl.col("matched_entity_ids").str.split(","))
             .explode("target_id").filter(pl.col("target_id").is_not_null() & (pl.col("target_id") != "")))
    if truth.unique(["s1_id", "target_id"]).height != truth.height:
        raise ValueError("Duplicate true pair")
    if truth["target_id"].n_unique() != truth.height:
        raise ValueError("Labels contradict one-owner-per-target")
    s1 = records(dataset / "train/train_source1.tsv").collect(engine="streaming")
    if s1["entity_id"].n_unique() != s1.height:
        raise ValueError("Duplicate S1 ID")
    if (s1.height != labels.height or
            s1.select(s1_id="entity_id").join(labels.select(s1_id="source1_entity_id"), on="s1_id", how="anti").height):
        raise ValueError("Truth roster does not equal S1 roster")
    parquet(out / "truth.parquet", truth)
    manifest = build_manifest(s1, truth)
    parquet(out / "manifest.parquet", manifest)
    audit = fresh_audit(manifest, audit_size)
    parquet(out / "fresh_audit.parquet", audit)
    print(f"Split: {manifest.height:,} owners; old Audit 20,000 excluded from new {audit.height:,}", flush=True)
    del labels

    if not (out / "translit.json").exists():
        names = [s1.select("entity_id", "business_name").with_columns(source=pl.lit("S1"))]
        for source in ("S2", "S3"):
            names.append(records(dataset / "train" / f"train_source{source[1]}.tsv")
                         .filter(pl.col("business_name").str.contains(INDIC_RE))
                         .select("entity_id", "business_name").with_columns(source=pl.lit(source))
                         .collect(engine="streaming"))
        mapping, stats = build_dictionary(pl.concat(names), truth, manifest.filter(pl.col("role") == "fit")["s1_id"])
        write_json(out / "translit.json", {"mapping": mapping, "stats": stats,
                                         "training_role": "fit", "manifest_sha256": sha256(out / "manifest.parquet")})
        del names
    dictionary = read_json(out / "translit.json")
    print(f"Fit-only dictionary: {len(dictionary['mapping']):,} words", flush=True)
    del s1, manifest, audit
    gc.collect()
    converter = make_converter(dictionary["mapping"])
    partitions = []
    for split in (("train", "test") if include_test else ("train",)):
        for source in ("S1", "S2", "S3"):
            raw = records(dataset / split / f"{split}_source{source[1]}.tsv").collect(engine="streaming")
            if raw.columns != ["entity_id", "business_name", "business_address", "country"]:
                raise ValueError("Unexpected record columns")
            if raw["entity_id"].n_unique() != raw.height or not raw["entity_id"].str.starts_with(source + "-").all():
                raise ValueError("Invalid source IDs")
            if raw["country"].is_null().any() or (raw["country"] == "").any():
                raise ValueError("Missing country")
            raw = raw.with_columns(source=pl.lit(source))
            if split == "train" and source != "S1":
                # Full-pool label/country check; orphan targets deliberately stay in the index.
                labelled = truth.filter(pl.col("target_id").str.starts_with(source + "-"))
                if labelled.join(raw.select(target_id="entity_id"), on="target_id", how="anti").height:
                    raise ValueError("Truth target missing from source")
                countries = pl.read_parquet(out / "manifest.parquet", columns=["s1_id", "country"])
                cross = labelled.join(countries, on="s1_id").join(
                    raw.select(target_id="entity_id", target_country="country"), on="target_id")
                if cross.filter(pl.col("country") != pl.col("target_country")).height:
                    raise ValueError("Same-country blocking would miss a labeled match")
                del cross, labelled, countries
            for country in sorted(raw["country"].unique().to_list()):
                path = partition_path(out, split, country, source)
                n = raw.filter(pl.col("country") == country).height
                if not path.exists():
                    normalized = normalize_records(raw.filter(pl.col("country") == country), converter)
                    parquet(path, normalized)
                    del normalized
                partitions.append({"split": split, "country": country, "source": source, "rows": n,
                                   "file": str(path.relative_to(out)).replace("\\", "/"), "sha256": sha256(path)})
                print(f"Normalized {split}/{country}/{source}: {n:,}", flush=True)
            del raw
            gc.collect()
    snapshot = {"identity": fingerprint(identity), "baseline_hashes": baseline_hashes, "inputs": inputs,
                "dataset": str(dataset), "partitions": partitions,
                "manifest_sha256": sha256(out / "manifest.parquet"),
                "fresh_audit_sha256": sha256(out / "fresh_audit.parquet"),
                "dictionary_sha256": sha256(out / "translit.json"), "dependencies": versions(),
                "elapsed_seconds": time.monotonic() - start}
    write_json(out / "prepared.json", snapshot)
    print(f"Prepared in {snapshot['elapsed_seconds']:.1f}s: {out}", flush=True)
