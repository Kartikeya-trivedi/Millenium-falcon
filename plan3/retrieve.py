"""Candidate unions and reproducible query samples for training or full-pool probes."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import polars as pl

from plan1.splits import fit_sample_ids, stable_unit, stratified_take
from .artifacts import (contract, fingerprint, load_prepared, parquet, partition_path,
                        read_json, sha256, source_hashes, write_json)
from .evaluate import retrieval_report
from .index import CHANNELS, SparseIndex

DEFAULT_BUDGETS = {"word": 200, "missing": 100, "name": 50, "char": 50}
INITIAL_BUDGETS = {"word": 200, "missing": 100}
PROFILES = {
    "probe": {"fit": 0, "tune": 2000, "select": 0, "development": 0},
    "pilot": {"fit": 10_000, "support": 3000, "tune": 2000, "select": 2000, "development": 2000},
    "full": {"fit": 300_000, "support": 50_000, "tune": 20_000, "select": 20_000, "development": 20_000},
}


def samples(prepared: Path, sizes: dict) -> pl.DataFrame:
    manifest = pl.read_parquet(prepared / "manifest.parquet")
    parts = []
    pool_names = {"tune": "tune_sample", "select": "c_select_sample", "development": "audit_panel"}
    for name, size in sizes.items():
        if not size:
            continue
        if name == "fit":
            ids = fit_sample_ids(manifest, size)
        else:
            # C-prob was unused by v2. It is explicitly repurposed as second-stage
            # training, NEVER final probability calibration, in this Plan 3 run.
            # Neither its labels nor owners entered the Fit-only dictionary or M0.
            pool = manifest.filter(pl.col("pool") == "c_prob") if name == "support" else manifest.filter(pl.col("sample") == pool_names[name])
            if not 0 < size <= pool.height:
                raise ValueError(f"{name} has {pool.height} available owners, requested {size}")
            pool = pool.with_columns(sample_u=pl.Series(stable_unit(pool["s1_id"].to_list(), "amc26-p3-panel-v1-" + name)))
            ids = stratified_take(pool, size)["s1_id"]
        if ids.len() != size:
            raise ValueError(f"{name} sample size mismatch")
        parts.append(pl.DataFrame({"s1_id": ids, "sample": pl.repeat(name, size, eager=True)}))
    if not parts:
        raise ValueError("Empty query population")
    out = pl.concat(parts)
    if out["s1_id"].n_unique() != out.height:
        raise ValueError("Samples overlap")
    protected = pl.read_parquet(prepared / "fresh_audit.parquet", columns=["s1_id"])
    if out.join(protected, on="s1_id", how="inner").height:
        raise ValueError("Fresh Audit leaked into development")
    return out


def union_channels(parts: dict[str, pl.DataFrame]) -> pl.DataFrame:
    schema = {"s1_id": pl.String, "target_id": pl.String}
    schema.update({f"{ch}_{suffix}": typ for ch in CHANNELS for suffix, typ in (("score", pl.Float32), ("rank", pl.Int32))})
    rows = []
    for channel, part in parts.items():
        if not part.height:
            continue
        row = part.select("s1_id", "target_id", pl.col("score").alias(channel + "_score"),
                          pl.col("rank").alias(channel + "_rank"))
        row = row.with_columns([pl.lit(None, typ).alias(col) for col, typ in schema.items() if col not in row.columns])
        rows.append(row.select(list(schema)))
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.concat(rows).group_by("s1_id", "target_id").agg(
        *[pl.col(c).max().alias(c) for c in schema if c not in ("s1_id", "target_id")]).sort("s1_id", "target_id")


def index_for(prepared: Path, indexes: Path, split: str, country: str, source: str, channel: str,
              shard_size: int = 50_000) -> SparseIndex:
    path = partition_path(prepared, split, country, source)
    if not path.exists():
        raise ValueError(f"No prepared partition: {split}/{country}/{source}")
    index_path = indexes / split / f"{fingerprint(country)[:16]}-{source}" / channel
    return SparseIndex.build(index_path, pl.read_parquet(path), channel, sha256(path), shard_size=shard_size)


def retrieve(prepared: Path, indexes: Path, run: Path, profile: str = "probe",
             budgets: dict | None = None, threads: int = 4, query_batch: int = 1000,
             shard_size: int = 50_000, sizes: dict | None = None) -> None:
    meta = load_prepared(prepared)
    budgets = INITIAL_BUDGETS.copy() if budgets is None else budgets
    if set(budgets) - set(CHANNELS) or budgets.get("word", 0) < 50 or any(v <= 0 for v in budgets.values()):
        raise ValueError("Valid positive channel budgets and word >=50 required")
    sizes = PROFILES[profile] if sizes is None else sizes
    sample = samples(prepared, sizes)
    contract(run / "retrieval_contract.json", {
        "prepared_identity": meta["identity"], "prepared_manifest": meta["manifest_sha256"],
        "query_ids": fingerprint(sample.sort("sample", "s1_id").to_dicts()), "sizes": sizes,
        "budgets": budgets, "query_batch": query_batch, "shard_size": shard_size,
        "index_code": sha256(Path(__file__).with_name("index.py")),
        "retrieve_code": sha256(Path(__file__)), "threads": threads})
    parquet(run / "samples.parquet", sample)
    qs = []
    for p in meta["partitions"]:
        if p["split"] == "train" and p["source"] == "S1":
            qs.append(pl.read_parquet(prepared / p["file"]).join(
                sample.rename({"s1_id": "entity_id"}), on="entity_id", how="inner"))
    queries = pl.concat(qs).sort("entity_id")
    if queries.height != sample.height:
        raise ValueError("Not all selected queries exist")
    parquet(run / "queries.parquet", queries)
    timings = []
    for country in sorted(queries["country"].unique().to_list()):
        q = queries.filter(pl.col("country") == country)
        for source in ("S2", "S3"):
            part_key = f"{fingerprint(country)[:16]}-{source}"
            # Search/cache channels independently; additional K experiments do not destroy prior results.
            for channel, k in budgets.items():
                dest = run / "channels" / f"{part_key}-{channel}.parquet"
                if dest.exists():
                    continue
                start = time.monotonic()
                print(f"Build/search {country}/{source}/{channel}, K={k}, queries={q.height:,}", flush=True)
                idx = index_for(prepared, indexes, "train", country, source, channel, shard_size)
                built = time.monotonic()
                blocks = [idx.query(qb, k, threads) for qb in q.iter_slices(query_batch)]
                result = pl.concat(blocks)
                parquet(dest, result)
                timing = {"country": country, "source": source, "channel": channel, "targets": idx.ids.len(),
                          "build_or_load_s": built - start, "search_s": time.monotonic() - built, "pairs": result.height}
                timings.append(timing)
                write_json(dest.with_suffix(".json"), timing)
                print(f"  {channel}: {result.height:,} proposals in {time.monotonic()-start:.1f}s", flush=True)
                del idx, blocks, result
                gc.collect()
            dest = run / "candidates" / f"{part_key}.parquet"
            if not dest.exists():
                channels = {ch: pl.read_parquet(run / "channels" / f"{part_key}-{ch}.parquet") for ch in budgets}
                merged = union_channels(channels)
                idx = index_for(prepared, indexes, "train", country, source, "word", shard_size)
                words = idx.pair_scores(q, merged)
                merged = (merged.join(words, on=["s1_id", "target_id"], how="left")
                          .with_columns(rank=pl.col("word_rank"), source=pl.lit(source), country=pl.lit(country))
                          .join(sample, on="s1_id", how="left"))
                parquet(dest, merged)
                del idx, merged, channels, words
                gc.collect()
    if timings:
        parquet(run / "retrieval_timings.parquet", pl.DataFrame(timings))
    report_run(prepared, run)
    write_json(run / "retrieval_complete.json", {"status": "retrieved", "sizes": sizes, "budgets": budgets})


def report_run(prepared: Path, run: Path) -> None:
    sample = pl.read_parquet(run / "samples.parquet")
    truth = pl.read_parquet(prepared / "truth.parquet").join(sample.select("s1_id"), on="s1_id", how="semi")
    # Metadata join only; all targets were already eligible during retrieval.
    info = []
    meta = load_prepared(prepared)
    wanted = truth.select(entity_id="target_id").unique()
    for p in meta["partitions"]:
        if p["split"] == "train" and p["source"] != "S1":
            info.append(pl.scan_parquet(prepared / p["file"]).join(wanted.lazy(), on="entity_id", how="semi")
                        .select(target_id="entity_id", source="source", country="country",
                                name_nonlatin="name_nonlatin", addr_missing="addr_missing",
                                target_name="name_cons", target_address="addr_norm").collect(engine="streaming"))
    info = pl.concat(info)
    report = {}
    for name in sample["sample"].unique().sort():
        roster = sample.filter(pl.col("sample") == name)["s1_id"]
        cands = pl.scan_parquet(str(run / "candidates/*.parquet")).filter(pl.col("sample") == name).collect(engine="streaming")
        report[name], misses = retrieval_report(cands, truth, roster, info)
        query_info = pl.read_parquet(run / "queries.parquet").select(
            s1_id="entity_id", query_name="name_cons", query_address="addr_norm")
        parquet(run / f"misses_{name}.parquet", misses.join(query_info, on="s1_id", how="left"))
        print(f"{name}: " + str(report[name]["union"]), flush=True)
    write_json(run / "retrieval_report.json", report)
