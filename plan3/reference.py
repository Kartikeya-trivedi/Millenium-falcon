"""Run the unchanged upstream v2 retriever as the local reference."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import polars as pl

from plan1.retrieve import TargetIndex, add_rank
from plan1.splits import fit_sample_ids
from .artifacts import contract, fingerprint, load_prepared, parquet, read_json, write_json
from .retrieve import report_run, union_channels


def retrieve_reference(prepared: Path, run: Path, threads: int = 4, fit_size: int = 300_000,
                       panel_size: int = 20_000) -> None:
    meta = load_prepared(prepared)
    manifest = pl.read_parquet(prepared / "manifest.parquet")
    sample_parts = [pl.DataFrame({"s1_id": fit_sample_ids(manifest, fit_size)}).with_columns(sample=pl.lit("fit"))]
    for label, upstream in (("tune", "tune_sample"), ("select", "c_select_sample"), ("development", "audit_panel")):
        ids = manifest.filter(pl.col("sample") == upstream).sort("s1_id").head(panel_size)["s1_id"]
        if ids.len() != panel_size:
            raise ValueError("Reference panel size exceeds upstream sample")
        sample_parts.append(pl.DataFrame({"s1_id": ids}).with_columns(sample=pl.lit(label)))
    samples = pl.concat(sample_parts)
    if samples["s1_id"].n_unique() != samples.height or fit_size <= 0 or panel_size <= 0:
        raise ValueError("Invalid reference sample")
    identity = {"prepared_identity": meta["identity"], "upstream": meta["baseline_hashes"],
                "sizes": {"fit": fit_size, "tune": panel_size, "select": panel_size, "development": panel_size},
                "budgets": {"word": 50}, "threads": threads, "reference": True,
                "full_v2_population": fit_size == 300_000 and panel_size == 20_000,
                "query_ids": fingerprint(samples.sort("sample", "s1_id").to_dicts())}
    contract(run / "retrieval_contract.json", identity)
    parquet(run / "samples.parquet", samples)
    qs = []
    for p in meta["partitions"]:
        if p["split"] == "train" and p["source"] == "S1":
            qs.append(pl.read_parquet(prepared / p["file"]).join(samples.rename({"s1_id": "entity_id"}), on="entity_id", how="semi"))
    queries = pl.concat(qs)
    parquet(run / "queries.parquet", queries)
    for country in sorted(queries["country"].unique().to_list()):
        q = queries.filter(pl.col("country") == country)
        for source in ("S2", "S3"):
            part_key = f"{fingerprint(country)[:16]}-{source}"
            dest = run / "candidates" / f"{part_key}.parquet"
            if dest.exists():
                continue
            start = time.monotonic()
            p = next(p for p in meta["partitions"] if (p["split"], p["country"], p["source"]) == ("train", country, source))
            print(f"Reference v2 {country}/{source}: {q.height:,} queries, {p['rows']:,} targets", flush=True)
            idx = TargetIndex(pl.read_parquet(prepared / p["file"]))
            blocks = []
            for offset in range(0, q.height, 20_000):
                chunk = run / "reference_chunks" / f"{part_key}-{offset:09d}.parquet"
                if not chunk.exists():
                    result = idx.query(q.slice(offset, 20_000), top_k=50, n_threads=threads).with_columns(source=pl.lit(source))
                    parquet(chunk, add_rank(result))
                blocks.append(chunk)
                print(f"  {country}/{source}: {min(offset+20_000,q.height):,}/{q.height:,}", flush=True)
            del idx
            gc.collect()
            retrieved = pl.concat([pl.read_parquet(p) for p in blocks])
            merged = union_channels({"word": retrieved})
            merged = (merged.with_columns(score=pl.col("word_score"), rank=pl.col("word_rank"),
                                          country=pl.lit(country), source=pl.lit(source))
                      .join(samples, on="s1_id", how="left"))
            parquet(dest, merged)
            write_json(dest.with_suffix(".json"), {"seconds": time.monotonic()-start,
                                                  "targets": p["rows"], "pairs": merged.height})
            del retrieved, merged
            gc.collect()
    report_run(prepared, run)
    write_json(run / "retrieval_complete.json", {"status": "retrieved", **identity})
