"""Measure a single bounded neighbor pass from predicted seeds; never emit matches."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import polars as pl

from .artifacts import ROOT, contract, fingerprint, load_prepared, parquet, read_json, sha256, write_json
from .budget import keep_candidates
from .evaluate import score
from .index import SparseIndex


def select_seeds(scored: pl.DataFrame, threshold: float, limit=2):
    # Labels are deliberately excluded before ordering or deduplicating seeds.
    fields = ["s1_id", "target_id", "country", "source", "p", "t_name", "t_address"]
    return (scored.select(fields).filter(pl.col("p") >= threshold)
            .filter((pl.col("t_name") != "") | (pl.col("t_address") != ""))
            .sort(["s1_id", "p", "target_id"], descending=[False, True, False])
            .unique(["s1_id", "t_name", "t_address"], keep="first", maintain_order=True)
            .group_by("s1_id", maintain_order=True).head(limit)
            .with_row_index("seed_index").with_columns(seed_id=pl.col("seed_index").cast(pl.String)))


def probe(prepared: Path, indexes: Path, run: Path, threads=4):
    meta = load_prepared(prepared)
    direct = read_json(run / "direct_meta.json")
    if direct["sample"] != "fit":
        raise ValueError("Require Fit-only direct model")
    threshold = read_json(run / "seed_selection.json")["threshold"]
    sample = pl.read_parquet(run / "samples.parquet").filter(pl.col("sample") == "tune")
    scored = (pl.scan_parquet(str(run / "direct_scores/*.parquet")).filter(pl.col("sample") == "tune")
              .select("s1_id", "target_id", "country", "source", "p", "t_name", "t_address").collect())
    seeds = select_seeds(scored, threshold)
    out = run / "onehop_probe"
    feature = read_json(run / "feature_contract.json")
    index_files = {f"{c}/{s}": indexes / "train" / f"{fingerprint(c)[:16]}-{s}" / "word/index.json"
                   for c in seeds["country"].unique() for s in ("S2", "S3")}
    contract(out / "contract.json", {"prepared_identity": meta["identity"], "sample": "tune",
        "seeds_sha": fingerprint(seeds.to_dicts()), "seed_threshold": threshold,
        "parent_model_sha": direct["model_sha256"], "source_sha": sha256(Path(__file__)),
        "index_specs": {k: sha256(v) for k, v in index_files.items()},
        "seeds_per_owner": 2, "neighbors_per_seed_source": 20, "max_new_per_source": 40,
        "baseline_feature_contract_sha": sha256(run / "feature_contract.json")})
    parquet(out / "seeds.parquet", seeds)
    base = keep_candidates(pl.scan_parquet(str(run / "candidates/*.parquet"))
                           .filter(pl.col("sample") == "tune").collect(),
                           feature["word_k"], feature["missing_k"])
    blocks = []
    for country in sorted(seeds["country"].unique()):
        seed = seeds.filter(pl.col("country") == country)
        q = seed.select(entity_id="seed_id", retrieval_text=pl.col("t_name")+" "+pl.col("t_address"))
        mapping = seed.select(seed_id="seed_id", s1_id="s1_id", seed_target="target_id", seed_p="p")
        for source in ("S2", "S3"):
            path = out / f"{fingerprint(country)[:16]}-{source}.parquet"
            if path.exists():
                blocks.append(pl.read_parquet(path))
                continue
            start = time.monotonic()
            idx = SparseIndex(index_files[f"{country}/{source}"].parent)
            found = pl.concat([idx.query(b, 20, threads) for b in q.iter_slices(1000)])
            found = (found.rename({"s1_id": "seed_id", "score": "seed_cosine", "rank": "seed_rank"})
                     .join(mapping, on="seed_id").with_columns(source=pl.lit(source), country=pl.lit(country)))
            found = found.join(base.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="anti")
            # Distinct seeds can propose the same neighbor. Keep its strongest edge.
            found = (found.sort(["s1_id", "seed_cosine", "seed_p", "target_id"], descending=[False, True, True, False])
                     .unique(["s1_id", "target_id"], keep="first", maintain_order=True)
                     .group_by("s1_id", maintain_order=True).head(40))
            parquet(path, found)
            blocks.append(found)
            print(f"One hop {country}/{source}: {q.height:,} seeds, {found.height:,} new pairs, "
                  f"{time.monotonic()-start:.1f}s", flush=True)
            del idx, found
            gc.collect()
    added = pl.concat(blocks)
    joined = pl.concat([base.select("s1_id", "target_id"), added.select("s1_id", "target_id")]).unique()
    truth = pl.read_parquet(prepared / "truth.parquet").join(sample.select("s1_id"), on="s1_id", how="semi")
    report = {"population": "Tune owners, full train target indexes", "probe_only": True,
              "seed_owners": seeds["s1_id"].n_unique(), "seed_records": seeds.height,
              "added_candidates": added.height, "before": {}, "after": {}}
    for name, frame in (("before", base), ("after", joined)):
        found = truth.join(frame.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="semi")
        counts = sample.join(frame.group_by("s1_id").len(name="n"), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
        report[name] = {"true_links": truth.height, "misses": truth.height-found.height,
            "candidate_recall": found.height/truth.height, "oracle_macro_f05": score(found, truth, sample["s1_id"])["macro_f05"],
            "mean_candidates": counts["n"].mean(), "p95_candidates": counts["n"].quantile(.95, interpolation="nearest"),
            "max_candidates": counts["n"].max()}
    recovered = truth.join(added, on=["s1_id", "target_id"], how="inner")
    parquet(out / "recovered_true_links.parquet", recovered)
    parquet(out / "added_candidates.parquet", added)
    report["recovered_links"] = recovered.height
    report["interpretation"] = ("Retrieval upper bound only. New candidates were not classified. "
        "A changed-candidate matcher must be trained on independently generated expansion features before promoting this pass.")
    write_json(out / "report.json", report)
    print(report, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--indexes", type=Path, default=ROOT / "work/plan3/indexes")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    probe(args.prepared, args.indexes, args.run, args.threads)


if __name__ == "__main__":
    main()
