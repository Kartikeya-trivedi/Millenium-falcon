"""Stream frozen-model predictions with the same proposals and features as training.

This module does not select a model, read labels or report accuracy. Explicit train
query lists support later evaluation; test inference uses the complete test roster.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .artifacts import (contract, fingerprint, load_prepared, parquet, read_json,
                        sha256, write_json)
from .budget import keep_candidates
from .features import FEATURES, rich_features
from .export import inference_manifest, iter_score_shards
from .retrieve import index_for, union_channels
from .support import SUPPORT_FEATURES, support_features
from .train import _matrix

KEYS = ["s1_id", "target_id"]
SCORE_SCHEMA = {"s1_id": pl.String, "target_id": pl.String, "source": pl.String,
                "country": pl.String, "p": pl.Float64, "p_direct": pl.Float64,
                "score": pl.Float32, "rank": pl.Int32}


def load_models(prepared: Path, run: Path, model_name: str):
    if model_name not in ("direct", "support"):
        raise ValueError("Choose a measured direct or support model")
    meta = load_prepared(prepared)
    if sha256(prepared / "translit.json") != meta["dictionary_sha256"]:
        raise ValueError("Saved transliteration dictionary was modified")
    feature = read_json(run / "feature_contract.json")
    retrieval = read_json(run / "retrieval_contract.json")
    if feature["prepared_identity"] != meta["identity"] or retrieval["prepared_identity"] != meta["identity"]:
        raise ValueError("Model preprocessing differs from this prepared snapshot")
    if feature["features"] != FEATURES or feature["features_code"] != sha256(Path(__file__).with_name("features.py")):
        raise ValueError("Direct feature implementation changed since training")
    if retrieval["index_code"] != sha256(Path(__file__).with_name("index.py")):
        raise ValueError("Retrieval implementation changed since training")
    if retrieval["retrieve_code"] != sha256(Path(__file__).with_name("retrieve.py")):
        raise ValueError("Candidate union implementation changed since training")
    if set(retrieval["budgets"]) != {"word", "missing"}:
        raise ValueError("This inference path supports the measured word/missing proposal union")
    direct_meta = read_json(run / "direct_meta.json")
    if direct_meta["feature_contract"] != sha256(run / "feature_contract.json"):
        raise ValueError("Direct model feature contract mismatch")
    if direct_meta["model_sha256"] != sha256(run / "direct.txt"):
        raise ValueError("Direct model hash mismatch")
    direct = lgb.Booster(model_file=str(run / "direct.txt"))
    if direct.feature_name() != FEATURES:
        raise ValueError("Unexpected direct model feature order")
    selected, seed = direct, None
    hashes = {"direct": sha256(run / "direct.txt")}
    if model_name == "support":
        support = read_json(run / "support_features_contract.json")
        if support["parent_model_sha256"] != hashes["direct"]:
            raise ValueError("Support model belongs to a different direct model")
        if support["support_code"] != sha256(Path(__file__).with_name("support.py")):
            raise ValueError("Support feature implementation changed since training")
        hashes["support"] = sha256(run / "support.txt")
        support_meta = read_json(run / "support_meta.json")
        if support_meta["feature_contract"] != sha256(run / "feature_contract.json"):
            raise ValueError("Support model feature contract mismatch")
        if hashes["support"] != support_meta["model_sha256"]:
            raise ValueError("Support model hash mismatch")
        selected = lgb.Booster(model_file=str(run / "support.txt"))
        if selected.feature_name() != FEATURES + SUPPORT_FEATURES:
            raise ValueError("Unexpected support model feature order")
        seed = support["seed_threshold"]
    decision = read_json(run / (model_name + "_decision.json"))
    if decision["selected_on"] != "select" or not 0 <= decision["threshold"] <= 1.000001:
        raise ValueError("A decision selected on the separate selection panel is required")
    frozen = {"model": model_name, "model_sha256": hashes, "seed_threshold": seed,
              "decision": decision, "proposal_budgets": retrieval["budgets"],
              "shard_size": retrieval["shard_size"],
              "word_k": feature["word_k"], "missing_k": feature["missing_k"],
              "prepared_identity": meta["identity"], "dictionary_sha256": meta["dictionary_sha256"],
              "feature_contract_sha256": sha256(run / "feature_contract.json"),
              "retrieval_contract_sha256": sha256(run / "retrieval_contract.json")}
    return frozen, direct, selected


def predict_candidates(candidates, queries, targets, idfs, direct, support=None, seed=None, threads=4):
    """Score a complete collection of owner groups; never trim by score first."""
    if not candidates.height:
        return pl.DataFrame(schema=SCORE_SCHEMA)
    f = rich_features(candidates, queries, targets, idfs)
    direct_p = direct.predict(_matrix(f, FEATURES), num_threads=threads)
    f = f.with_columns(p=pl.Series(direct_p))
    final_p = direct_p
    if support is not None:
        if seed is None:
            raise ValueError("Support inference requires the saved seed threshold")
        f = f.join(targets.select(target_id="entity_id", t_name="name_red", t_address="addr_norm",
                                  t_numbers="addr_nums"), on="target_id", how="left", maintain_order="left")
        f = f.join(support_features(f, seed), on=KEYS, how="left", maintain_order="left")
        final_p = support.predict(_matrix(f, FEATURES + SUPPORT_FEATURES), num_threads=threads)
    for values in (direct_p, final_p):
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Invalid model probabilities")
    return candidates.select(*KEYS, "source", "country", "score", "rank").with_columns(
        p=pl.Series(final_p, dtype=pl.Float64), p_direct=pl.Series(direct_p, dtype=pl.Float64)
    ).select(list(SCORE_SCHEMA))


def retrieve_batch(queries, indexes, budgets, word_k, missing_k, country, threads):
    parts = []
    for source, channels in indexes.items():
        proposals = {name: idx.query(queries, budgets[name], threads) for name, idx in channels.items()}
        candidates = keep_candidates(union_channels(proposals), word_k, missing_k)
        # Retain proposal ranks beyond the final cut when another channel admits
        # the pair. Those ranks/channel-presence values were present in training.
        actual = channels["word"].pair_scores(queries, candidates)
        parts.append(candidates.join(actual, on=KEYS, how="left").with_columns(
            rank=pl.col("word_rank"), source=pl.lit(source), country=pl.lit(country)))
    return pl.concat(parts).sort(*KEYS)


def committed_batch(path: Path, query_ids: pl.Series):
    """Never adopt uncommitted or modified predictions merely because a file exists."""
    marker = path.with_suffix(".json")
    if not path.exists() or not marker.exists():
        return False
    saved = read_json(marker)
    if saved["query_ids"] != fingerprint(query_ids.to_list()) or saved["sha256"] != sha256(path):
        raise ValueError(f"Inference checkpoint changed: {path.name}")
    if saved["rows"] != pl.scan_parquet(path).select(pl.len()).collect().item():
        raise ValueError(f"Inference checkpoint row count changed: {path.name}")
    return True


def infer(prepared: Path, views: Path, indexes: Path, model_run: Path, out: Path,
          model_name="support", split="test", query_ids: Path | None = None,
          threads=4, query_batch=1000, feature_batch=250):
    if query_batch < 1 or feature_batch < 1 or threads < 1:
        raise ValueError("Batch sizes and threads must be positive")
    if split not in ("train", "test"):
        raise ValueError("Choose train or test inference")
    if split == "train" and query_ids is None:
        raise ValueError("Train inference requires an explicit query list")
    if split == "test" and query_ids is not None:
        raise ValueError("Test inference must cover the complete test roster")
    frozen, direct, chosen = load_models(prepared, model_run, model_name)
    view_contract = read_json(views / f"{split}_contract.json")
    if (view_contract["prepared_identity"] != frozen["prepared_identity"] or
            view_contract["views_code"] != sha256(Path(__file__).with_name("views.py"))):
        raise ValueError("Inference views are incompatible with the saved model")
    # The training view definition must match too, even when test has other countries.
    train_view = read_json(views / "train_contract.json")
    feature_contract = read_json(model_run / "feature_contract.json")
    if sha256(views / "train_contract.json") != feature_contract["views_contract"]:
        raise ValueError("Training view contract mismatch")
    if train_view["views_code"] != view_contract["views_code"]:
        raise ValueError("Train and inference view definitions differ")
    meta = load_prepared(prepared)
    queries = pl.concat([pl.read_parquet(prepared / p["file"]) for p in meta["partitions"]
                         if p["split"] == split and p["source"] == "S1"])
    if query_ids is not None:
        wanted = pl.read_parquet(query_ids).select("s1_id")
        if wanted["s1_id"].n_unique() != wanted.height:
            raise ValueError("Duplicate query IDs")
        queries = queries.join(wanted.rename({"s1_id": "entity_id"}), on="entity_id", how="semi")
        if queries.height != wanted.height:
            raise ValueError("Requested query is outside the prepared split")
    queries = queries.sort("entity_id")
    if not queries.height:
        raise ValueError("Empty inference population")
    spec = {"frozen": frozen, "split": split, "queries": queries.height,
            "raw_inputs": {name: value for name, value in meta["inputs"].items()
                           if name.startswith(split + "/") and "source" in name},
            "query_ids": fingerprint(queries["entity_id"].to_list()),
            "query_batch": query_batch, "feature_batch": feature_batch,
            "threads": threads, "inference_code": sha256(Path(__file__)),
            "view_contract_sha256": sha256(views / f"{split}_contract.json")}
    contract(out / "inference_contract.json", spec)
    if (out / "inference_complete.json").exists():
        _, complete, _ = inference_manifest(out)
        for _, _ in iter_score_shards(out, complete):
            pass
        print(f"Verified completed inference: {out}", flush=True)
        return
    parquet(out / "roster.parquet", queries.select(s1_id="entity_id", country="country"))
    freq = pl.read_parquet(views / f"{split}_name_frequency.parquet")
    expected = []
    for country in sorted(queries["country"].unique().to_list()):
        q = queries.filter(pl.col("country") == country)
        paths = [(start, out / "scores" / f"{fingerprint(country)[:16]}-{start:08d}.parquet")
                 for start in range(0, q.height, query_batch)]
        expected.extend(p.name for _, p in paths)
        ready = {p.name: committed_batch(p, q.slice(start, query_batch)["entity_id"]) for start, p in paths}
        if all(ready.values()):
            continue
        records, search, idfs = {}, {}, {}
        for part in meta["partitions"]:
            if part["split"] != split or part["country"] != country:
                continue
            frame = pl.read_parquet(prepared / part["file"])
            if part["source"] == "S1":
                frame = frame.join(q.select("entity_id"), on="entity_id", how="semi")
            records[part["source"]] = (frame.join(pl.read_parquet(views / part["file"]), on="entity_id")
                                       .join(freq, on=["name_red", "country"], how="left"))
            if part["source"] != "S1":
                source = part["source"]
                search[source] = {name: index_for(prepared, indexes, split, country, source, name, frozen["shard_size"])
                                  for name in frozen["proposal_budgets"]}
                word = search[source]["word"]
                idfs[source] = ({term: float(word.idf[i]) for term, i in word.vocabulary.items()}, word.ids.len())
        if not search:
            raise ValueError(f"No targets for {country}")
        targets = pl.concat([frame for source, frame in records.items() if source != "S1"])
        target_rows = {key: i for i, key in enumerate(targets["entity_id"])}
        query_rows = {key: i for i, key in enumerate(records["S1"]["entity_id"])}
        for start, destination in paths:
            if ready[destination.name]:
                continue
            qb = q.slice(start, query_batch)
            candidates = retrieve_batch(qb, search, frozen["proposal_budgets"], frozen["word_k"],
                                        frozen["missing_k"], country, threads)
            scored = []
            for raw in qb.iter_slices(feature_batch):
                c = candidates.filter(pl.col("s1_id").is_in(raw["entity_id"].implode()))
                qr = records["S1"][pl.Series([query_rows[key] for key in raw["entity_id"]], dtype=pl.UInt32)]
                rows = [target_rows[key] for key in c["target_id"].unique(maintain_order=True)]
                tr = targets[pl.Series(rows, dtype=pl.UInt32)]
                scored.append(predict_candidates(c, qr, tr, idfs, direct,
                    chosen if model_name == "support" else None, frozen["seed_threshold"], threads))
            result = pl.concat(scored)
            if result.height != candidates.height or result.select(KEYS).n_unique() != result.height:
                raise ValueError("Scoring lost or duplicated candidate pairs")
            parquet(destination, result)
            write_json(destination.with_suffix(".json"), {"query_ids": fingerprint(qb["entity_id"].to_list()),
                "sha256": sha256(destination), "rows": result.height})
            print(f"Scored {split}/{country}: {min(start + query_batch, q.height):,}/{q.height:,} owners; "
                  f"{result.height:,} pairs in this batch", flush=True)
        del records, search, idfs, targets, target_rows, query_rows
        gc.collect()
    found = sorted(p.name for p in (out / "scores").glob("*.parquet"))
    if found != sorted(expected):
        raise ValueError("Inference has missing or unexpected score shards")
    write_json(out / "inference_complete.json", {"contract_sha256": sha256(out / "inference_contract.json"),
        "shards": expected, "queries": queries.height,
        "roster_sha256": sha256(out / "roster.parquet"),
        "score_shards": {name: {"sha256": sha256(out / "scores" / name),
            "rows": pl.scan_parquet(out / "scores" / name).select(pl.len()).collect().item()}
            for name in expected}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "views", "indexes", "model-run", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--model", choices=("direct", "support"), default="support")
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--queries", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--query-batch", type=int, default=1000)
    p.add_argument("--feature-batch", type=int, default=250)
    a = p.parse_args()
    infer(a.prepared, a.views, a.indexes, a.model_run, a.out, a.model, a.split, a.queries,
          a.threads, a.query_batch, a.feature_batch)


if __name__ == "__main__":
    main()
