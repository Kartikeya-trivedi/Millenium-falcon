"""Score disjoint complete-owner test partitions, then authenticate their union."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import wraps
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import polars as pl

from .artifacts import contract, fingerprint, load_prepared, parquet, read_json, sha256, write_json
from .export import _raw_roster, inference_manifest, iter_score_shards
from .featurize import bounded_process_jobs, initialize_native_threads
from .inference import infer, load_models

_NATIVE_LIMITS = None
# The only earlier executor admitted here differs in thread scheduling only.
_COMPATIBLE_EXECUTOR_HASHES = frozenset({
    "43fabcb232c0afcfeefa19a551d252f43716663e67471375783ab9ad726794c8",
})


def _limit_rapidfuzz(threads):
    from rapidfuzz import process

    original = getattr(process.cpdist, "_plan3_original", process.cpdist)

    @wraps(original)
    def limited(*args, **kwargs):
        if kwargs.get("workers") == -1:
            kwargs["workers"] = threads
        return original(*args, **kwargs)

    limited._plan3_original = original
    process.cpdist = limited


def _timed(function, label):
    original = getattr(function, "_plan3_original", function)

    @wraps(original)
    def timed(*args, **kwargs):
        start = perf_counter()
        result = original(*args, **kwargs)
        print(f"Worker {os.getpid()} {label}: {perf_counter()-start:.3f}s; {result.height:,} rows", flush=True)
        return result

    timed._plan3_original = original
    return timed


def _executor_contract(out: Path, spec: dict):
    """Preserve a known prior contract while recording this actual executor."""
    path = out / "inference_contract.json"
    current = spec["parallel_code"]
    selected = spec
    existed = path.exists()
    if existed:
        prior = read_json(path).get("parallel_code")
        if prior != current:
            if prior not in _COMPATIBLE_EXECUTOR_HASHES:
                raise ValueError("Unknown prior parallel executor hash")
            selected = {**spec, "parallel_code": prior}
    # All other fields remain exactly equal, including every model/data hash.
    contract(path, selected)
    write_json(out / "executors" / f"{uuid4().hex}.json", {
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "executor_sha256": current, "contract_parallel_code": selected["parallel_code"],
        "contract_sha256": sha256(path), "resumed": existed,
        "rapidfuzz_workers_for_minus_one": spec["threads"],
        "workers": spec["workers"], "per_call_timing": True,
    })
    return selected


def partitions(roster: pl.DataFrame, workers: int):
    if workers < 1 or not roster.height or roster["s1_id"].n_unique() != roster.height:
        raise ValueError("Unique nonempty roster and positive worker count required")
    if not roster["s1_id"].equals(roster["s1_id"].sort()):
        raise ValueError("Partition roster must be sorted by owner ID")
    total = roster.height
    count = min(workers, total)
    full = fingerprint(roster["s1_id"].to_list())
    return [{"start": i*total//count, "stop": (i+1)*total//count,
             "full_queries": total, "full_query_ids": full} for i in range(count)]


def _initialize(threads):
    global _NATIVE_LIMITS
    _NATIVE_LIMITS = initialize_native_threads(threads)
    _limit_rapidfuzz(threads)
    from . import inference
    inference.retrieve_batch = _timed(inference.retrieve_batch, "retrieve")
    inference.predict_candidates = _timed(inference.predict_candidates, "features+predict")


def _run_partition(job):
    number, parameters = job
    infer(**parameters)
    return number


def merge_partitions(out: Path, roster: pl.DataFrame, spec: dict):
    """Verify complete, disjoint child populations before publishing completion."""
    expected_parts = partitions(roster, spec["workers"])
    if spec["partitions"] != expected_parts or spec["queries"] != roster.height:
        raise ValueError("Partition plan does not cover the complete roster")
    if spec["query_ids"] != fingerprint(roster["s1_id"].to_list()):
        raise ValueError("Parent roster fingerprint mismatch")
    all_shards, metadata, children = [], {}, {}
    for number, partition in enumerate(expected_parts):
        folder = out / "scores" / f"part-{number:04d}"
        child, complete, child_roster = inference_manifest(folder)
        for key in ("frozen", "raw_inputs", "view_contract_sha256", "inference_code"):
            if child.get(key) != spec[key]:
                raise ValueError(f"Partition {number} differs from frozen parent {key}")
        if child.get("split") != "test" or child.get("partition") != partition:
            raise ValueError("Child is not the declared internal test partition")
        wanted = roster.slice(partition["start"], partition["stop"]-partition["start"])
        if not child_roster.sort("s1_id").equals(wanted):
            raise ValueError("Child roster differs from its complete disjoint owner range")
        owner_countries = dict(child_roster.iter_rows())
        for name, frame in iter_score_shards(folder, complete):
            for owner, country in frame.select("s1_id", "country").unique().iter_rows():
                if owner_countries.get(owner) != country:
                    raise ValueError("Scored owner is outside its assigned partition")
            relative = f"part-{number:04d}/scores/{name}"
            all_shards.append(relative)
            metadata[relative] = complete["score_shards"][name]
        children[str(number)] = {"contract_sha256": sha256(folder / "inference_contract.json"),
                                 "complete_sha256": sha256(folder / "inference_complete.json")}
    actual = {p.relative_to(out / "scores").as_posix() for p in (out / "scores").rglob("*.parquet")}
    if actual != set(all_shards):
        raise ValueError("Unexpected or missing recursive score shard inventory")
    parquet(out / "roster.parquet", roster)
    write_json(out / "inference_complete.json", {
        "contract_sha256": sha256(out / "inference_contract.json"), "queries": roster.height,
        "roster_sha256": sha256(out / "roster.parquet"), "shards": all_shards,
        "score_shards": metadata, "children": children})


def parallel_infer(prepared: Path, views: Path, indexes: Path, model_run: Path, out: Path,
                   model_name="support", split="test", workers=16, threads=4,
                   query_batch=1000, feature_batch=250):
    if split != "test" or workers < 1 or not 1 <= threads <= 4 or query_batch < 1 or feature_batch < 1:
        raise ValueError("Parallel full-test inference needs positive workers/batches and one to four threads")
    meta = load_prepared(prepared)
    frozen, _, _ = load_models(prepared, model_run, model_name)
    raw_inputs = {name: value for name, value in meta["inputs"].items()
                  if name.startswith("test/") and "source" in name}
    if set(raw_inputs) != {f"test/test_source{i}.tsv" for i in (1, 2, 3)}:
        raise ValueError("All raw test sources must be present")
    source = Path(meta["dataset"]) / "test/test_source1.tsv"
    if sha256(source) != raw_inputs["test/test_source1.tsv"]["sha256"]:
        raise ValueError("Raw test owner roster changed")
    roster = _raw_roster(source, "S1").rename({"entity_id": "s1_id"}).sort("s1_id")
    normalized = pl.concat([pl.read_parquet(prepared / p["file"], columns=["entity_id", "country"])
                           for p in meta["partitions"] if p["split"] == "test" and p["source"] == "S1"])
    if not normalized.rename({"entity_id": "s1_id"}).sort("s1_id").equals(roster):
        raise ValueError("Prepared test owners differ from the complete raw roster")
    # Shared indexes must already be committed: never let parallel children
    # race to construct or overwrite the same index files.
    for part in meta["partitions"]:
        if part["split"] == "test" and part["source"] != "S1":
            for channel in frozen["proposal_budgets"]:
                path = indexes / "test" / f"{fingerprint(part['country'])[:16]}-{part['source']}" / channel
                if not (path / "index.json").exists():
                    raise ValueError(f"Complete test indexes are required before parallel inference: {path}")
                saved = read_json(path / "index.json")
                if saved["partition_sha256"] != part["sha256"] or saved["shard_size"] != frozen["shard_size"]:
                    raise ValueError("Shared test index differs from frozen preprocessing")
    spec = {"frozen": frozen, "split": "test", "queries": roster.height, "raw_inputs": raw_inputs,
            "query_ids": fingerprint(roster["s1_id"].to_list()), "workers": workers, "threads": threads,
            "query_batch": query_batch, "feature_batch": feature_batch,
            "partitions": partitions(roster, workers), "parallel_code": sha256(Path(__file__)),
            "inference_code": sha256(Path(__file__).with_name("inference.py")),
            "view_contract_sha256": sha256(views / "test_contract.json")}
    spec = _executor_contract(out, spec)
    if (out / "inference_complete.json").exists():
        _, complete, stored = inference_manifest(out)
        if not stored.equals(roster):
            raise ValueError("Completed parallel roster changed")
        for _, _ in iter_score_shards(out, complete):
            pass
        return
    jobs = []
    for number, partition in enumerate(spec["partitions"]):
        jobs.append((number, {"prepared": prepared, "views": views, "indexes": indexes, "model_run": model_run,
            "out": out / "scores" / f"part-{number:04d}", "model_name": model_name, "split": "test",
            "threads": threads, "query_batch": query_batch, "feature_batch": feature_batch, "_partition": partition}))
    for number in bounded_process_jobs(jobs, _run_partition, initializer=_initialize, initargs=(threads,),
                                       workers=min(workers, len(jobs)), worker_threads=threads):
        print(f"Completed test partition {number+1}/{len(jobs)}", flush=True)
    merge_partitions(out, roster, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "views", "indexes", "model-run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("direct", "support"), default="support")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--query-batch", type=int, default=1000)
    parser.add_argument("--feature-batch", type=int, default=250)
    args = parser.parse_args()
    parallel_infer(args.prepared, args.views, args.indexes, args.model_run, args.out,
                   args.model, args.split, args.workers, args.threads, args.query_batch, args.feature_batch)


if __name__ == "__main__":
    main()
