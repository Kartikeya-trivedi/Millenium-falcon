"""Build complete-query features and label only after retrieval."""
from __future__ import annotations

import gc
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import polars as pl

from .artifacts import contract, fingerprint, load_prepared, parquet, read_json, sha256, write_json
from .budget import keep_candidates
from .features import FEATURES, rich_features

_WORKER_IDFS = None
_WORKER_LIMITS = None


@dataclass
class FeatureBatch:
    start: int
    candidates: pl.DataFrame
    queries: pl.DataFrame
    targets: pl.DataFrame
    labels: pl.DataFrame
    roles: pl.DataFrame
    country: str
    output: Path


def _write_batch(batch: FeatureBatch, idfs):
    """Keep all candidates for an owner in the same feature call and output."""
    c, q, t = batch.candidates, batch.queries, batch.targets
    f = rich_features(c, q, t, idfs)
    f = (f.join(batch.labels.with_columns(label=pl.lit(1, pl.Int8)),
                on=["s1_id", "target_id"], how="left", maintain_order="left")
         .with_columns(pl.col("label").fill_null(0))
         .join(batch.roles, on="s1_id", how="left", maintain_order="left")
         .join(t.select(target_id="entity_id", t_name="name_red", t_address="addr_norm",
                        t_numbers="addr_nums"), on="target_id", how="left", maintain_order="left")
         .with_columns(country=pl.lit(batch.country)))
    if f.height != c.height or not f.select("s1_id", "target_id").equals(c.select("s1_id", "target_id")):
        raise ValueError("Batch feature joins changed candidate rows or order")
    parquet(batch.output, f)
    return batch.start, q.height, f.height


def initialize_native_threads(threads):
    from threadpoolctl import threadpool_limits

    limits = threadpool_limits(limits=threads)
    if pl.thread_pool_size() > threads:
        raise RuntimeError("Feature worker started with too many Polars threads")
    return limits


def _initialize_worker(idfs, threads):
    global _WORKER_IDFS, _WORKER_LIMITS
    _WORKER_IDFS = idfs
    _WORKER_LIMITS = initialize_native_threads(threads)


def _process_batch(batch):
    if _WORKER_IDFS is None:
        raise RuntimeError("Feature worker has not been initialized")
    return _write_batch(batch, _WORKER_IDFS)


@contextmanager
def _worker_environment(threads):
    # Spawn imports NumPy and Polars before the initializer runs. Their native
    # thread pools must inherit limits before those imports, not afterward.
    names = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def bounded_process_jobs(jobs, function, *, initializer, initargs, workers, worker_threads):
    """Spawn workers without retaining more than twice their count in flight."""
    if workers < 1 or not 1 <= worker_threads <= 4:
        raise ValueError("Positive workers and one to four native threads per worker are required")
    tasks = iter(jobs)
    # A dequeued batch can contain many target records. Do not use executor.map,
    # which eagerly consumes the generator and retains the entire country.
    with _worker_environment(worker_threads):
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=initializer, initargs=initargs) as pool:
            pending = set()
            exhausted = False
            try:
                while pending or not exhausted:
                    while len(pending) < 2 * workers and not exhausted:
                        try:
                            batch = next(tasks)
                        except StopIteration:
                            exhausted = True
                        else:
                            pending.add(pool.submit(function, batch))
                    if not pending:
                        break
                    ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        yield future.result()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise


def run_feature_batches(batches, idfs, workers=1, worker_threads=4):
    """Write independent complete-owner batches with a bounded process queue."""
    if workers < 1 or not 1 <= worker_threads <= 4:
        raise ValueError("Positive workers and one to four native threads per worker are required")
    if workers == 1:
        for batch in batches:
            yield _write_batch(batch, idfs)
        return
    yield from bounded_process_jobs(batches, _process_batch, initializer=_initialize_worker,
        initargs=(idfs, worker_threads), workers=workers, worker_threads=worker_threads)


def query_batches(candidates, queries, targets, size=250):
    """Index complete owner groups once instead of rescanning a country per batch."""
    if size < 1:
        raise ValueError("Feature batch size must be positive")
    candidates=candidates.sort("s1_id","target_id")
    queries=queries.sort("entity_id")
    counts=candidates.group_by("s1_id").len().sort("s1_id")
    if counts["s1_id"].to_list()!=queries["entity_id"].to_list():
        raise ValueError("Query records do not match candidate groups")
    offsets=[0]+counts["len"].cum_sum().to_list()
    target_rows={key:i for i,key in enumerate(targets["entity_id"])}
    if len(target_rows)!=targets.height or queries["entity_id"].n_unique()!=queries.height:
        raise ValueError("Duplicate record IDs")
    for start in range(0,queries.height,size):
        end=min(start+size,queries.height)
        c=candidates.slice(offsets[start],offsets[end]-offsets[start])
        q=queries.slice(start,end-start)
        indices=[target_rows[key] for key in c["target_id"].unique(maintain_order=True)]
        t=targets[pl.Series(indices,dtype=pl.UInt32)]
        yield start,c,q,t


def featurize(prepared: Path, views: Path, indexes: Path, run: Path, query_batch=250,
              word_k: int|None=None, missing_k: int|None=None, workers=1, worker_threads=4):
    if query_batch < 1 or workers < 1 or not 1 <= worker_threads <= 4:
        raise ValueError("Positive batch size/workers and one to four native threads per worker are required")
    meta = load_prepared(prepared)
    if not (views/"train_complete.json").exists():
        raise ValueError("Additional text views are incomplete")
    default=read_json(run/"budget.json")["chosen"]
    word_k=default["word_k"] if word_k is None else word_k
    missing_k=default["missing_k"] if missing_k is None else missing_k
    spec={"prepared_identity":meta["identity"], "word_k":word_k,"missing_k":missing_k,
          "features":FEATURES,"features_code":sha256(Path(__file__).with_name("features.py")),
          "batching_code":sha256(Path(__file__)),
          "views_contract":sha256(views/"train_contract.json"),
          "query_batch":query_batch,"workers":workers,"worker_threads":worker_threads,
          "reason":"Explicit bounded scoring experiment; compare with stored Tune budget frontier."}
    contract(run/"feature_contract.json",spec)
    sample=pl.read_parquet(run/"samples.parquet")
    truth=pl.read_parquet(prepared/"truth.parquet").join(sample.select("s1_id"),on="s1_id",how="semi")
    true_by_owner=dict(truth.group_by("s1_id").agg("target_id").iter_rows())
    sample_by_owner=dict(sample.select("s1_id","sample").iter_rows())
    freq=pl.read_parquet(views/"train_name_frequency.parquet")
    cands=pl.scan_parquet(str(run/"candidates/*.parquet"))
    for country in sorted(pl.read_parquet(run/"queries.parquet")["country"].unique().to_list()):
        country_cands=keep_candidates(cands.filter(pl.col("country")==country).collect(),word_k,missing_k)
        qids=country_cands.select(entity_id="s1_id").unique()
        tids=country_cands.select(entity_id="target_id").unique()
        records=[]
        for p in meta["partitions"]:
            if p["split"]!="train" or p["country"]!=country:
                continue
            ids=qids if p["source"]=="S1" else tids
            frame=(pl.scan_parquet(prepared/p["file"]).join(ids.lazy(),on="entity_id",how="semi")
                   .join(pl.scan_parquet(views/p["file"]),on="entity_id")
                   .join(freq.lazy(),on=["name_red","country"],how="left").collect(engine="streaming"))
            records.append(frame)
        qrecords=pl.concat([f for f in records if f["source"][0]=="S1"])
        trecords=pl.concat([f for f in records if f["source"][0]!="S1"])
        idfs={}
        for source in ("S2","S3"):
            d=indexes/"train"/f"{fingerprint(country)[:16]}-{source}"/"word"
            vocab=pl.read_parquet(d/"vocabulary.parquet")
            idfs[source]=(dict(zip(vocab["term"].to_list(),vocab["idf"].to_list())),read_json(d/"index.json")["rows"])
        country_ids=qrecords["entity_id"].sort()
        print(f"Feature batches: {country}, {country_ids.len():,} queries, {country_cands.height:,} pairs",flush=True)
        def batches():
            for start,c,q,t in query_batches(country_cands,qrecords,trecords,query_batch):
                output=run/"features"/f"{fingerprint(country)[:16]}-{start:08d}.parquet"
                if output.exists():
                    continue
                ids=q["entity_id"].to_list()
                labels=pl.DataFrame([(qid,tid) for qid in ids for tid in true_by_owner.get(qid,[])],
                                    schema={"s1_id":pl.String,"target_id":pl.String},orient="row")
                roles=pl.DataFrame({"s1_id":ids,"sample":[sample_by_owner[qid] for qid in ids]})
                yield FeatureBatch(start,c,q,t,labels,roles,country,output)

        completed = 0
        for start, owners, pairs in run_feature_batches(batches(), idfs, workers, worker_threads):
            completed += 1
            if completed % 5 == 0 or completed == 1:
                print(f"  {country}: completed {completed:,} new batches; "
                      f"batch {start:,} has {owners:,} owners / {pairs:,} pairs",flush=True)
        del records,qrecords,trecords,country_cands,idfs
        gc.collect()
    write_json(run/"features_complete.json",{"status":"featurized","feature_contract_sha":sha256(run/"feature_contract.json")})
