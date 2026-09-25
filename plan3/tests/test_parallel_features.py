from dataclasses import replace
import os

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from plan1.normalize import normalize_records
from plan1.translit import make_converter
import plan3.featurize as module
from plan3.features import FEATURES, rich_features
from plan3.retrieve import union_channels
from plan3.views import build_views


def fixture_batches(folder):
    raw = pl.DataFrame([
        ("S1-05", "Lakshmi DBA Alpha", "12B Main Rd Apt 5", "India", "S1"),
        ("S1-02", "Alpha", "", "India", "S1"),
        ("S1-03", "www.alpha.co.in", "N°49 Rue Alpha", "India", "S1"),
        ("S1-04", "Beta Ltd", "2 Hill St", "India", "S1"),
        ("S1-01", "Lakshmi", "12 Main Road", "India", "S1"),
        ("S2-a", "लक्ष्मी अपरिचित", "12/3 Main Rd Apt 6", "India", "S2"),
        ("S2-b", "Beta Labs", "2 Hill Street", "India", "S2"),
        ("S3-c", "www.alpha.co.in", "", "India", "S3"),
        ("S3-d", "Alpha", "49 Rue Alpha", "India", "S3"),
    ], schema=["entity_id", "business_name", "business_address", "country", "source"], orient="row")
    mapping = {"लक्ष्मी": "lakshmi"}
    records = (normalize_records(raw, make_converter(mapping))
               .join(build_views(raw, mapping), on="entity_id")
               .with_columns(reference_name_count=pl.lit(1, pl.UInt32)))
    q = records.filter(pl.col("source") == "S1")
    t = records.filter(pl.col("source") != "S1")
    proposal = pl.DataFrame([
        (qid, tid, score, rank) for qid in q["entity_id"]
        for tid, score, rank in [("S3-d", .72, 2), ("S2-a", .95, 1), ("S3-c", .81, 1), ("S2-b", .5, 2)]
    ], schema={"s1_id": pl.String, "target_id": pl.String, "score": pl.Float32, "rank": pl.Int32},
        orient="row")
    missing = proposal.filter(pl.col("target_id") == "S3-c").with_columns(score=pl.lit(.9, pl.Float32))
    candidates = (union_channels({"word": proposal, "missing": missing})
                  .join(proposal, on=["s1_id", "target_id"])
                  .with_columns(source=pl.col("target_id").str.slice(0, 2)))
    truth = {"S1-01": ["S2-a", "S3-c"], "S1-03": ["S3-d"], "S1-05": ["S2-a"]}
    roles = dict(zip(sorted(q["entity_id"]), ["fit", "support", "tune", "select", "development"]))
    batches = []
    for start, c, queries, targets in module.query_batches(candidates, q, t, size=2):
        ids = queries["entity_id"].to_list()
        labels = pl.DataFrame([(qid, tid) for qid in ids for tid in truth.get(qid, [])],
                              schema={"s1_id": pl.String, "target_id": pl.String}, orient="row")
        samples = pl.DataFrame({"s1_id": ids, "sample": [roles[qid] for qid in ids]})
        batches.append(module.FeatureBatch(start, c, queries, targets, labels, samples,
                                           "India", folder / f"{start:08d}.parquet"))
    idfs = {"S2": ({"lakshmi": 3.5, "alpha": 2.25, "12": 4.75, "main": 2.0}, 100),
            "S3": ({"alpha": 2.5, "49": 3.0, "rue": 2.25}, 150)}
    return batches, idfs, candidates, q, t, truth, roles


def test_spawned_features_match_serial_exactly_and_keep_owner_context(tmp_path):
    batches, idfs, candidates, queries, targets, truth, roles = fixture_batches(tmp_path / "serial")
    serial = list(module.run_feature_batches(batches, idfs))
    concurrent = [replace(batch, output=tmp_path / "parallel" / batch.output.name) for batch in batches]
    parallel = list(module.run_feature_batches(iter(concurrent), idfs, workers=2, worker_threads=1))
    assert sorted(serial) == sorted(parallel)
    assert sum(result[1] for result in parallel) == 5
    assert sum(result[2] for result in parallel) == candidates.height == 20
    for before, after in zip(batches, concurrent):
        assert_frame_equal(pl.read_parquet(before.output), pl.read_parquet(after.output), check_exact=True)
    actual = pl.concat([pl.read_parquet(batch.output) for batch in batches])
    original = rich_features(candidates.sort("s1_id", "target_id"), queries, targets, idfs)
    assert_frame_equal(actual.select("s1_id", "target_id", *FEATURES),
                       original.select("s1_id", "target_id", *FEATURES), check_exact=True)
    assert actual["label"].to_list() == [
        int(tid in truth.get(qid, [])) for qid, tid in actual.select("s1_id", "target_id").iter_rows()]
    assert actual["sample"].to_list() == [roles[qid] for qid in actual["s1_id"]]
    assert actual.select("s1_id", "target_id").equals(candidates.sort("s1_id", "target_id").select("s1_id", "target_id"))


def test_pool_queue_is_bounded_and_restores_parent_environment(monkeypatch):
    monkeypatch.setenv("POLARS_MAX_THREADS", "7")
    submitted = []
    observed = []

    class Result:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

        def cancel(self):
            pass

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["max_workers"] == 2
            assert kwargs["initializer"] is module._initialize_worker
            assert kwargs["initargs"] == ({}, 3)
            assert os.environ["POLARS_MAX_THREADS"] == "3"
            assert os.environ["OPENBLAS_NUM_THREADS"] == "3"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, batch):
            assert function is module._process_batch
            submitted.append(batch)
            return Result(batch)

    def one_finished(pending, **kwargs):
        observed.append(len(pending))
        assert len(pending) <= 4
        first = min(pending, key=lambda result: result.value)
        return {first}, pending - {first}

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(module, "wait", one_finished)
    result = list(module.run_feature_batches(iter(range(20)), {}, workers=2, worker_threads=3))
    assert result == submitted == list(range(20))
    assert max(observed) == 4
    assert os.environ["POLARS_MAX_THREADS"] == "7"


def test_parallel_worker_failure_is_reported_without_writing_bad_batch(tmp_path):
    batches, idfs, *_ = fixture_batches(tmp_path / "invalid")
    bad = replace(batches[0], roles=pl.concat([batches[0].roles, batches[0].roles.head(1)]))
    with pytest.raises(ValueError, match="changed candidate rows"):
        list(module.run_feature_batches([bad], idfs, workers=2, worker_threads=1))
    assert not bad.output.exists()


@pytest.mark.parametrize("workers,threads", [(0, 1), (-1, 4), (2, 0), (2, 5)])
def test_worker_limits_reject_oversubscription(workers, threads):
    with pytest.raises(ValueError):
        list(module.run_feature_batches([], {}, workers=workers, worker_threads=threads))
