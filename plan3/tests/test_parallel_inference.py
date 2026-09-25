from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from plan1.normalize import normalize_records
from plan1.translit import make_converter
from plan3.artifacts import fingerprint, partition_path, read_json, sha256, source_hashes, write_json
from plan3.export import export, inference_manifest, iter_score_shards, score_shard_path
from plan3.features import FEATURES
from plan3.index import SparseIndex
from plan3.inference import infer, partition_queries
from plan3.parallel_inference import merge_partitions, parallel_infer, partitions
from plan3.views import build_views
from plan3.tests.test_export import make_inputs


def test_ranges_preserve_complete_sorted_roster_and_public_test_gate(tmp_path):
    roster = pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(5)], "country": ["US"]*5})
    ranges = partitions(roster, 3)
    queries = roster.rename({"s1_id": "entity_id"})
    combined = pl.concat([partition_queries(queries, part) for part in ranges])
    assert_frame_equal(combined, queries)
    assert len(partitions(roster, 20)) == 5
    with pytest.raises(ValueError, match="complete test roster"):
        infer(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, split="test", query_ids=tmp_path/"partial")
    with pytest.raises(ValueError, match="complete test roster"):
        partition_queries(queries, {**ranges[0], "full_query_ids": "wrong"})


def child_fixture(tmp_path):
    original, raw, unused = make_inputs(tmp_path)
    original_spec, original_complete, roster = inference_manifest(original)
    scored = pl.concat([frame for _, frame in iter_score_shards(original, original_complete)])
    out = tmp_path / "parallel"
    spec = {**original_spec, "workers": 2, "partitions": partitions(roster, 2),
            "view_contract_sha256": "views", "inference_code": "code"}
    write_json(out / "inference_contract.json", spec)
    for number, part in enumerate(spec["partitions"]):
        child = out / "scores" / f"part-{number:04d}"
        (child / "scores").mkdir(parents=True)
        subset = roster.slice(part["start"], part["stop"]-part["start"])
        subset.write_parquet(child / "roster.partition")
        child_spec = {**spec, "queries": subset.height, "query_ids": fingerprint(subset["s1_id"].to_list()),
                      "partition": part}
        write_json(child / "inference_contract.json", child_spec)
        frame = scored.filter(pl.col("s1_id").is_in(subset["s1_id"].implode()))
        frame.write_parquet(child / "scores/one.parquet")
        write_json(child / "inference_complete.json", {
            "contract_sha256": sha256(child / "inference_contract.json"), "queries": subset.height,
            "roster_file": "roster.partition", "roster_sha256": sha256(child / "roster.partition"),
            "shards": ["one.parquet"],
            "score_shards": {"one.parquet": {"sha256": sha256(child/"scores/one.parquet"), "rows": frame.height}},
        })
    return out, raw, roster, spec


def test_merge_and_nested_export_preserve_global_ownership_and_empty_owners(tmp_path):
    out, raw, roster, spec = child_fixture(tmp_path)
    merge_partitions(out, roster, spec)
    _, complete, actual = inference_manifest(out)
    assert_frame_equal(actual, roster)
    assert set(complete["shards"]) == {"part-0000/scores/one.parquet", "part-0001/scores/one.parquet"}
    report = export(out, raw, tmp_path / "output")
    assert report["counts"]["owners"] == 5 and report["counts"]["matched_pairs"] == 4
    assert report["counts"]["claims_lost_at_ownership"] == 2
    assert b"S1-empty\t\n" in (tmp_path / "output/matching_results.tsv").read_bytes()
    # A child is authenticated but remains ineligible as a complete test output.
    with pytest.raises(ValueError, match="complete raw test roster"):
        export(out / "scores/part-0000", raw, tmp_path / "partial-output")


@pytest.mark.parametrize("corruption", ["roster", "frozen", "score", "foreign_owner", "extra"])
def test_merge_rejects_incomplete_mixed_or_corrupt_children(tmp_path, corruption):
    out, raw, roster, spec = child_fixture(tmp_path)
    child = out / "scores/part-0000"
    cp, cm = child/"inference_contract.json", child/"inference_complete.json"
    child_spec, marker = read_json(cp), read_json(cm)
    if corruption == "roster":
        subset = pl.read_parquet(child/"roster.partition").head(1)
        subset.write_parquet(child/"roster.partition")
        child_spec["queries"] = marker["queries"] = subset.height
        child_spec["query_ids"] = fingerprint(subset["s1_id"].to_list())
        marker["roster_sha256"] = sha256(child/"roster.partition")
    elif corruption == "frozen":
        child_spec["frozen"]["decision"]["threshold"] = .91
    elif corruption == "score":
        path = child/"scores/one.parquet"
        pl.read_parquet(path).with_columns(p=pl.lit(.2)).write_parquet(path)
    elif corruption == "foreign_owner":
        path = child/"scores/one.parquet"
        frame = pl.read_parquet(path).with_columns(s1_id=pl.lit("S1-outside"))
        frame = frame.unique(["s1_id", "target_id"])
        frame.write_parquet(path)
        marker["score_shards"]["one.parquet"] = {"sha256": sha256(path), "rows": frame.height}
    else:
        pl.read_parquet(child/"scores/one.parquet").write_parquet(child/"scores/extra.parquet")
    write_json(cp, child_spec)
    marker["contract_sha256"] = sha256(cp)
    write_json(cm, marker)
    with pytest.raises(ValueError):
        merge_partitions(out, roster, spec)
    assert not (out / "inference_complete.json").exists()


@pytest.mark.parametrize("name", ["../bad.parquet", "/bad.parquet", "C:/bad.parquet",
                                  "a\\bad.parquet", "a/../bad.parquet", "a//bad.parquet",
                                  "./bad.parquet", "bad.txt"])
def test_nested_shards_reject_path_traversal(tmp_path, name):
    with pytest.raises(ValueError, match="path"):
        score_shard_path(tmp_path, name)


def scoring_fixture(tmp_path):
    prepared, views, indexes, model = [tmp_path/name for name in ("prepared", "views", "indexes", "model")]
    dataset = tmp_path / "dataset"
    dictionary = {"mapping": {}, "training_role": "fit"}
    write_json(prepared/"translit.json", dictionary)
    raw = {
        "S1": [("S1-e", "Alpha Tools", "12 Main Road"), ("S1-b", "", ""),
               ("S1-a", "Beta Tools", "15 Main Road"), ("S1-d", "Alpha Tools", ""),
               ("S1-c", "Beta Tools", "12 Main Road")],
        "S2": [("S2-a", "Alpha Tools", "12 Main Road"), ("S2-b", "Beta Tools", "15 Main Road")],
        "S3": [("S3-a", "Alpha Tools", ""), ("S3-b", "Beta Tools", "12 Main Road")],
    }
    parts, inputs, normalized = [], {}, {}
    for source, rows in raw.items():
        frame = pl.DataFrame([(qid, name, address, "US") for qid, name, address in rows],
            schema=["entity_id", "business_name", "business_address", "country"], orient="row")
        path = dataset / "test" / f"test_source{source[1]}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(path, separator="\t", quote_style="never")
        inputs[f"test/{path.name}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        frame = frame.with_columns(source=pl.lit(source))
        n = normalize_records(frame, make_converter({}))
        target_path = partition_path(prepared, "test", "US", source)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        name = target_path.relative_to(prepared).as_posix()
        n.write_parquet(target_path)
        (views/name).parent.mkdir(parents=True, exist_ok=True)
        build_views(frame, {}).write_parquet(views/name)
        parts.append({"split": "test", "source": source, "country": "US", "file": name,
                      "sha256": sha256(prepared/name), "rows": n.height})
        normalized[source] = n
    normalized["S1"].group_by("name_red", "country").len(name="reference_name_count").write_parquet(
        views/"test_name_frequency.parquet")
    for split in ("train", "test"):
        write_json(views/f"{split}_contract.json", {"prepared_identity": "fixture", "split": split,
                   "views_code": sha256(Path(__file__).parents[1]/"views.py")})
    write_json(prepared/"prepared.json", {"identity": "fixture", "baseline_hashes": source_hashes("plan1"),
        "dictionary_sha256": sha256(prepared/"translit.json"), "dataset": str(dataset),
        "inputs": inputs, "partitions": parts})
    write_json(model/"feature_contract.json", {"prepared_identity": "fixture", "features": FEATURES,
        "features_code": sha256(Path(__file__).parents[1]/"features.py"), "word_k": 100, "missing_k": 25,
        "views_contract": sha256(views/"train_contract.json")})
    write_json(model/"retrieval_contract.json", {"prepared_identity": "fixture", "budgets": {"word": 200, "missing": 100},
        "shard_size": 50000, "index_code": sha256(Path(__file__).parents[1]/"index.py"),
        "retrieve_code": sha256(Path(__file__).parents[1]/"retrieve.py")})
    rng = np.random.default_rng(42)
    x = rng.random((30, len(FEATURES))).astype(np.float32)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1,
                         "num_leaves": 3, "seed": 42}, lgb.Dataset(x, label=(x[:, 0] > .5),
                         feature_name=FEATURES), num_boost_round=3)
    booster.save_model(str(model/"direct.txt"))
    write_json(model/"direct_meta.json", {"feature_contract": sha256(model/"feature_contract.json"),
                                          "model_sha256": sha256(model/"direct.txt")})
    write_json(model/"direct_decision.json", {"selected_on": "select", "threshold": .5})
    for source in ("S2", "S3"):
        for channel in ("word", "missing"):
            SparseIndex.build(indexes/"test"/f"{fingerprint('US')[:16]}-{source}"/channel,
                              normalized[source], channel, sha256(partition_path(prepared, "test", "US", source)))
    return prepared, views, indexes, model, dataset


def test_spawned_full_inference_matches_serial_scores_and_outputs(tmp_path):
    prepared, views, indexes, model, dataset = scoring_fixture(tmp_path)
    serial, parallel = tmp_path/"serial", tmp_path/"parallel"
    infer(prepared, views, indexes, model, serial, model_name="direct", threads=1, query_batch=2, feature_batch=1)
    parallel_infer(prepared, views, indexes, model, parallel, model_name="direct",
                   workers=2, threads=1, query_batch=2, feature_batch=1)
    def scores(path):
        _, complete, _ = inference_manifest(path)
        return pl.concat([frame for _, frame in iter_score_shards(path, complete)]).sort("s1_id", "target_id")
    assert_frame_equal(scores(serial), scores(parallel), check_exact=True)
    for root in (serial, parallel):
        export(root, dataset/"test", root/"output")
    for name in ("candidate_pairs.tsv", "matching_results.tsv"):
        assert (serial/"output"/name).read_bytes() == (parallel/"output"/name).read_bytes()
    parallel_infer(prepared, views, indexes, model, parallel, model_name="direct",
                   workers=2, threads=1, query_batch=2, feature_batch=1)
