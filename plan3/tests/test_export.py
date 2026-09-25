import polars as pl
import pytest

from plan3.artifacts import fingerprint, read_json, sha256, write_json
from plan3.export import SCORE_SCHEMA, export, inference_manifest, iter_score_shards, verify_files


def make_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source_rows = {
        1: [("S1-b", "US"), ("S1-empty", "India"), ("S1-a", "US"),
            ("S1-c", "US"), ("S1-no", "US")],
        2: [("S2-shared", "US"), ("S2-low", "US"), ("S2-india", "India"),
            ("S2-Ω", "US")],
        3: [("S3-tie", "US"), ("S3-alone", "US")],
    }
    inputs = {}
    for source, rows in source_rows.items():
        path = raw / f"test_source{source}.tsv"
        path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                        "".join(f'{qid}\tA, "name"\t1 Road\t{country}\n' for qid, country in rows),
                        encoding="utf-8", newline="\n")
        inputs[f"test/{path.name}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    inference = tmp_path / "inference"
    (inference / "scores").mkdir(parents=True)
    roster = pl.DataFrame(source_rows[1], schema=["s1_id", "country"], orient="row").sort("s1_id")
    roster.write_parquet(inference / "roster.parquet")
    write_json(inference / "inference_contract.json", {
        "split": "test", "queries": roster.height,
        "query_ids": fingerprint(roster["s1_id"].to_list()), "raw_inputs": inputs,
        "frozen": {"decision": {"threshold": .9, "selected_on": "select"}},
    })
    rows = {
        "a.parquet": [("S1-b", "S2-shared", .94), ("S1-b", "S3-tie", .95),
                      ("S1-b", "S2-low", .7), ("S1-c", "S2-low", .7), ("S1-b", "S2-Ω", .91)],
        "z.parquet": [("S1-a", "S2-shared", .98), ("S1-a", "S3-tie", .95),
                      ("S1-a", "S3-alone", .9)],
    }
    for name, pairs in rows.items():
        frame = pl.DataFrame([(qid, tid, tid[:2], "US", p, p, .5, 1) for qid, tid, p in pairs],
                             schema=SCORE_SCHEMA, orient="row")
        frame.write_parquet(inference / "scores" / name)
    complete = {"contract_sha256": sha256(inference / "inference_contract.json"),
                "roster_sha256": sha256(inference / "roster.parquet"), "queries": roster.height,
                "shards": ["z.parquet", "a.parquet"], "score_shards": {}}
    write_json(inference / "inference_complete.json", complete)
    rehash_shards(inference)
    return inference, raw, tmp_path / "output"


def rehash_shards(inference):
    complete = read_json(inference / "inference_complete.json")
    complete["score_shards"] = {
        name: {"sha256": sha256(inference / "scores" / name),
               "rows": pl.read_parquet(inference / "scores" / name).height}
        for name in complete["shards"]
    }
    write_json(inference / "inference_complete.json", complete)


def replace_shard(inference, frame, name="z.parquet"):
    frame.write_parquet(inference / "scores" / name)
    rehash_shards(inference)


def test_global_ownership_across_shards_empty_owners_and_exact_format(tmp_path):
    inference, raw, out = make_inputs(tmp_path)
    report = export(inference, raw, out)
    assert (out / "candidate_pairs.tsv").read_bytes() == (
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-b\tS2-low,S2-shared,S2-Ω,S3-tie\n"
        "S1-empty\t\n"
        "S1-a\tS2-shared,S3-alone,S3-tie\n"
        "S1-c\tS2-low\n"
        "S1-no\t\n").encode("utf-8")
    assert (out / "matching_results.tsv").read_bytes() == (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-b\tS2-Ω\n"
        "S1-empty\t\n"
        "S1-a\tS2-shared,S3-alone,S3-tie\n"
        "S1-c\t\n"
        "S1-no\t\n").encode("utf-8")
    assert report["counts"] == {
        "owners": 5, "candidate_pairs": 8, "accepted_pairs": 6, "matched_pairs": 4,
        "owners_without_candidates": 2, "owners_without_matches": 3,
        "claims_lost_at_ownership": 2,
    }
    assert verify_files(out) == report
    with pytest.raises(FileExistsError):
        export(inference, raw, out)
    with (out / "candidate_pairs.tsv").open("ab") as handle:
        handle.write(b"extra\t\n")
    with pytest.raises(ValueError, match="Output file hash"):
        verify_files(out)


@pytest.mark.parametrize("what", ["contract", "raw", "score", "roster", "missing", "extra"])
def test_rejects_corruption_or_missing_provenance(tmp_path, what):
    inference, raw, out = make_inputs(tmp_path)
    if what == "contract":
        path = inference / "inference_contract.json"
        value = read_json(path)
        value["frozen"]["decision"]["threshold"] = .5
        write_json(path, value)
    elif what == "raw":
        with (raw / "test_source1.tsv").open("ab") as handle:
            handle.write(b"\n")
    elif what == "score":
        path = inference / "scores/a.parquet"
        pl.read_parquet(path).with_columns(p=pl.lit(.99)).write_parquet(path)
    elif what == "roster":
        path = inference / "roster.parquet"
        pl.read_parquet(path).with_columns(country=pl.lit("Other")).write_parquet(path)
    elif what == "missing":
        (inference / "scores/a.parquet").unlink()
    else:
        pl.read_parquet(inference / "scores/a.parquet").write_parquet(inference / "scores/extra.parquet")
    with pytest.raises(ValueError):
        export(inference, raw, out)
    assert not (out / "candidate_pairs.tsv").exists()
    assert not (out / "matching_results.tsv").exists()


@pytest.mark.parametrize("field,value,message", [
    ("p", float("nan"), "Non-finite"),
    ("p_direct", float("inf"), "Non-finite"),
    ("score", float("-inf"), "Non-finite"),
    ("p", 1.1, "outside"),
    ("s1_id", "S1-unknown", "Unknown scored owner"),
    ("target_id", "S2-unknown", "Unknown target"),
    ("target_id", "S2-india", "Cross-country"),
    ("source", "S3", "incorrect source"),
    ("country", "India", "Cross-country"),
])
def test_rejects_invalid_scored_pairs(tmp_path, field, value, message):
    inference, raw, out = make_inputs(tmp_path)
    frame = pl.read_parquet(inference / "scores/z.parquet")
    # Alter just one row to preserve unrelated pair uniqueness.
    frame = pl.concat([frame.head(1).with_columns(
        pl.lit(value).cast(SCORE_SCHEMA[field]).alias(field)), frame.slice(1)])
    replace_shard(inference, frame)
    with pytest.raises(ValueError, match=message):
        export(inference, raw, out)
    assert not (out / "candidate_pairs.tsv").exists()


@pytest.mark.parametrize("across", [False, True])
def test_rejects_duplicate_pairs_even_across_shards(tmp_path, across):
    inference, raw, out = make_inputs(tmp_path)
    frame = pl.read_parquet(inference / "scores/z.parquet")
    duplicate = (pl.read_parquet(inference / "scores/a.parquet").head(1) if across else frame.head(1))
    replace_shard(inference, pl.concat([frame, duplicate]))
    message = "multiple score shards" if across else "Duplicate scored candidate pairs"
    with pytest.raises(ValueError, match=message):
        export(inference, raw, out)
    assert not (out / "candidate_pairs.tsv").exists()
    assert not (out / "matching_results.tsv").exists()


def test_rejects_incomplete_roster_even_if_its_file_hash_is_current(tmp_path):
    inference, raw, out = make_inputs(tmp_path)
    roster_path = inference / "roster.parquet"
    pl.read_parquet(roster_path).head(4).write_parquet(roster_path)
    complete = read_json(inference / "inference_complete.json")
    complete["roster_sha256"] = sha256(roster_path)
    write_json(inference / "inference_complete.json", complete)
    with pytest.raises(ValueError, match="roster"):
        export(inference, raw, out)


def test_all_empty_scores_still_write_every_owner(tmp_path):
    inference, raw, out = make_inputs(tmp_path)
    for name in ("a.parquet", "z.parquet"):
        pl.DataFrame(schema=SCORE_SCHEMA).write_parquet(inference / "scores" / name)
    rehash_shards(inference)
    report = export(inference, raw, out)
    assert report["counts"]["candidate_pairs"] == report["counts"]["matched_pairs"] == 0
    assert report["counts"]["owners_without_candidates"] == 5
    for name in ("candidate_pairs.tsv", "matching_results.tsv"):
        assert (out / name).read_bytes().splitlines()[1:] == [
            b"S1-b\t", b"S1-empty\t", b"S1-a\t", b"S1-c\t", b"S1-no\t"]


def test_owner_groups_cannot_split_even_without_duplicate_pairs(tmp_path):
    inference, raw, out = make_inputs(tmp_path)
    frame = pl.read_parquet(inference / "scores/z.parquet")
    extra = frame.head(1).with_columns(s1_id=pl.lit("S1-c"), target_id=pl.lit("S2-Ω"))
    replace_shard(inference, pl.concat([frame, extra]))
    with pytest.raises(ValueError, match="multiple score shards"):
        export(inference, raw, out)


def test_shared_manifest_and_shard_helpers_allow_train_evaluation(tmp_path):
    inference, _, _ = make_inputs(tmp_path)
    path = inference / "inference_contract.json"
    spec = read_json(path)
    spec["split"] = "train"
    write_json(path, spec)
    complete = read_json(inference / "inference_complete.json")
    complete["contract_sha256"] = sha256(path)
    write_json(inference / "inference_complete.json", complete)
    loaded, marker, roster = inference_manifest(inference)
    assert loaded["split"] == "train" and roster.height == 5
    assert sum(frame.height for _, frame in iter_score_shards(inference, marker)) == 8
