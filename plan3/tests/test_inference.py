import numpy as np
import polars as pl

from plan3.inference import retrieve_batch, predict_candidates, SCORE_SCHEMA
from plan3.inference import committed_batch
from plan3.artifacts import fingerprint, sha256, write_json
import pytest


class Index:
    def __init__(self, rows):
        self.rows = rows
        self.requested = []

    def query(self, queries, k, threads):
        self.requested.append(k)
        return pl.DataFrame(self.rows, schema={"s1_id": pl.String, "target_id": pl.String,
            "score": pl.Float32, "rank": pl.Int32}, orient="row")

    def pair_scores(self, queries, pairs):
        return pairs.select("s1_id", "target_id").with_columns(score=pl.lit(.37, pl.Float32))


def test_inference_preserves_cross_channel_ranks_beyond_final_cut():
    word = Index([("q", "S2-a", .8, 1), ("q", "S2-b", .4, 150), ("q", "S2-c", .3, 180)])
    missing = Index([("q", "S2-b", .9, 2), ("q", "S2-a", .7, 40), ("q", "S2-c", .6, 80)])
    c = retrieve_batch(pl.DataFrame({"entity_id": ["q"]}), {"S2": {"word": word, "missing": missing}},
                       {"word": 200, "missing": 100}, 100, 25, "India", 1)
    assert c["target_id"].to_list() == ["S2-a", "S2-b"]
    assert c["word_rank"].to_list() == [1, 150]
    assert c["missing_rank"].to_list() == [40, 2]
    assert word.requested == [200] and missing.requested == [100]
    np.testing.assert_allclose(c["score"].to_numpy(), [.37, .37])


def test_zero_candidate_batch_does_not_call_models():
    class NeverPredict:
        def predict(self, *args, **kwargs):
            raise AssertionError("An empty batch must not call the model")
    result = predict_candidates(pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String}),
        None, None, None, NeverPredict())
    assert result.schema == SCORE_SCHEMA and result.height == 0


def test_incomplete_run_does_not_adopt_or_bless_uncommitted_shards(tmp_path):
    path = tmp_path / "scores.parquet"
    ids = pl.Series(["q"])
    pl.DataFrame({"p": [.9]}).write_parquet(path)
    assert not committed_batch(path, ids)
    write_json(path.with_suffix(".json"), {"query_ids": fingerprint(ids.to_list()),
        "sha256": sha256(path), "rows": 1})
    assert committed_batch(path, ids)
    pl.DataFrame({"p": [.1]}).write_parquet(path)
    with pytest.raises(ValueError, match="checkpoint changed"):
        committed_batch(path, ids)
