from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from plan1.normalize import normalize_records
from plan1.translit import make_converter
from plan3.artifacts import contract
from plan3.evaluate import score
from plan3.index import SparseIndex
from plan3.retrieve import union_channels


def normalized(rows):
    raw = pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country", "source"], orient="row")
    return normalize_records(raw, make_converter({"लक्ष्मी": "lakshmi"}))


def test_transliteration_unicode_and_compound_numbers():
    n = normalized([
        ("S1-q", "Lakshmi Pvt. Ltd.", "No.5/257 Main Road", "India", "S1"),
        ("S2-very-long-identifier-with-leading-0000", "लक्ष्मी", "N°49 12B 12/3 A-5", "India", "S2"),
    ])
    assert n["name_red"].to_list() == ["lakshmi", "lakshmi"]
    assert "5/257" in n["addr_nums"][0]
    assert set(n["addr_nums"][1]) == {"49", "12b", "12/3", "a-5"}


@pytest.mark.parametrize("channel", ["word", "char", "name", "missing"])
def test_sharded_search_matches_full_matrix_and_keeps_orphans(tmp_path, channel):
    t = normalized([
        ("S2-00001", "Alpha Tools", "12 Main Road", "NewCountry", "S2"),
        ("S2-00002", "Alpha Tool", "12 Main St", "NewCountry", "S2"),
        ("S2-00003", "Beta Labs", "", "NewCountry", "S2"),
        ("S2-00004", "Alpha Tools", "", "NewCountry", "S2"),
        ("S2-unowned-very-long-target-0000000001", "Gamma Supplies", "9 Oak Road", "NewCountry", "S2"),
    ])
    q = normalized([
        ("S1-0001", "Alpha Tools", "12 Main Road", "NewCountry", "S1"),
        ("S1-0002", "Beta Labs", "", "NewCountry", "S1"),
        ("S1-0003", "Gamma Supplies", "9 Oak Road", "NewCountry", "S1"),
    ])
    idx = SparseIndex.build(tmp_path / channel, t, channel, "fixture", shard_size=2, min_df=1)
    from plan3.index import CHANNELS
    subset = t.filter(pl.col("addr_missing")) if channel == "missing" else t
    subset = subset.sort("entity_id")
    field = CHANNELS[channel]["field"]
    expected = (idx.transform(q[field].to_list()) @ idx.transform(subset[field].to_list()).T).toarray()
    found = idx.query(q, 2, threads=1)
    for row, qid in enumerate(q["entity_id"]):
        values = sorted(expected[row][expected[row] > 0], reverse=True)[:2]
        scores = found.filter(pl.col("s1_id") == qid).sort("rank")["score"].to_numpy()
        np.testing.assert_allclose(scores, values, atol=2e-6)
    if channel != "missing":
        assert "S2-unowned-very-long-target-0000000001" in found["target_id"].to_list()
    subset_pairs = pl.DataFrame({"s1_id": [q["entity_id"][0]] * subset.height, "target_id": subset["entity_id"]})
    np.testing.assert_allclose(idx.pair_scores(q, subset_pairs)["score"], expected[0], atol=2e-6)


def test_word_idf_matches_upstream_definition(tmp_path):
    t = normalized([(f"S2-{i}", n, a, "India", "S2") for i, (n,a) in enumerate([
        ("Alpha Alpha Tools", "12 Main Rd"), ("Alpha Tools", "13 Main Road"),
        ("Beta Labs", "5 Oak Rd"), ("Beta Supplies", "12 Oak Rd")])])
    idx = SparseIndex.build(tmp_path / "word", t, "word", "fixture", shard_size=2, min_df=2)
    texts = t.sort("entity_id")["retrieval_text"].to_list()
    vec = TfidfVectorizer(analyzer=str.split, min_df=2, max_features=300000,
                          sublinear_tf=True, smooth_idf=True, norm="l2", dtype=np.float32)
    expected = vec.fit_transform(texts)
    assert idx.vocabulary == vec.vocabulary_
    np.testing.assert_allclose(idx.transform(texts).toarray(), expected.toarray(), atol=2e-7)


def test_empty_vocabulary_and_no_padding(tmp_path):
    t = normalized([("S2-0", "", "", "France", "S2")])
    idx = SparseIndex.build(tmp_path / "empty", t, "word", "empty", shard_size=1)
    assert idx.query(normalized([("S1-0", "test", "", "France", "S1")]), 10).is_empty()


def test_cache_rejects_changed_contract(tmp_path):
    p = tmp_path / "contract.json"
    contract(p, {"dictionary": "a"})
    with pytest.raises(ValueError, match="changed"):
        contract(p, {"dictionary": "b"})


def test_metric_counts_missing_candidates_and_empty_owners():
    truth = pl.DataFrame({"s1_id": ["a", "a", "c"], "target_id": ["1", "2", "3"]})
    pred = pl.DataFrame({"s1_id": ["a"], "target_id": ["1"]})
    result = score(pred, truth, pl.Series(["a", "b", "c"]))
    assert result["link_recall"] == pytest.approx(1/3)
    assert result["macro_f05"] == pytest.approx((1.25/1.5 + 1 + 0)/3)


def test_union_keeps_channel_evidence():
    def part(ids, scores):
        return pl.DataFrame({"s1_id": ["q"]*len(ids), "target_id": ids,
                             "score": pl.Series(scores,dtype=pl.Float32),
                             "rank": pl.Series(list(range(1,len(ids)+1)),dtype=pl.Int32)})
    out = union_channels({"word": part(["x","y"], [.9,.8]), "missing": part(["y","z"], [.99,.7])})
    assert out.height == 3
    y = out.filter(pl.col("target_id")=="y").row(0,named=True)
    assert y["word_rank"] == 2 and y["missing_rank"] == 1
    assert out.filter(pl.col("target_id")=="z")["word_rank"].null_count() == 1
