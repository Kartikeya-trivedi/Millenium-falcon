import polars as pl
import pytest

from plan3.diagnose import orphan_summary


def test_orphan_share_denominators_do_not_mix_targets_and_pairs():
    owners = pl.DataFrame({"s1_id": ["a", "b"], "target_id": ["x", "y"]})
    scored = pl.DataFrame({"s1_id": ["a", "a", "a", "b"], "target_id": ["x", "y", "z", "z"],
                           "label": [1, 0, 0, 0]})
    accepted = scored.head(2).select("s1_id", "target_id")
    r = orphan_summary(scored, accepted, owners, total_targets=4)
    assert r["full_target_corpus"]["unowned_share"] == .5
    assert r["retrieved_negative_pairs"]["unowned_pair_share"] == pytest.approx(2/3)
    assert r["retrieved_negative_pairs"]["distinct_unowned_targets"] == 1
    assert r["false_accepted_pairs_before_ownership"]["unowned_pair_share"] == 0
