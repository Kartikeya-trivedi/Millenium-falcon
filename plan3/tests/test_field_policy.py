import numpy as np
import polars as pl
import pytest

from plan3.evaluate import score
from plan3.field_policy import apply_policy, select_thresholds


def test_two_threshold_search_matches_complete_owner_metric():
    p = pl.DataFrame({"s1_id": ["a", "a", "b", "empty"], "target_id": ["x", "z", "y", "u"],
                      "p": [.9, .6, .6, .7], "cand_addr_missing": [0., 0., 1., 0.]})
    truth = pl.DataFrame({"s1_id": ["a", "b"], "target_id": ["x", "y"]})
    roster = pl.Series(["a", "b", "empty", "zero_candidate"])
    best, table = select_thresholds(p, truth, roster, .8, grid=np.array([.5, .8, 1.000001]))
    for row in table.iter_rows(named=True):
        assert row["macro_f05"] == pytest.approx(score(apply_policy(p, row), truth, roster)["macro_f05"])
    assert best["address_threshold"] == .8 and best["missing_threshold"] == .5
    assert best["macro_f05"] == 1
