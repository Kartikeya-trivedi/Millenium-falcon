import polars as pl
import pytest

from plan3.compare import paired_summary


def test_owner_bootstrap_is_paired_and_keeps_empty_owners():
    a = pl.DataFrame({"s1_id": ["a", "b", "empty"], "f3": [.5, .5, 1.]})
    b = pl.DataFrame({"s1_id": ["empty", "b", "a"], "f3": [1., .7, .6]})
    report, pairs = paired_summary(a, b, repeats=100)
    assert report["paired_delta"] == pytest.approx(.1)
    assert report["improved_owners"] == 2 and report["unchanged_owners"] == 1
    assert report["owners"] == 3
    same, _ = paired_summary(a, a.reverse(), repeats=100)
    assert same["owner_bootstrap_95pct"] == [0., 0.]
    with pytest.raises(ValueError, match="identical"):
        paired_summary(a, b.head(2), repeats=100)
