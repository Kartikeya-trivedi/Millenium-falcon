import polars as pl

from plan3.onehop_probe import select_seeds


def test_onehop_seed_selection_ignores_labels_and_duplicate_evidence():
    frame = pl.DataFrame({"s1_id": ["q"]*4, "target_id": ["c", "a", "b", "d"],
        "country": ["X"]*4, "source": ["S2", "S3", "S2", "S3"], "p": [.99, .99, .98, .8],
        "t_name": ["one", "one", "two", "three"], "t_address": ["road"]*4, "label": [1, 0, 1, 1]})
    before = select_seeds(frame, .95)
    after = select_seeds(frame.with_columns(label=1-pl.col("label")), .95)
    assert before.equals(after)
    assert before["target_id"].to_list() == ["a", "b"]
    assert "label" not in before.columns
