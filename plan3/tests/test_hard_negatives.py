import polars as pl
import pytest
from plan3.hard_negatives import mine


def test_hard_mining_uses_model_errors_and_preserves_full_weight_mass():
    frame = pl.DataFrame({"s1_id": ["q"]*102, "target_id": [f"t{i}" for i in range(102)],
                           "label": [1, 1]+[0]*100, "p1": [i/102 for i in range(102)]})
    chosen = mine(frame)
    assert chosen["label"].sum() == 2
    assert chosen.height == 34
    assert chosen["weight"].sum() == pytest.approx(102)
    fixed = chosen.filter((pl.col("label") == 0) & (pl.col("weight") == 1))
    assert set(fixed["target_id"]) == {f"t{i}" for i in range(94, 102)}
    assert chosen.equals(mine(frame))
