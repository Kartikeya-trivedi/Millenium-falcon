import polars as pl
import pytest

from plan3.orphan_weighting import reweight_unowned, validate_populations
from plan3.train import sample_negatives


def test_only_unowned_negative_weights_change_after_identical_sampling():
    frame = pl.DataFrame({
        "s1_id": ["q"] * 102, "target_id": [f"t{i}" for i in range(102)],
        "label": [1, 1] + [0] * 100,
        "name_red_ratio": [i / 102 for i in range(102)],
        "addr_token_set": [0.] * 102, "retrieval_score": [0.] * 102,
    })
    sampled = sample_negatives(frame)
    owned = pl.Series([f"t{i}" for i in range(0, 102, 2)] + ["t1"])
    weighted, counts = reweight_unowned(sampled, owned)
    assert weighted.drop("weight").equals(sampled.drop("weight"))
    assert "_unowned" not in weighted.columns
    for old, new in zip(sampled.iter_rows(named=True), weighted.iter_rows(named=True)):
        multiplier = 2 if old["label"] == 0 and old["target_id"] not in owned else 1
        assert new["weight"] == old["weight"] * multiplier
    assert counts["sampled_positive_pairs"] == 2
    assert counts["sampled_negative_pairs"] == 32
    assert counts["base_positive_weight"] == 2
    assert counts["base_negative_weight"] == pytest.approx(100)
    assert counts["adjusted_unowned_negative_weight"] == 2 * counts["base_unowned_negative_weight"]
    assert counts["adjusted_owned_negative_weight"] == counts["base_owned_negative_weight"]


def test_missing_positive_owner_cannot_silently_become_an_orphan():
    frame = pl.DataFrame({"target_id": ["positive"], "label": [1], "weight": [1.]})
    with pytest.raises(ValueError, match="positive training target"):
        reweight_unowned(frame, pl.Series(["other"]))


def test_population_validation_rejects_overlap_and_fresh_audit():
    samples = pl.DataFrame({"s1_id": ["a", "b", "c", "d", "e"],
                           "sample": ["fit", "support", "tune", "select", "development"]})
    manifest = pl.DataFrame({"s1_id": ["a", "b", "c", "d", "e"],
                            "pool": ["fit", "c_prob", "tune", "c_select", "audit"]})
    protected = pl.DataFrame({"s1_id": ["fresh"]})
    assert validate_populations(samples, manifest, protected).height == 5
    with pytest.raises(ValueError, match="unique across roles"):
        validate_populations(pl.concat([samples, samples.head(1)]), manifest, protected)
    with pytest.raises(ValueError, match="Fresh Audit"):
        validate_populations(samples, manifest, pl.DataFrame({"s1_id": ["e"]}))
    wrong = manifest.with_columns(pool=pl.when(pl.col("s1_id") == "b")
                                  .then(pl.lit("fit")).otherwise(pl.col("pool")))
    with pytest.raises(ValueError, match="support population"):
        validate_populations(samples, wrong, protected)
