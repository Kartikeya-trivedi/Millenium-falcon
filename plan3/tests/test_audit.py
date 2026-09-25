import polars as pl
import pytest

from plan3.audit import evaluation_population, clustered_intervals


def test_audit_rejects_training_claimants_and_incomplete_or_exposed_panels():
    manifest = pl.DataFrame({"s1_id": ["fit", "support", "tune", "fresh", "fresh2"],
        "role": ["fit", "calibrate", "tune", "audit", "audit"],
        "pool": ["fit", "c_prob", "tune", "audit", "audit"], "country": ["India"] * 5})
    fresh = manifest.filter(pl.col("s1_id").str.starts_with("fresh"))
    def roster(*ids):
        return manifest.filter(pl.col("s1_id").is_in(ids)).select("s1_id", "country")
    with pytest.raises(ValueError, match="Fit/dictionary"):
        evaluation_population(roster("fit", "fresh", "fresh2"), manifest, fresh, "direct", "fresh-audit")
    with pytest.raises(ValueError, match="Support training"):
        evaluation_population(roster("support"), manifest, fresh, "support", "development")
    with pytest.raises(ValueError, match="excluded from development"):
        evaluation_population(roster("fresh"), manifest, fresh, "direct", "development")
    with pytest.raises(ValueError, match="every reserved owner"):
        evaluation_population(roster("fresh"), manifest, fresh, "direct", "fresh-audit")
    assert set(evaluation_population(roster("tune", "fresh", "fresh2"), manifest, fresh,
                                    "support", "fresh-audit")) == {"fresh", "fresh2"}


def test_clustered_recall_handles_empty_owners_without_treating_links_as_independent():
    per = pl.DataFrame({"f3": [1., 1.], "n_true": [10, 0], "n_retrieved": [10, 0]})
    result = clustered_intervals(per, repeats=100)
    assert result["macro_f05_95pct"] == [1, 1]
    assert result["candidate_recall_95pct"] == [1, 1]
    none = clustered_intervals(per.with_columns(n_true=pl.lit(0), n_retrieved=pl.lit(0)), repeats=100)
    assert none["candidate_recall_95pct"] is None
