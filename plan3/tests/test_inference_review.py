"""Regression coverage for inference artifact reuse."""
import polars as pl
import pytest

import plan3.inference as inference
from plan3.artifacts import fingerprint, read_json, sha256, write_json


def test_completed_inference_rejects_modified_score_shard(tmp_path, monkeypatch):
    prepared, views, run, out = [tmp_path / name for name in ("prepared", "views", "run", "out")]
    for directory in (prepared, views, run, out / "scores"):
        directory.mkdir(parents=True)
    source = prepared / "source1.parquet"
    pl.DataFrame({"entity_id": ["S1-one"], "country": ["US"]}).write_parquet(source)
    meta = {"identity": "fixture", "inputs": {},
            "partitions": [{"split": "test", "source": "S1", "country": "US", "file": source.name}]}
    view = {"prepared_identity": "fixture",
            "views_code": sha256(inference.Path(inference.__file__).with_name("views.py"))}
    write_json(views / "train_contract.json", view)
    write_json(views / "test_contract.json", view)
    write_json(run / "feature_contract.json", {"views_contract": sha256(views / "train_contract.json")})
    pl.DataFrame(schema={"name_red": pl.String, "country": pl.String,
                         "reference_name_count": pl.UInt32}).write_parquet(views / "test_name_frequency.parquet")
    monkeypatch.setattr(inference, "load_prepared", lambda path: meta)
    monkeypatch.setattr(inference, "load_models",
                        lambda *args: ({"prepared_identity": "fixture", "seed_threshold": None,
                            "decision": {"selected_on": "select", "threshold": .8}}, None, None))
    shard = out / "scores" / f"{fingerprint('US')[:16]}-00000000.parquet"
    frame = pl.DataFrame([("S1-one", "S2-one", "S2", "US", .9, .9, .5, 1)],
                         schema=inference.SCORE_SCHEMA, orient="row")
    frame.write_parquet(shard)
    write_json(shard.with_suffix(".json"), {"query_ids": fingerprint(["S1-one"]),
        "sha256": sha256(shard), "rows": frame.height})
    inference.infer(prepared, views, tmp_path / "indexes", run, out, model_name="direct")
    original = read_json(out / "inference_complete.json")["score_shards"][shard.name]["sha256"]
    frame.with_columns(p=pl.lit(.1)).write_parquet(shard)
    assert sha256(shard) != original
    with pytest.raises(ValueError, match="hash|changed|modified|mismatch"):
        inference.infer(prepared, views, tmp_path / "indexes", run, out, model_name="direct")
