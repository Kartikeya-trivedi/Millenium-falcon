import polars as pl
from polars.testing import assert_frame_equal
import pytest

from plan3.artifacts import read_json, write_json
from plan3 import support


def direct_frame(number, sample="support"):
    rows = []
    for owner in (f"q{number}-b", f"q{number}-a"):
        for target, source, score, name, address, numbers in [
            ("d", "S3", .97, "alpha", "12 road", ["12"]),
            ("a", "S2", .99, "alpha", "12 road", ["12"]),
            ("b", "S2", .98, "alpha", "12 road", ["12"]),
            ("c", "S2", .96, "beta", "14 road", ["14"]),
            ("e", "S3", .50, "other", "", []),
        ]:
            rows.append((owner, f"{source}-{number}-{target}", source, score, name, address, numbers,
                         sample, 1 if target == "a" else 0, .7))
    return pl.DataFrame(rows, schema={
        "s1_id": pl.String, "target_id": pl.String, "source": pl.String, "p": pl.Float64,
        "t_name": pl.String, "t_address": pl.String, "t_numbers": pl.List(pl.String),
        "sample": pl.String, "label": pl.Int8, "name_red_ratio": pl.Float32}, orient="row")


def test_support_processes_match_serial_exactly_without_losing_columns_or_order(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    for run in (serial, parallel):
        (run / "direct_scores").mkdir(parents=True)
        for number in range(3):
            direct_frame(number).write_parquet(run / "direct_scores" / f"{number}.parquet")
    before = list(support.generate_support_files(serial, .95))
    after = list(support.generate_support_files(parallel, .95, workers=2, worker_threads=1))
    assert sorted(before) == sorted(after) == [(f"{number}.parquet", 10) for number in range(3)]
    for name, _ in before:
        a, b = [pl.read_parquet(run / "support_features" / name) for run in (serial, parallel)]
        assert_frame_equal(a, b, check_exact=True)
        original = pl.read_parquet(serial / "direct_scores" / name)
        assert_frame_equal(b.select(original.columns), original, check_exact=True)
        expected = original.join(support.support_features(original, .95),
                                 on=["s1_id", "target_id"], how="left", maintain_order="left")
        assert_frame_equal(b, expected, check_exact=True)
    assert list(support.generate_support_files(parallel, .95, workers=2, worker_threads=1)) == []


def test_support_process_rejects_in_sample_upstream_predictions(tmp_path):
    (tmp_path / "direct_scores").mkdir()
    direct_frame(0, "fit").write_parquet(tmp_path / "direct_scores/0.parquet")
    with pytest.raises(ValueError, match="In-sample upstream predictions"):
        list(support.generate_support_files(tmp_path, .95, workers=2, worker_threads=1))
    assert not (tmp_path / "support_features/0.parquet").exists()


def test_support_contract_records_parallel_execution_configuration(tmp_path, monkeypatch):
    run, prepared = tmp_path / "run", tmp_path / "prepared"
    write_json(run / "direct_meta.json", {"sample": "fit", "model_sha256": "parent"})
    write_json(prepared / "translit.json", {"training_role": "fit"})
    monkeypatch.setattr(support, "choose_seed_threshold", lambda _: .95)
    monkeypatch.setattr(support, "fit_model", lambda *args: object())
    monkeypatch.setattr(support, "predict_model", lambda *args: None)
    monkeypatch.setattr(support, "evaluate_model", lambda *args: {"status": "fixture"})
    assert support.train_support(prepared, run, workers=2, worker_threads=1) == {"status": "fixture"}
    metadata = read_json(run / "support_features_contract.json")
    assert metadata["workers"] == 2 and metadata["worker_threads"] == 1
    assert len(metadata["batch_execution_code"]) == 64
