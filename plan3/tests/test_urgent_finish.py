from pathlib import Path
import pytest
from plan3 import urgent_finish as u
from plan3.artifacts import read_json, write_json, sha256
from plan3.tests.test_finish import evidence


@pytest.mark.parametrize("validator_fails", [False, True])
def test_test_outputs_precede_audit_and_validation_gates_completion(tmp_path, monkeypatch, validator_fails):
    work = tmp_path / "work"
    run = work / "runs/full-w100-m25"
    names = ["direct_report.json", "support_report.json", "comparisons/direct-vs-support/report.json",
             "comparisons/support_control-vs-support/report.json"]
    for name, value in zip(names, evidence()):
        write_json(run / name, value)
    write_json(work / "test_assets/status.json", {"status": "complete"})
    monkeypatch.setattr(u, "load_prepared", lambda p: {"identity": "fixture", "dataset": str(tmp_path / "dataset")})
    validator = tmp_path / "validator.py"
    validator.write_text("fixture")
    stages = []
    def execute(cmd, log, env):
        stages.append(log.stem)
        if log.stem == "supplied_validator":
            assert "--check-ids" in cmd
            if validator_fails:
                raise RuntimeError("invalid output")
    def publish(inference, raw, folder):
        stages.append("export")
        write_json(folder / "output/export_complete.json", {"verified": True})
    monkeypatch.setattr(u, "run_stage", execute)
    monkeypatch.setattr(u, "publish_output", publish)
    if validator_fails:
        with pytest.raises(RuntimeError, match="invalid output"):
            u.finish(work, validator, sha256(validator))
        assert not (work / "urgent/complete.json").exists()
    else:
        result = u.finish(work, validator, sha256(validator))
        assert result["audit_status"] == "pending"
        assert result["archive_sha256"] == sha256(Path(result["archive"]))
    assert stages == ["test_inference", "export", "supplied_validator"]


def direct_only(tmp_path, monkeypatch):
    work = tmp_path / "work"
    run = work / "runs/full-w100-m25"
    write_json(run / "direct_report.json", {"select": {"after_ownership": {"macro_f05": .95}}})
    write_json(run / "direct_decision.json", {"threshold": .8, "selected_on": "select"})
    (run / "direct.txt").write_text("fixture model")
    write_json(run / "direct_meta.json", {"model_sha256": sha256(run / "direct.txt")})
    write_json(work / "test_assets/status.json", {"status": "complete"})
    monkeypatch.setattr(u, "load_prepared", lambda p: {"identity": "fixture", "dataset": str(tmp_path / "dataset")})
    validated = []
    def models(prepared, model_run, name):
        assert prepared == work / "prepared" and model_run == run and name == "direct"
        validated.append(name)
        return {"model": "direct", "model_sha256": {"direct": sha256(run / "direct.txt")},
                "decision": read_json(run / "direct_decision.json")}, object(), object()
    monkeypatch.setattr(u, "load_models", models)
    validator = tmp_path / "validator.py"
    validator.write_text("fixture validator")
    return work, run, validator, validated


@pytest.mark.parametrize("validator_fails", [False, True])
def test_direct_first_needs_no_support_artifacts_and_still_requires_validation(tmp_path, monkeypatch, validator_fails):
    work, run, validator, validated = direct_only(tmp_path, monkeypatch)
    stages = []
    def execute(command, log, env):
        stages.append(log.stem)
        if log.stem == "test_inference":
            assert command[command.index("--model") + 1] == "direct"
            assert command[command.index("--split") + 1] == "test"
            assert "urgent-direct" in command[command.index("--out") + 1]
        elif log.stem == "supplied_validator":
            assert "--check-ids" in command
            if validator_fails:
                raise RuntimeError("validation failed")
    def publish(inference, raw, folder):
        stages.append("export")
        assert folder == work / "urgent-direct"
        write_json(folder / "output/export_complete.json", {"verified": True})
    monkeypatch.setattr(u, "run_stage", execute)
    monkeypatch.setattr(u, "publish_output", publish)
    if validator_fails:
        with pytest.raises(RuntimeError, match="validation failed"):
            u.finish(work, validator, sha256(validator), direct_first=True)
        assert not (work / "urgent-direct/complete.json").exists()
    else:
        result = u.finish(work, validator, sha256(validator), direct_first=True)
        assert result["model"] == "direct" and result["audit_status"] == "pending"
        assert result["archive_sha256"] == sha256(Path(result["archive"]))
    choice = read_json(work / "urgent-direct/selection.json")
    assert choice["mode"] == "direct-first" and choice["model_comparison_performed"] is False
    assert set(choice["evidence_sha256"]) == {
        "direct_report.json", "direct_decision.json", "direct_meta.json", "direct.txt"}
    assert choice["frozen"]["decision"]["threshold"] == .8
    assert validated == ["direct"]
    assert stages == ["test_inference", "export", "supplied_validator"]
    assert not (run / "support.txt").exists() and not (work / "urgent").exists()


def test_direct_first_rejects_missing_decision_before_inference(tmp_path, monkeypatch):
    work, run, validator, validated = direct_only(tmp_path, monkeypatch)
    (run / "direct_decision.json").unlink()
    monkeypatch.setattr(u, "run_stage", lambda *args: pytest.fail("Inference must not start"))
    with pytest.raises(ValueError, match="direct_decision.json"):
        u.finish(work, validator, sha256(validator), direct_first=True)
    assert validated == []
    assert not (work / "urgent-direct/complete.json").exists()


def test_direct_first_requires_model_provenance_validation(tmp_path, monkeypatch):
    work, run, validator, _ = direct_only(tmp_path, monkeypatch)
    def invalid(*args):
        raise ValueError("Direct model hash mismatch")
    monkeypatch.setattr(u, "load_models", invalid)
    monkeypatch.setattr(u, "run_stage", lambda *args: pytest.fail("Inference must not start"))
    with pytest.raises(ValueError, match="model hash mismatch"):
        u.finish(work, validator, sha256(validator), direct_first=True)
    assert not (work / "urgent-direct/selection.json").exists()
