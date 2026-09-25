from pathlib import Path
import pytest
from plan3 import urgent_finish as u
from plan3.artifacts import write_json, sha256
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
