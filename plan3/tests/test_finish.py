from copy import deepcopy
import pytest

from plan3.finish import publish_output, select_model, worker_lock


def evidence():
    direct = {"select": {"after_ownership": {"macro_f05": .95}}}
    support = {"select": {"after_ownership": {"macro_f05": .96}}}
    comparison = {"development": {"owner_bootstrap_95pct": [.003, .012],
                    "by_country": [{"country": "India", "delta": .006}, {"country": "US", "delta": .01}]}}
    return direct, support, comparison, deepcopy(comparison)


def test_support_requires_benefit_over_direct_and_retraining_control():
    args = evidence()
    assert select_model(*args)["model"] == "support"
    args[-1]["development"]["owner_bootstrap_95pct"][0] = -.001
    assert select_model(*args)["model"] == "direct"


def test_country_regression_or_select_tie_keeps_simpler_model():
    args = evidence()
    args[2]["development"]["by_country"][0]["delta"] = -.001
    assert select_model(*args)["model"] == "direct"
    args = evidence()
    args[1]["select"]["after_ownership"]["macro_f05"] = .95
    assert select_model(*args)["model"] == "direct"


def test_worker_lock_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="interrupted"):
        with worker_lock(tmp_path):
            with pytest.raises(RuntimeError, match="already active"):
                with worker_lock(tmp_path):
                    pass
            raise RuntimeError("interrupted")
    with worker_lock(tmp_path):
        pass


def test_interrupted_export_is_not_published_and_retry_is_independent(tmp_path, monkeypatch):
    from pathlib import Path
    from plan3.artifacts import read_json
    from plan3.export import export, verify_files
    from plan3.tests.test_export import make_inputs

    inference, raw, _ = make_inputs(tmp_path)
    folder = tmp_path / "final"
    folder.mkdir()
    attempted = []
    def run(command, log, env):
        output = Path(command[command.index("--out") + 1])
        attempted.append(output)
        if len(attempted) == 1:
            output.mkdir()
            (output / "candidate_pairs.tsv").write_text("interrupted")
            raise RuntimeError("interrupted")
        export(inference, raw, output)
    monkeypatch.setattr("plan3.finish.run_stage", run)
    with pytest.raises(RuntimeError, match="interrupted"):
        publish_output(inference, raw, folder)
    assert not (folder / "output").exists()
    report = publish_output(inference, raw, folder)
    assert attempted[0] != attempted[1]
    assert verify_files(folder / "output") == report
    assert read_json(folder / "output/export_complete.json") == report
    assert publish_output(inference, raw, folder) == report
    assert len(attempted) == 2
