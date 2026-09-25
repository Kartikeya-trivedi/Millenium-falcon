import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from plan3.collect import collect
from plan3.server import commands, input_files, parser, preflight, run_stage


def test_server_passes_all_artifact_paths_and_bounded_budget(tmp_path):
    args = parser().parse_args(["--dataset", str(tmp_path / "raw data"),
                                "--work-root", str(tmp_path / "work area"), "--threads", "8"])
    run, plan = commands(args)
    assert run == tmp_path / "work area/runs/full-w100-m25"
    for stage, command in plan:
        assert command[command.index("--prepared") + 1] == str(tmp_path / "work area/prepared")
        assert command[:4] == [sys.executable, "-u", "-m", "plan3"]
    features = dict(plan)["features"]
    assert features[features.index("--word-k") + 1] == "100"
    assert features[features.index("--missing-k") + 1] == "25"
    assert "--include-test" not in dict(plan)["prepare"]


def test_server_preflight_rejects_wrong_headers_and_bad_stage_order(tmp_path):
    args = parser().parse_args(["--dataset", str(tmp_path), "--work-root", str(tmp_path / "work")])
    for path, header in input_files(tmp_path, False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n", encoding="utf-8")
    assert preflight(args)["dependencies"]["polars"]
    (tmp_path / "train/train_source1.tsv").write_text("wrong,header\n")
    with pytest.raises(ValueError, match="Unexpected TSV header"):
        preflight(args)
    args.start_at, args.stop_after = "support", "train"
    with pytest.raises(ValueError, match="start-at"):
        preflight(args)


def test_server_logs_subprocess_failure(tmp_path):
    import os
    log = tmp_path / "stage.log"
    with pytest.raises(RuntimeError, match="exited 7"):
        run_stage([sys.executable, "-c", "print('stage failed'); raise SystemExit(7)"], log, dict(os.environ))
    assert "stage failed" in log.read_text()


def test_return_bundle_excludes_large_shards_and_never_overwrites(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "server_status.json").write_text(json.dumps({"status": "failed"}))
    (run / "direct_report.json").write_text("{}")
    (run / "direct.txt").write_text("model")
    (run / "queries.parquet").write_bytes(b"private raw query cache")
    (run / "features").mkdir()
    (run / "features/large.parquet").write_bytes(b"large")
    output = tmp_path / "return.zip"
    collect(run, output)
    with ZipFile(output) as z:
        assert set(z.namelist()) == {"server_status.json", "direct_report.json", "direct.txt"}
    with pytest.raises(FileExistsError):
        collect(run, output)
