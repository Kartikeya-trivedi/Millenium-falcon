"""Produce validated test outputs before the separate held-out evaluation."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from .artifacts import contract, load_prepared, read_json, sha256, write_json
from .finish import select_model, publish_output, worker_lock
from .server import run_stage


def finish(work, validator, validator_sha, workers=16, threads=4):
    if sha256(validator) != validator_sha:
        raise ValueError("Validator identity mismatch")
    with worker_lock(work):
        prepared = work / "prepared"
        meta = load_prepared(prepared)
        if read_json(work / "test_assets/status.json")["status"] != "complete":
            raise ValueError("Test assets are incomplete")
        run = work / "runs/full-w100-m25"
        evidence = [run / "direct_report.json", run / "support_report.json",
                    run / "comparisons/direct-vs-support/report.json",
                    run / "comparisons/support_control-vs-support/report.json"]
        choice = select_model(*(read_json(p) for p in evidence))
        choice["evidence_sha256"] = {p.relative_to(run).as_posix(): sha256(p) for p in evidence}
        folder = work / "urgent"
        contract(folder / "selection.json", choice)
        contract(folder / "finish_contract.json", {"selection_sha256": sha256(folder / "selection.json"),
            "validator_sha256": validator_sha, "code": sha256(Path(__file__)),
            "workers": workers, "threads": threads, "prepared_identity": meta["identity"]})
        dataset = Path(meta["dataset"])
        stages = [
            ("test_inference", [sys.executable, "-u", "-m", "plan3.parallel_inference",
                "--prepared", str(prepared), "--views", str(work / "views"),
                "--indexes", str(work / "indexes"), "--model-run", str(run),
                "--model", choice["model"], "--split", "test", "--out", str(folder / "test-inference"),
                "--workers", str(workers), "--threads", str(threads)]),
            ("export", None),
            ("supplied_validator", [sys.executable, str(validator),
                "--matching", str(folder / "output/matching_results.tsv"),
                "--candidate", str(folder / "output/candidate_pairs.tsv"),
                "--test-dir", str(dataset / "test"), "--check-ids"]),
        ]
        for stage, command in stages:
            status = {"stage": stage, "status": "running", "model": choice["model"],
                      "started_at": datetime.now(timezone.utc).isoformat()}
            write_json(folder / "status.json", status)
            try:
                if command is None:
                    publish_output(folder / "test-inference", dataset / "test", folder)
                else:
                    run_stage(command, folder / "logs" / (stage + ".log"), dict(os.environ))
            except BaseException as error:
                write_json(folder / "status.json", {**status, "status": "failed", "error": str(error)})
                raise
        archive = folder / "outputs.zip"
        temp = archive.with_suffix(".tmp")
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
            for path in [*sorted((folder / "output").glob("*")), folder / "selection.json"]:
                z.write(path, path.relative_to(folder).as_posix())
        temp.replace(archive)
        result = {"status": "complete", "model": choice["model"], "audit_status": "pending",
                  "outputs": read_json(folder / "output/export_complete.json"),
                  "archive": str(archive), "archive_sha256": sha256(archive),
                  "finished_at": datetime.now(timezone.utc).isoformat()}
        write_json(folder / "complete.json", result)
        write_json(folder / "status.json", {"stage": "final", "status": "complete", "model": choice["model"]})
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--validator", type=Path, required=True)
    p.add_argument("--validator-sha", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    finish(a.work_root, a.validator, a.validator_sha, a.workers, a.threads)
