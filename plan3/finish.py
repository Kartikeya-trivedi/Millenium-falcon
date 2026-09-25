"""Select from completed development evidence, then evaluate and produce outputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl

from .artifacts import contract, fingerprint, load_prepared, parquet, read_json, sha256, write_json
from .server import run_stage


def select_model(direct, support, versus_direct, versus_control):
    """Conservative predeclared promotion rule; fresh Audit never participates."""
    checks = {
        "better_select_macro": support["select"]["after_ownership"]["macro_f05"] > direct["select"]["after_ownership"]["macro_f05"],
        "development_gain_over_direct": versus_direct["development"]["owner_bootstrap_95pct"][0] > 0,
        "relationship_gain_over_control": versus_control["development"]["owner_bootstrap_95pct"][0] > 0,
        "no_country_macro_regression": bool(versus_direct["development"]["by_country"]) and all(
            row["delta"] >= 0 for row in versus_direct["development"]["by_country"]),
    }
    return {"model": "support" if all(checks.values()) else "direct", "checks": checks,
            "selection_data": "Select and exposed development; no fresh Audit labels",
            "rule": "Retain support only with better Select macro F0.5, positive lower paired-owner-bootstrap bounds "
                    "against direct and retraining control, and no development country macro regression. Otherwise retain direct."}


@contextmanager
def worker_lock(work: Path):
    """Host-local OS lock, released on process death; Modal serializes containers."""
    path = Path(tempfile.gettempdir()) / ("falcon-finish-" + fingerprint(str(work.resolve())) + ".lock")
    with path.open("a+b") as handle:
        if not path.stat().st_size:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError("A finish worker is already active on this host") from error
        try:
            yield
        finally:
            release()


def publish_output(inference: Path, test_dir: Path, folder: Path):
    """Publish a complete directory; interrupted attempts remain separate."""
    from .export import verify_files

    output = folder / "output"
    if not output.exists():
        stage = folder / (".output-" + uuid4().hex)
        run_stage([sys.executable, "-u", "-m", "plan3.export", "--inference", str(inference),
                   "--test-dir", str(test_dir), "--out", str(stage)],
                  folder / "logs/export.log", dict(os.environ))
        verify_files(stage)
        stage.rename(output)
    report = verify_files(output)
    for name in ("inference_contract", "inference_complete"):
        if report[name + "_sha256"] != sha256(inference / (name + ".json")):
            raise ValueError("Published output belongs to a different inference run")
    return report


def finish(work: Path, validator: Path, validator_sha: str, threads=16):
    # Modal's max_containers=1 and default one input per container enforce the
    # cross-container limit. Do not launch CLI workers on different hosts for
    # the same work root. The OS lock also protects same-host CLI invocations.
    with worker_lock(work):
        return _finish(work, validator, validator_sha, threads)


def _finish(work: Path, validator: Path, validator_sha: str, threads):

    if sha256(validator) != validator_sha:
        raise ValueError("Supplied validator hash mismatch")
    prepared, run = work / "prepared", work / "runs/full-w100-m25"
    meta = load_prepared(prepared)
    if read_json(work / "test_assets/status.json")["status"] != "complete":
        raise ValueError("Test views/indexes are not complete")
    comparisons = run / "comparisons"
    choice = select_model(read_json(run / "direct_report.json"), read_json(run / "support_report.json"),
        read_json(comparisons / "direct-vs-support/report.json"),
        read_json(comparisons / "support_control-vs-support/report.json"))
    evidence = [run / name for name in ("direct_report.json", "support_report.json")]
    evidence += [comparisons / name / "report.json" for name in ("direct-vs-support", "support_control-vs-support")]
    choice["evidence_sha256"] = {str(p.relative_to(run)).replace("\\", "/"): sha256(p) for p in evidence}
    folder = work / "final"
    contract(folder / "selection.json", choice)
    contract(folder / "finish_contract.json", {"selection_sha256": sha256(folder / "selection.json"),
        "validator_sha256": validator_sha, "finish_code": sha256(Path(__file__)), "threads": threads,
        "prepared_identity": meta["identity"]})
    if (folder / "complete.json").exists():
        from .export import verify_files
        result = read_json(folder / "complete.json")
        if verify_files(folder / "output") != result["outputs"]:
            raise ValueError("Completed output manifest changed")
        if sha256(folder / "outputs.zip") != result["archive_sha256"]:
            raise ValueError("Completed output archive changed")
        if sha256(folder / "audit/report.json") != result["audit_report_sha256"]:
            raise ValueError("Completed Audit report changed")
        return result
    # The complete eligible claimant world excludes every owner used by the
    # dictionary/M0, plus the second-stage training pool when M1 is selected.
    manifest = pl.read_parquet(prepared / "manifest.parquet")
    world = manifest.filter(pl.col("role") != "fit")
    if choice["model"] == "support":
        world = world.filter(pl.col("pool") != "c_prob")
    parquet(folder / "held_out_world.parquet", world.select("s1_id").sort("s1_id"))
    base = [sys.executable, "-u", "-m"]
    infer = ["plan3.inference", "--prepared", str(prepared), "--views", str(work / "views"),
             "--indexes", str(work / "indexes"), "--model-run", str(run), "--model", choice["model"],
             "--threads", str(threads)]
    dataset = Path(meta["dataset"])
    stages = [
        ("held_out_world", base + infer + ["--split", "train", "--queries", str(folder / "held_out_world.parquet"),
                                            "--out", str(folder / "world-inference")]),
        ("fresh_audit", base + ["plan3.audit", "--prepared", str(prepared), "--inference", str(folder / "world-inference"),
                               "--out", str(folder / "audit"), "--panel", "fresh-audit"]),
        ("test_inference", base + infer + ["--split", "test", "--out", str(folder / "test-inference")]),
        ("export", None),
        ("supplied_validator", [sys.executable, str(validator), "--matching", str(folder / "output/matching_results.tsv"),
                               "--candidate", str(folder / "output/candidate_pairs.tsv"),
                               "--test-dir", str(dataset / "test"), "--check-ids"]),
    ]
    for stage, command in stages:
        status = {"stage": stage, "status": "running", "model": choice["model"],
                  "started_at": datetime.now(timezone.utc).isoformat()}
        write_json(folder / "status.json", status)
        try:
            if stage == "export":
                publish_output(folder / "test-inference", dataset / "test", folder)
            else:
                run_stage(command, folder / "logs" / (stage + ".log"), dict(os.environ))
        except BaseException as error:
            write_json(folder / "status.json", {**status, "status": "failed", "error": str(error)})
            raise
        write_json(folder / "status.json", {**status, "status": "complete"})
    archive = folder / "outputs.zip"
    temp = archive.with_suffix(".tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
        for path in [*sorted((folder / "output").glob("*")), folder / "selection.json", folder / "audit/report.json"]:
            z.write(path, path.relative_to(folder).as_posix())
    temp.replace(archive)
    result = {"status": "complete", "model": choice["model"], "claimant_owners": world.height,
              "audit": read_json(folder / "audit/report.json"),
              "audit_report_sha256": sha256(folder / "audit/report.json"),
              "outputs": read_json(folder / "output/export_complete.json"),
              "archive": str(archive), "archive_sha256": sha256(archive),
              "finished_at": datetime.now(timezone.utc).isoformat()}
    write_json(folder / "complete.json", result)
    write_json(folder / "status.json", {"stage": "final", "status": "complete", "model": choice["model"]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--validator", type=Path, required=True)
    p.add_argument("--validator-sha", required=True)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    finish(a.work_root, a.validator, a.validator_sha, a.threads)


if __name__ == "__main__":
    main()
