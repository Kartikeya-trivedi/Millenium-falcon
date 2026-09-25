"""Continue a stopped full run with more CPU, preserving its retrieval snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from .artifacts import contract, load_prepared, read_json, sha256, source_hashes, write_json
from .finish import worker_lock
from .server import run_stage


def stage_commands(work: Path, threads: int, workers: int, retrieval: dict):
    prepared, run = work / "prepared", work / "runs/full-w100-m25"
    base = [sys.executable, "-u", "-m"]
    paths = ["--prepared", str(prepared), "--run", str(run)]
    return [
        # All expensive channel searches are already committed. Preserve their
        # exact original contract; Polars can use the larger current CPU pool
        # for the unfinished union and report generation.
        ("retrieve", base + ["plan3", "retrieve", *paths, "--indexes", str(work / "indexes"),
            "--profile", "full", "--channels", ",".join(retrieval["budgets"]),
            "--threads", str(retrieval["threads"]), "--query-batch", str(retrieval["query_batch"]),
            "--shard-size", str(retrieval["shard_size"])]),
        ("budget", base + ["plan3", "budget", *paths]),
        ("features", base + ["plan3", "features", *paths, "--views", str(work / "views"),
            "--indexes", str(work / "indexes"), "--word-k", "100", "--missing-k", "25",
            "--query-batch", "250", "--workers", str(workers), "--worker-threads", "1"]),
        ("train", base + ["plan3", "train", *paths, "--threads", str(threads)]),
        ("support", base + ["plan3", "support", *paths, "--threads", str(threads),
            "--workers", str(workers), "--worker-threads", "1"]),
        ("control", base + ["plan3.ablate_support", *paths, "--threads", str(threads)]),
        ("compare_direct", base + ["plan3.compare", *paths, "--before", "direct", "--after", "support"]),
        ("compare_control", base + ["plan3.compare", *paths, "--before", "support_control", "--after", "support"]),
        ("diagnose_direct", base + ["plan3.diagnose", *paths, "--model", "direct"]),
        ("diagnose_support", base + ["plan3.diagnose", *paths, "--model", "support"]),
    ]


def accelerate(work: Path, threads=64, workers=64):
    if threads < 1 or not 1 <= workers <= threads:
        raise ValueError("Positive CPU threads and no more workers than threads required")
    prepared, run = work / "prepared", work / "runs/full-w100-m25"
    meta = load_prepared(prepared)
    retrieval = read_json(run / "retrieval_contract.json")
    from .artifacts import fingerprint
    from .retrieve import PROFILES
    if retrieval["sizes"] != PROFILES["full"] or retrieval["prepared_identity"] != meta["identity"]:
        raise ValueError("Expected the unchanged full training population and prepared data")
    expected_channels = {f"{fingerprint(country)[:16]}-{source}-{channel}.parquet"
        for country in ("India", "US") for source in ("S2", "S3") for channel in retrieval["budgets"]}
    if {p.name for p in (run / "channels").glob("*.parquet")} != expected_channels:
        raise ValueError("All original channel searches must be committed before CPU continuation")
    with worker_lock(work / "accelerated-training"):
        spec = {"threads": threads, "feature_workers": workers, "prepared_identity": meta["identity"],
                "original_server_contract_sha256": sha256(run / "server_contract.json"),
                "retrieval_contract_sha256": sha256(run / "retrieval_contract.json"),
                "plan1_code": source_hashes("plan1"), "plan3_code": source_hashes("plan3"),
                "candidate_searches_reused": sorted(expected_channels)}
        contract(run / "accelerated_contract.json", spec)
        for stage, command in stage_commands(work, threads, workers, retrieval):
            status = {"stage": stage, "status": "running", "cpu_threads": threads,
                      "feature_workers": workers, "started_at": datetime.now(timezone.utc).isoformat()}
            write_json(run / "accelerated_status.json", status)
            try:
                run_stage(command, run / "accelerated_logs" / (stage + ".log"), dict(os.environ))
            except BaseException as error:
                write_json(run / "accelerated_status.json", {**status, "status": "failed", "error": str(error)})
                raise
            write_json(run / "accelerated_status.json", {**status, "status": "complete"})
        write_json(run / "accelerated_complete.json", {"status": "complete", "threads": threads,
            "feature_workers": workers, "finished_at": datetime.now(timezone.utc).isoformat()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    accelerate(a.work_root, a.threads, a.workers)


if __name__ == "__main__":
    main()
