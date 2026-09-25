"""Run the measured experiment on another machine with explicit paths and logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "views", "retrieve", "budget", "features", "train", "support")
RECORD_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry"
LABEL_HEADER = "source1_entity_id\tmatched_entity_ids"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--work-root", type=Path, default=ROOT / "work/server_v1")
    p.add_argument("--profile", choices=("pilot", "full"), default="full")
    p.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--max-rounds", type=int, default=6000)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--start-at", choices=STAGES, default="prepare")
    p.add_argument("--stop-after", choices=STAGES, default="train")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def input_files(dataset: Path, include_test: bool):
    files = [(dataset / "train" / f"train_source{s}.tsv", RECORD_HEADER) for s in (1, 2, 3)]
    files.append((dataset / "train/train_ground_truth.tsv", LABEL_HEADER))
    if include_test:
        files += [(dataset / "test" / f"test_source{s}.tsv", RECORD_HEADER) for s in (1, 2, 3)]
    return files


def preflight(args):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use Python 3.12 for this pinned release.")
    if args.threads < 1 or args.max_rounds < 1:
        raise ValueError("Threads and maximum rounds must be positive.")
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise ValueError("start-at must precede stop-after.")
    for path, header in input_files(args.dataset, args.include_test):
        with path.open(encoding="utf-8") as f:
            if f.readline().rstrip("\r\n") != header:
                raise ValueError(f"Unexpected TSV header: {path}")
    dependencies = {}
    for line in (ROOT / "plan3/requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, required = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != required:
            raise ValueError(f"{name}: expected {required}, installed {actual}")
        dependencies[name] = actual
    existing = args.work_root
    while not existing.exists():
        existing = existing.parent
    memory = None
    if hasattr(os, "sysconf"):
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (OSError, ValueError):
            pass
    return {"platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "physical_ram_gib": memory,
            "free_disk_gib": round(shutil.disk_usage(existing).free / 2**30, 1),
            "dependencies": dependencies}


def commands(args):
    base = [sys.executable, "-u", "-m", "plan3"]
    prepared = args.work_root / "prepared"
    views = args.work_root / "views"
    indexes = args.work_root / "indexes"
    run = args.work_root / "runs" / (args.profile + "-w100-m25")
    p = ["--prepared", str(prepared)]
    r = ["--run", str(run)]
    return run, [
        ("prepare", base + ["prepare", "--dataset", str(args.dataset)] + p +
         (["--include-test"] if args.include_test else [])),
        ("views", base + ["views"] + p + ["--out", str(views)]),
        ("retrieve", base + ["retrieve"] + p + r + ["--indexes", str(indexes),
         "--profile", args.profile, "--channels", "word,missing", "--threads", str(args.threads),
         "--query-batch", "1000"]),
        ("budget", base + ["budget"] + p + r),
        ("features", base + ["features"] + p + r + ["--views", str(views), "--indexes", str(indexes),
         "--word-k", "100", "--missing-k", "25", "--query-batch", "250"]),
        ("train", base + ["train"] + p + r + ["--threads", str(args.threads),
         "--max-rounds", str(args.max_rounds)]),
        ("support", base + ["support"] + p + r + ["--threads", str(args.threads),
         "--max-rounds", str(args.max_rounds)]),
    ]


def run_stage(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nSTART " + datetime.now(timezone.utc).isoformat() + "\n")
        log.write(json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Stage exited {code}; see {log_path}")


def main(argv=None):
    args = parser().parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.work_root = args.work_root.resolve()
    host = preflight(args)
    print(json.dumps(host, indent=2), flush=True)
    run, plan = commands(args)
    chosen = plan[STAGES.index(args.start_at):STAGES.index(args.stop_after) + 1]
    for stage, command in chosen:
        print(f"{stage}: {json.dumps(command)}", flush=True)
    if args.check_only or args.dry_run:
        return
    # Thread limits must be set before the first Polars import.
    limits = {"POLARS_MAX_THREADS": str(min(args.threads, 8)), "OMP_NUM_THREADS": str(args.threads),
              "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    os.environ.update(limits)
    from .artifacts import contract, source_hashes, write_json
    spec = {"dataset": str(args.dataset), "profile": args.profile, "threads": args.threads,
            "max_rounds": args.max_rounds, "include_test": args.include_test,
            "word_k": 100, "missing_k": 25, "dependencies": host["dependencies"],
            "plan1_code": source_hashes("plan1"), "plan3_code": source_hashes("plan3")}
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "server_running.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"Run already locked: {lock}. Check its process before removing a stale lock.") from None
    try:
        contract(run / "server_contract.json", spec)
        write_json(run / "server_host.json", host)
        for stage, command in chosen:
            status = {"stage": stage, "status": "running", "command": command,
                      "started_at": datetime.now(timezone.utc).isoformat()}
            write_json(run / "server_status.json", status)
            try:
                run_stage(command, run / "server_logs" / (stage + ".log"), dict(os.environ))
            except BaseException as e:
                write_json(run / "server_status.json", {**status, "status": "failed", "error": str(e)})
                raise
            write_json(run / "server_status.json", {**status, "status": "complete",
                                                    "finished_at": datetime.now(timezone.utc).isoformat()})
    finally:
        lock.unlink(missing_ok=True)
    print(f"Completed through {args.stop_after}. Results: {run}", flush=True)


if __name__ == "__main__":
    main()
