"""Deadline-oriented test-output worker with a separate inexpensive coordinator."""
from pathlib import Path
import modal
from .modal_app import image, volume

app = modal.App("millenium-falcon-p3-urgent")
deadline = "2026-09-25T19:25:12+00:00"


def within_budget():
    from datetime import datetime, timezone
    if datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        raise RuntimeError("Cloud spending cutoff reached")


@app.function(image=image.env({"POLARS_MAX_THREADS": "4", "OMP_NUM_THREADS": "4"}),
              volumes={"/vol": volume}, cpu=(64, 64), memory=(196608, 262144),
              max_containers=1, timeout=14400, retries=0, scaledown_window=2)
def produce(run_id: str, validator_sha: str, direct_first: bool = False):
    import json
    import subprocess
    within_budget()
    if not run_id.replace("-", "").isalnum() or len(validator_sha) != 64 or any(c not in "0123456789abcdef" for c in validator_sha):
        raise ValueError("Invalid artifact identifiers")
    volume.reload()
    work = Path("/vol/runs") / run_id
    validator = Path("/vol/utilities") / validator_sha / "validate_submission.py"
    try:
        command = ["/opt/falcon/.venv/bin/python", "-u", "-m", "plan3.urgent_finish",
                        "--work-root", str(work), "--validator", str(validator),
                        "--validator-sha", validator_sha, "--workers", "16", "--threads", "4"]
        if direct_first:
            command.append("--direct-first")
        subprocess.run(command, check=True)
        folder = "urgent-direct" if direct_first else "urgent"
        return json.loads((work / folder / "complete.json").read_text())
    finally:
        volume.commit()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=0.125,
              memory=256, timeout=14400, retries=0, max_containers=1, scaledown_window=2)
def coordinate(training_call: str, run_id: str, validator_sha: str):
    import time
    within_budget()
    call = modal.FunctionCall.from_id(training_call)
    while True:
        within_budget()
        try:
            result = call.get(timeout=30)
            break
        except TimeoutError:
            continue
    child = modal.FunctionCall.from_id(result["finish_call_id"])
    child.cancel(terminate_containers=True)
    for _ in range(30):
        graph = child.get_call_graph()
        if graph and all(n.status.name in ("TERMINATED", "SUCCESS", "FAILURE", "INIT_FAILURE", "TIMEOUT") for n in graph):
            break
        time.sleep(2)
    else:
        raise RuntimeError("Prior finish worker did not terminate")
    within_budget()
    new_call = produce.spawn(run_id, validator_sha)
    return {"status": "submitted", "finish_call_id": new_call.object_id,
            "cancelled_finish_call_id": child.object_id, "budget_deadline": deadline}


def direct_ready(run):
    """Report publication follows scoring, threshold selection and panel reports."""
    import hashlib
    import json
    names = ("direct_report.json", "direct_decision.json", "direct_meta.json", "direct.txt")
    if not all((run / name).is_file() for name in names):
        return None
    report = json.loads((run / names[0]).read_text())
    decision = json.loads((run / names[1]).read_text())
    meta = json.loads((run / names[2]).read_text())
    if not all(panel in report for panel in ("tune", "select", "development")):
        raise ValueError("Direct evaluation is incomplete")
    if decision.get("selected_on") != "select":
        raise ValueError("Direct threshold was not selected on Select")
    hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in names}
    if hashes["direct.txt"] != meta["model_sha256"]:
        raise ValueError("Direct model identity mismatch")
    return hashes


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/vol": volume},
              cpu=0.125, memory=256, timeout=14400, retries=0, max_containers=1, scaledown_window=2)
def coordinate_direct(training_call: str, run_id: str, validator_sha: str):
    import json
    import time
    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    work = Path("/vol/runs") / run_id
    run = work / "runs/full-w100-m25"
    call = modal.FunctionCall.from_id(training_call)
    terminal = {"TERMINATED", "SUCCESS", "FAILURE", "INIT_FAILURE", "TIMEOUT"}
    print("Waiting for complete direct evaluation; support/control are deferred.", flush=True)
    while True:
        within_budget()
        volume.reload()
        hashes = direct_ready(run)
        if hashes is not None:
            break
        graph = call.get_call_graph()
        if graph and all(n.status.name in terminal for n in graph):
            raise RuntimeError("Training ended without a complete direct checkpoint")
        time.sleep(15)
    print("Direct checkpoint verified. Stopping later training stages before test inference.", flush=True)
    call.cancel(terminate_containers=True)
    for _ in range(60):
        graph = call.get_call_graph()
        if graph and all(n.status.name in terminal for n in graph):
            break
        time.sleep(2)
    else:
        raise RuntimeError("Training worker did not terminate; refusing concurrent heavy jobs")
    volume.reload()
    if direct_ready(run) != hashes:
        raise ValueError("Direct artifacts changed during worker handoff")
    within_budget()
    child = produce.spawn(run_id, validator_sha, True)
    result = {"status": "submitted", "finish_call_id": child.object_id,
              "cancelled_training_call_id": training_call, "direct_artifacts": hashes,
              "budget_deadline": deadline, "model": "direct"}
    (work / "direct_finish_queue.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps(result), flush=True)
    return result
