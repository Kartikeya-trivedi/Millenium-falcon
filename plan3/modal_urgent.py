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
def produce(run_id: str, validator_sha: str):
    import json
    import subprocess
    within_budget()
    if not run_id.replace("-", "").isalnum() or len(validator_sha) != 64 or any(c not in "0123456789abcdef" for c in validator_sha):
        raise ValueError("Invalid artifact identifiers")
    volume.reload()
    work = Path("/vol/runs") / run_id
    validator = Path("/vol/utilities") / validator_sha / "validate_submission.py"
    try:
        subprocess.run(["/opt/falcon/.venv/bin/python", "-u", "-m", "plan3.urgent_finish",
                        "--work-root", str(work), "--validator", str(validator),
                        "--validator-sha", validator_sha, "--workers", "16", "--threads", "4"], check=True)
        return json.loads((work / "urgent/complete.json").read_text())
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
