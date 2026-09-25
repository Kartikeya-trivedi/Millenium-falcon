"""Independent CPU preparation; it never changes a running training experiment."""
from pathlib import Path
import modal
from .modal_app import image, volume

app = modal.App("millenium-falcon-p3-followup")


@app.function(image=image, volumes={"/vol": volume}, cpu=(8, 8), memory=(32768, 65536),
              max_containers=1, timeout=86400, retries=0, scaledown_window=60)
def prepare_test_job(run_id: str):
    import json
    import subprocess

    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid parent run ID")
    work = Path("/vol/runs") / run_id
    if not (work / "prepared/prepared.json").exists():
        raise ValueError("Wait for parent preparation to finish before starting test assets")
    log = work / "test_assets" / "prepare.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a", encoding="utf-8") as f:
            cmd = ["/opt/falcon/.venv/bin/python", "-u", "-m", "plan3.test_assets", "--work-root", str(work)]
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        return json.loads((work / "test_assets/status.json").read_text())
    finally:
        volume.commit()


@app.function(image=image, volumes={"/vol": volume}, cpu=(16, 16), memory=(131072, 196608),
              max_containers=1, timeout=86400, retries=0, scaledown_window=60)
def finish_job(run_id: str, validator_sha: str):
    import json
    import subprocess

    if not run_id.replace("-", "").isalnum() or len(validator_sha) != 64 or any(c not in "0123456789abcdef" for c in validator_sha):
        raise ValueError("Invalid final job parameters")
    work = Path("/vol/runs") / run_id
    validator = Path("/vol/utilities") / validator_sha / "validate_submission.py"
    volume.reload()
    try:
        subprocess.run(["/opt/falcon/.venv/bin/python", "-m", "pytest", "plan3/tests", "-q"], check=True)
        subprocess.run(["/opt/falcon/.venv/bin/python", "-u", "-m", "plan3.finish", "--work-root", str(work),
                        "--validator", str(validator), "--validator-sha", validator_sha, "--threads", "16"], check=True)
        result = json.loads((work / "final/complete.json").read_text())
        return {"status": result["status"], "model": result["model"], "archive": result["archive"],
                "archive_sha256": result["archive_sha256"],
                "audit": result["audit"]["after_ownership"]}
    finally:
        volume.commit()


@app.function(image=image, volumes={"/vol": volume}, cpu=0.125, memory=256,
              max_containers=1, timeout=86400, retries=0, scaledown_window=60)
def finish_after_training(run_id: str, training_call: str, test_assets_call: str, validator_sha: str):
    import json
    import os
    import time

    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    status = Path("/vol/runs") / run_id / "finish_queue.json"
    def update(value):
        status.parent.mkdir(parents=True, exist_ok=True)
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temp, status)
        volume.commit()
    volume.reload()
    update({"status": "waiting", "stage": "training", "training_call": training_call})
    try:
        deadline = time.monotonic() + 86300
        modal.FunctionCall.from_id(training_call).get(timeout=max(0, deadline - time.monotonic()))
        update({"status": "waiting", "stage": "test-assets", "test_assets_call": test_assets_call})
        modal.FunctionCall.from_id(test_assets_call).get(timeout=max(0, deadline - time.monotonic()))
        volume.reload()
        call = finish_job.spawn(run_id, validator_sha)
        result = {"status": "submitted", "stage": "finish", "finish_call_id": call.object_id}
        update(result)
        # The CPU child has its own timeout; this cheap coordinator must not
        # spend its remaining lifetime waiting for that child's result.
        return result
    except BaseException as error:
        update({"status": "failed", "error": str(error)})
        raise
