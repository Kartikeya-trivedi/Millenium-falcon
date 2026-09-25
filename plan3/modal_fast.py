"""64-core continuation from stopped, persistent full-run checkpoints."""
from pathlib import Path
import modal
from .modal_app import image, volume

app = modal.App("millenium-falcon-p3-fast")
fast_image = image.env({"POLARS_MAX_THREADS": "64", "OMP_NUM_THREADS": "64",
                        "OPENBLAS_NUM_THREADS": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})


@app.function(image=fast_image, volumes={"/vol": volume}, cpu=(64, 64), memory=(196608, 262144),
              max_containers=1, timeout=86400, retries=0, scaledown_window=60)
def finish_fast(run_id: str, validator_sha: str):
    import json
    import subprocess
    if not run_id.replace("-", "").isalnum() or len(validator_sha) != 64 or any(c not in "0123456789abcdef" for c in validator_sha):
        raise ValueError("Invalid run or validator identity")
    volume.reload()
    work = Path("/vol/runs") / run_id
    validator = Path("/vol/utilities") / validator_sha / "validate_submission.py"
    try:
        subprocess.run(["/opt/falcon/.venv/bin/python", "-u", "-m", "plan3.finish", "--work-root", str(work),
            "--validator", str(validator), "--validator-sha", validator_sha, "--threads", "64"], check=True)
        result = json.loads((work / "final/complete.json").read_text())
        return {"status": result["status"], "model": result["model"], "archive": result["archive"],
                "archive_sha256": result["archive_sha256"], "audit": result["audit"]["after_ownership"]}
    finally:
        volume.commit()


@app.function(image=fast_image, volumes={"/vol": volume}, cpu=(64, 64), memory=(196608, 262144),
              max_containers=1, timeout=86400, retries=0, scaledown_window=60)
def continue_fast(run_id: str, stopped_training_call: str, stopped_queue_call: str, validator_sha: str):
    import json
    import os
    import subprocess
    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    for call_id in (stopped_training_call, stopped_queue_call):
        graph = modal.FunctionCall.from_id(call_id).get_call_graph()
        if not graph or any(node.status.name != "TERMINATED" for node in graph):
            raise ValueError("The prior training and finishing queue must be terminated first")
    volume.reload()
    work = Path("/vol/runs") / run_id
    python = "/opt/falcon/.venv/bin/python"
    try:
        print(json.dumps({"requested_physical_cpus": 64, "cpu_affinity": len(os.sched_getaffinity(0)),
                          "memory_limit_gib": 256, "run": run_id}), flush=True)
        subprocess.run([python, "-m", "pytest", "plan3/tests", "-q"], check=True)
        subprocess.run([python, "-u", "-m", "plan3.accelerate", "--work-root", str(work),
                        "--threads", "64", "--workers", "64"], check=True)
        volume.commit()
        call = finish_fast.spawn(run_id, validator_sha)
        result = {"status": "submitted", "stage": "finish", "finish_call_id": call.object_id}
        (work / "finish_queue_fast.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        volume.commit()
