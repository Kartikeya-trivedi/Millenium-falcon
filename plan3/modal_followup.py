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
