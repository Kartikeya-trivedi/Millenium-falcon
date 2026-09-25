"""Persistent CPU training with a pinned uv environment and volume checkpoints."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("millenium-falcon-p3")
volume = modal.Volume.from_name("falcon-record-runs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install("uv==0.9.18")
    .add_local_file(ROOT/"pyproject.toml", "/opt/falcon/pyproject.toml", copy=True)
    .add_local_file(ROOT/"uv.lock", "/opt/falcon/uv.lock", copy=True)
    .add_local_file(ROOT/".python-version", "/opt/falcon/.python-version", copy=True)
    .run_commands("cd /opt/falcon && uv sync --locked --default-index https://pypi.org/simple")
    .add_local_dir(ROOT/"plan1", "/opt/falcon/plan1", copy=True, ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT/"plan3", "/opt/falcon/plan3", copy=True, ignore=["**/__pycache__/**"])
    .workdir("/opt/falcon")
    .env({"POLARS_MAX_THREADS": "8", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "1",
          "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}))


@app.function(image=image, volumes={"/vol": volume}, cpu=(16, 16), memory=(131072, 196608),
              max_containers=1, timeout=86400, retries=0, scaledown_window=60)
def train_job(dataset_id: str, run_id: str, profile: str = "full"):
    import json
    import shutil
    import subprocess
    from zipfile import ZipFile

    if not dataset_id.isalnum() or not run_id.replace("-", "").isalnum() or profile not in ("full", "pilot"):
        raise ValueError("Invalid job parameters")
    source = Path("/vol/inputs") / dataset_id
    dataset = source / "dataset"
    manifest = json.loads((source/"manifest.json").read_text())
    if manifest["dataset_id"] != dataset_id:
        raise ValueError("Dataset ID mismatch")
    ready = dataset/"ready.json"
    if ready.exists() and json.loads(ready.read_text()) != manifest:
        raise ValueError("Extracted dataset manifest mismatch")
    if not ready.exists():
        import hashlib
        expected = set(manifest["files"])
        allowed = {f"train/train_source{s}.tsv" for s in (1, 2, 3)} | {"train/train_ground_truth.tsv"}
        allowed |= {f"test/test_source{s}.tsv" for s in (1, 2, 3)}
        if expected != allowed:
            raise ValueError("Unexpected dataset files")
        with ZipFile(source/"dataset.zip") as z:
            if set(z.namelist()) != allowed or len(z.namelist()) != len(allowed):
                raise ValueError("Unexpected archive members")
            for name in sorted(allowed):
                target = dataset/name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 << 20)
                h = hashlib.sha256()
                with target.open("rb") as f:
                    for block in iter(lambda: f.read(8 << 20), b""):
                        h.update(block)
                expected_file = manifest["files"][name]
                if target.stat().st_size != expected_file["bytes"] or h.hexdigest() != expected_file["sha256"]:
                    raise ValueError(f"Dataset integrity failure: {name}")
        ready.write_text(json.dumps(manifest))
        volume.commit()
    work = Path("/vol/runs")/run_id
    work.mkdir(parents=True, exist_ok=True)
    python = "/opt/falcon/.venv/bin/python"
    cmd = [python, "-u", "-m", "plan3.server", "--dataset", str(dataset), "--work-root", str(work),
           "--profile", profile, "--threads", "16", "--include-test", "--stop-after", "support"]
    try:
        subprocess.run([python, "-m", "pytest", "plan3/tests", "-q"], check=True)
        subprocess.run(cmd, check=True)
        run = work/"runs"/(profile+"-w100-m25")
        for args in (
            ["plan3.ablate_support", "--threads", "16"],
            ["plan3.compare", "--before", "direct", "--after", "support"],
            ["plan3.compare", "--before", "support_control", "--after", "support"],
            ["plan3.diagnose", "--model", "direct"],
            ["plan3.diagnose", "--model", "support"],
        ):
            subprocess.run([python, "-u", "-m", *args, "--prepared", str(work/"prepared"), "--run", str(run)], check=True)
            volume.commit()
        subprocess.run([python, "-m", "plan3.collect", "--run", str(run), "--out", str(work/"results.zip")], check=True)
        return {"run_id": run_id, "profile": profile, "results": str(work/"results.zip"),
                "direct": json.loads((run/"direct_report.json").read_text()),
                "support": json.loads((run/"support_report.json").read_text())}
    finally:
        volume.commit()
