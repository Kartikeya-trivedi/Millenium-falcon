"""Upload a fixed dataset and control persistent CPU jobs through Modal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .artifacts import ROOT, fingerprint, read_json, sha256, write_json

APP = "millenium-falcon-p3"
VOLUME = "falcon-record-runs"


def pack_dataset(dataset: Path, out: Path):
    names = [f"train/train_source{s}.tsv" for s in (1, 2, 3)] + ["train/train_ground_truth.tsv"]
    names += [f"test/test_source{s}.tsv" for s in (1, 2, 3)]
    files = {name: {"bytes": (dataset/name).stat().st_size, "sha256": sha256(dataset/name)} for name in names}
    identity = fingerprint(files)[:20]
    folder = out / identity
    manifest = {"dataset_id": identity, "files": files}
    archive = folder / "dataset.zip"
    if archive.exists() and (folder/"manifest.json").exists():
        if read_json(folder/"manifest.json") != manifest:
            raise ValueError("Dataset archive metadata mismatch")
        return identity, archive, manifest
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / "dataset.zip.tmp"
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
        for name in names:
            print(f"Packing {name}", flush=True)
            z.write(dataset/name, name)
    temp.replace(archive)
    write_json(folder/"manifest.json", manifest)
    return identity, archive, manifest


def upload(dataset: Path, out: Path):
    import modal
    identity, archive, manifest = pack_dataset(dataset, out)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    print(f"Uploading {archive.stat().st_size/1e6:.1f} MB to {VOLUME}/inputs/{identity}", flush=True)
    with volume.batch_upload() as batch:
        batch.put_file(str(archive), f"/inputs/{identity}/dataset.zip")
        batch.put_file(io.BytesIO(json.dumps(manifest).encode()), f"/inputs/{identity}/manifest.json")
    write_json(out / "uploaded.json", {"dataset_id": identity, "volume": VOLUME})
    print(f"Dataset ready for cloud preparation: {identity}", flush=True)
    return identity


def submit(dataset_id: str, run_id: str, out: Path, profile="full"):
    import modal
    if not dataset_id.isalnum() or not run_id.replace("-", "").isalnum():
        raise ValueError("Use alphanumeric dataset IDs and alphanumeric/hyphen run IDs")
    path = out / (run_id + ".json")
    if path.exists():
        raise ValueError(f"Job already recorded at {path}; check it before submitting another job")
    fn = modal.Function.from_name(APP, "train_job")
    call = fn.spawn(dataset_id, run_id, profile)
    job = {"call_id": call.object_id, "dataset_id": dataset_id, "run_id": run_id, "profile": profile,
           "app": APP, "volume": VOLUME, "submitted_at": datetime.now(timezone.utc).isoformat()}
    write_json(path, job)
    print(json.dumps(job, indent=2), flush=True)
    return job


def status(job_file: Path):
    import modal
    job = read_json(job_file)
    call = modal.FunctionCall.from_id(job["call_id"])
    try:
        result = call.get(timeout=0)
        print(json.dumps({"status": "complete", "result": result}, indent=2), flush=True)
    except TimeoutError:
        print("Cloud job is running.", flush=True)
    volume = modal.Volume.from_name(job["volume"])
    path = (f"/runs/{job['run_id']}/test_assets/status.json" if job.get("kind") == "test-assets" else
            f"/runs/{job['run_id']}/runs/{job['profile']}-w100-m25/server_status.json")
    try:
        print(b"".join(volume.read_file(path)).decode(), flush=True)
    except FileNotFoundError:
        print("Preparing the input dataset; no pipeline stage status yet.", flush=True)


def submit_test_assets(parent_job: Path):
    import modal
    parent = read_json(parent_job)
    path = parent_job.with_name(parent_job.stem + "-test-assets.json")
    if path.exists():
        raise ValueError(f"Job already recorded at {path}; check it before resubmitting")
    app = APP + "-followup"
    call = modal.Function.from_name(app, "prepare_test_job").spawn(parent["run_id"])
    job = {**parent, "kind": "test-assets", "app": app, "call_id": call.object_id,
           "submitted_at": datetime.now(timezone.utc).isoformat()}
    write_json(path, job)
    print(json.dumps(job, indent=2), flush=True)
    return job


def download(job_file: Path, out: Path):
    import modal
    job = read_json(job_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(job["volume"])
    with out.open("xb") as f:
        for block in volume.read_file(f"/runs/{job['run_id']}/results.zip"):
            f.write(block)
    print(f"Downloaded private results to {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--dataset", type=Path, required=True)
    u.add_argument("--out", type=Path, default=ROOT/"work/modal")
    s = sub.add_parser("submit")
    s.add_argument("--dataset-id", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--profile", choices=("full", "pilot"), default="full")
    s.add_argument("--out", type=Path, default=ROOT/"work/modal")
    g = sub.add_parser("status")
    g.add_argument("--job", type=Path, required=True)
    a = sub.add_parser("prepare-test")
    a.add_argument("--job", type=Path, required=True)
    d = sub.add_parser("download")
    d.add_argument("--job", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "upload":
        upload(args.dataset, args.out)
    elif args.command == "submit":
        submit(args.dataset_id, args.run_id, args.out, args.profile)
    elif args.command == "status":
        status(args.job)
    elif args.command == "prepare-test":
        submit_test_assets(args.job)
    else:
        download(args.job, args.out)


if __name__ == "__main__":
    main()
