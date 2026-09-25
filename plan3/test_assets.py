"""Prepare label-free test views and indexes while model experiments run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import contract, load_prepared, sha256, write_json
from .retrieve import index_for
from .views import prepare_views


def prepare_test_assets(work: Path):
    prepared, views, indexes = work / "prepared", work / "views", work / "indexes"
    meta = load_prepared(prepared)
    if not any(p["split"] == "test" for p in meta["partitions"]):
        raise ValueError("Prepared snapshot does not contain test records")
    folder = work / "test_assets"
    spec = {"prepared_identity": meta["identity"], "views_code": sha256(Path(__file__).with_name("views.py")),
            "index_code": sha256(Path(__file__).with_name("index.py")),
            "channels": ["word", "missing"], "shard_size": 50000}
    contract(folder / "contract.json", spec)
    status = {"stage": "views", "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        write_json(folder / "status.json", status)
        prepare_views(prepared, views, "test")
        for part in meta["partitions"]:
            if part["split"] != "test" or part["source"] == "S1":
                continue
            for channel in spec["channels"]:
                status = {"stage": "index", "status": "running", "country": part["country"],
                          "source": part["source"], "channel": channel}
                write_json(folder / "status.json", status)
                print(f"Test index {part['country']}/{part['source']}/{channel}", flush=True)
                index_for(prepared, indexes, "test", part["country"], part["source"], channel)
        write_json(folder / "status.json", {"stage": "test_assets", "status": "complete",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "contract_sha256": sha256(folder / "contract.json")})
    except BaseException as e:
        write_json(folder / "status.json", {**status, "status": "failed", "error": str(e)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    prepare_test_assets(parser.parse_args().work_root)


if __name__ == "__main__":
    main()
