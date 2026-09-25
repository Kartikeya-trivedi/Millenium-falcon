"""Package compact run evidence for private review without large feature caches."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def collect(run: Path, output: Path):
    if not (run / "server_status.json").exists() and not (run / "retrieval_contract.json").exists():
        raise ValueError("No run metadata found")
    # Explicit root-level types exclude raw records, full candidates and all feature shards.
    patterns = ("*.json", "budget_grid.tsv", "*_errors.parquet", "*_per_owner.parquet",
                "*_threshold_sweep.parquet", "retrieval_timings.parquet", "misses_*.parquet")
    paths = {p for pattern in patterns for p in run.glob(pattern) if p.is_file()}
    paths.update(p for p in (run / "server_logs").glob("*.log") if p.is_file())
    paths.update(p for p in (run / "direct.txt", run / "support.txt") if p.exists())
    paths.update(p for p in run.glob("support_*.txt") if p.is_file())
    for directory in ("comparisons", "diagnostics", "onehop_probe"):
        paths.update(p for p in (run/directory).rglob("*.json") if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path, path.relative_to(run).as_posix())
    print(f"Packed {len(paths)} files into {output}. Share privately; error ledgers contain record IDs.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    collect(args.run.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
