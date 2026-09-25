"""Assemble complete ordered TSV outputs from authenticated inference shards.

Score shards are processed in sequence and candidate lists are spooled to disk.
Accepted target claims are resolved across every shard.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import tempfile

import polars as pl

from .artifacts import fingerprint, read_json, records, sha256, write_json

SCORE_SCHEMA = {"s1_id": pl.String, "target_id": pl.String, "source": pl.String,
                "country": pl.String, "p": pl.Float64, "p_direct": pl.Float64,
                "score": pl.Float32, "rank": pl.Int32}
OUTPUTS = {"candidate_pairs.tsv": "candidate_entity_ids",
           "matching_results.tsv": "matched_entity_ids"}


def _raw_roster(path: Path, source: str) -> pl.DataFrame:
    frame = records(path).select("entity_id", "country").collect(engine="streaming")
    if (frame["entity_id"].null_count() or frame["country"].null_count() or
            frame["entity_id"].n_unique() != frame.height or
            not frame["entity_id"].str.starts_with(source + "-").all() or
            frame["entity_id"].str.contains(r'[\s,\"]').any() or
            (frame["country"] == "").any()):
        raise ValueError(f"Invalid or duplicate raw {source} IDs/countries")
    return frame


def score_shard_path(inference: Path, name: str) -> Path:
    """Resolve only canonical relative paths inside the score directory."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Invalid score shard path")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or relative.as_posix() != name or
            any(part in ("", ".", "..") for part in relative.parts) or
            not name.endswith(".parquet")):
        raise ValueError("Invalid score shard path")
    root = (Path(inference) / "scores").resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Score shard path escapes its directory")
    return path


def inference_manifest(inference: Path):
    """Validate shared provenance without imposing an evaluation/output split."""
    inference = Path(inference)
    spec = read_json(inference / "inference_contract.json")
    complete = read_json(inference / "inference_complete.json")
    contract_hash = sha256(inference / "inference_contract.json")
    if complete.get("contract_sha256") != contract_hash:
        raise ValueError("Inference contract hash mismatch")
    try:
        decision = spec["frozen"]["decision"]
        threshold = float(decision["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing frozen decision") from exc
    if (decision.get("selected_on") != "select" or not math.isfinite(threshold) or
            not 0 <= threshold <= 1.000001):
        raise ValueError("Invalid frozen decision")
    roster_name = complete.get("roster_file", "roster.parquet")
    if roster_name not in ("roster.parquet", "roster.partition"):
        raise ValueError("Invalid inference roster path")
    roster_path = inference / roster_name
    if complete.get("roster_sha256") != sha256(roster_path):
        raise ValueError("Inference roster hash mismatch")
    roster = pl.read_parquet(roster_path)
    if (dict(roster.schema) != {"s1_id": pl.String, "country": pl.String} or
            roster["s1_id"].null_count() or roster["country"].null_count() or
            roster["s1_id"].n_unique() != roster.height or not roster.height or
            spec.get("queries") != roster.height or complete.get("queries") != roster.height or
            spec.get("query_ids") != fingerprint(roster["s1_id"].sort().to_list())):
        raise ValueError("Inference roster count or query fingerprint mismatch")
    names = complete.get("shards")
    metadata = complete.get("score_shards")
    if (not isinstance(names, list) or not isinstance(metadata, dict) or
            any(not isinstance(n, str) for n in names) or
            len(names) != len(set(names)) or set(names) != set(metadata)):
        raise ValueError("Invalid score shard manifest")
    for name in names:
        score_shard_path(inference, name)
    root = inference / "scores"
    if set(p.relative_to(root).as_posix() for p in root.rglob("*.parquet")) != set(names):
        raise ValueError("Missing or unexpected score shards")
    return spec, complete, roster


def _inputs(inference: Path, test_dir: Path):
    spec, complete, roster = inference_manifest(inference)
    if spec.get("split") != "test":
        raise ValueError("Output assembly requires complete test inference")
    expected = {f"test/test_source{s}.tsv" for s in (1, 2, 3)}
    if set(spec.get("raw_inputs", {})) != expected:
        raise ValueError("Inference must bind all three raw test inputs")
    for name, metadata in spec["raw_inputs"].items():
        path = test_dir / Path(name).name
        if metadata.get("bytes") != path.stat().st_size or metadata.get("sha256") != sha256(path):
            raise ValueError(f"Raw input hash/size mismatch: {path.name}")
    raw = _raw_roster(test_dir / "test_source1.tsv", "S1")
    expected_roster = raw.rename({"entity_id": "s1_id"}).sort("s1_id")
    if not raw.height or not roster.sort("s1_id").equals(expected_roster):
        raise ValueError("Inference roster differs from the complete raw test roster")
    return spec, complete, raw, float(spec["frozen"]["decision"]["threshold"])


def _score_shard(path: Path, metadata: dict) -> pl.DataFrame:
    # Parse exactly the bytes that were authenticated, even if another process
    # replaces the path while export is running.
    data = path.read_bytes()
    if metadata.get("sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError(f"Score shard hash mismatch: {path.name}")
    frame = pl.read_parquet(data)
    if dict(frame.schema) != SCORE_SCHEMA or frame.height != metadata.get("rows"):
        raise ValueError(f"Score shard schema/row count mismatch: {path.name}")
    if any(frame[name].null_count() for name in SCORE_SCHEMA if name != "rank"):
        raise ValueError(f"Null score fields: {path.name}")
    for name in ("p", "p_direct", "score"):
        if not frame[name].is_finite().all():
            raise ValueError(f"Non-finite scores: {path.name}")
    if any(((frame[name] < 0) | (frame[name] > 1)).any() for name in ("p", "p_direct")):
        raise ValueError(f"Probability outside [0, 1]: {path.name}")
    if (frame["rank"] < 1).any():
        raise ValueError(f"Invalid retrieval rank: {path.name}")
    if frame.select("s1_id", "target_id").n_unique() != frame.height:
        raise ValueError(f"Duplicate scored candidate pairs: {path.name}")
    return frame.sort("s1_id", "target_id")


def iter_score_shards(inference: Path, complete: dict):
    """Yield authenticated shards, rejecting pairs or owner groups repeated later."""
    seen_owners = set()
    for name in sorted(complete["shards"]):
        frame = _score_shard(score_shard_path(inference, name), complete["score_shards"][name])
        owners = set(frame["s1_id"].unique())
        if owners & seen_owners:
            raise ValueError(f"Owner groups appear in multiple score shards: {name}")
        seen_owners.update(owners)
        yield name, frame


def export(inference: Path, test_dir: Path, out: Path) -> dict:
    """Validate inference, resolve target ownership globally, and write both TSVs.

    Every owner's candidates must stay together in one score shard.
    The highest probability at or above the frozen threshold wins each target;
    exact ties choose the lexicographically smallest source-one ID.
    """
    inference, test_dir, out = Path(inference), Path(test_dir), Path(out)
    if any((out / name).exists() for name in (*OUTPUTS, "export_complete.json")):
        raise FileExistsError("Choose a new output directory; existing outputs are preserved")
    spec, complete, raw, threshold = _inputs(inference, test_dir)
    raw_ids = raw["entity_id"].to_list()
    owners = {qid: (i, country) for i, (qid, country) in enumerate(raw.iter_rows())}
    target_countries = {}
    for source in ("S2", "S3"):
        part = _raw_roster(test_dir / f"test_source{source[1]}.tsv", source)
        target_countries.update(part.iter_rows())
    del raw, part
    # Per-owner offsets are small; candidate strings live in the temporary file.
    chunks: list[tuple[int, int] | None] = [None] * len(raw_ids)
    winners: dict[str, tuple[float, str]] = {}
    counts = {"owners": len(raw_ids), "candidate_pairs": 0, "accepted_pairs": 0,
              "matched_pairs": 0, "owners_without_candidates": 0, "owners_without_matches": 0}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=out) as temporary:
        stage = Path(temporary)
        with (stage / "candidates.bin").open("w+b") as spool:
            for name, frame in iter_score_shards(inference, complete):
                counts["candidate_pairs"] += frame.height
                for qid, tid, source, country, probability in frame.select(
                        "s1_id", "target_id", "source", "country", "p").iter_rows():
                    owner = owners.get(qid)
                    if owner is None:
                        raise ValueError(f"Unknown scored owner: {qid}")
                    if tid not in target_countries or source not in ("S2", "S3") or not tid.startswith(source + "-"):
                        raise ValueError(f"Unknown target or incorrect source: {tid}")
                    if owner[1] != country or target_countries[tid] != country:
                        raise ValueError(f"Cross-country candidate: {qid}, {tid}")
                    if probability >= threshold:
                        counts["accepted_pairs"] += 1
                        old = winners.get(tid)
                        if old is None or probability > old[0] or (probability == old[0] and qid < old[1]):
                            winners[tid] = (probability, qid)
                grouped = frame.group_by("s1_id", maintain_order=True).agg("target_id")
                for qid, ids in grouped.iter_rows():
                    position = owners[qid][0]
                    encoded = ",".join(ids).encode("utf-8")
                    offset = spool.tell()
                    spool.write(encoded)
                    chunks[position] = (offset, len(encoded))
                print(f"Validated {name}: {frame.height:,} scored candidates", flush=True)
            del owners, target_countries
            by_owner: dict[str, list[str]] = {}
            for target, (_, owner) in winners.items():
                by_owner.setdefault(owner, []).append(target)
            counts["matched_pairs"] = len(winners)
            counts["claims_lost_at_ownership"] = counts["accepted_pairs"] - len(winners)
            del winners
            hashes = {name: hashlib.sha256() for name in OUTPUTS}
            sizes = {name: 0 for name in OUTPUTS}
            with (stage / "candidate_pairs.tsv").open("wb") as candidates, (stage / "matching_results.tsv").open("wb") as matches:
                streams = {"candidate_pairs.tsv": candidates, "matching_results.tsv": matches}

                def write(name, text):
                    encoded = text.encode("utf-8")
                    streams[name].write(encoded)
                    hashes[name].update(encoded)
                    sizes[name] += len(encoded)

                for name, second in OUTPUTS.items():
                    write(name, f"source1_entity_id\t{second}\n")
                for i, qid in enumerate(raw_ids):
                    ids = []
                    if chunks[i] is not None:
                        offset, length = chunks[i]
                        spool.seek(offset)
                        ids.extend(spool.read(length).decode("utf-8").split(","))
                    if len(ids) != len(set(ids)):
                        raise ValueError(f"Duplicate scored candidate pairs across shards: {qid}")
                    ids.sort()
                    matched = sorted(by_owner.get(qid, []))
                    if not set(matched).issubset(ids):
                        raise ValueError(f"Final matches outside the scored candidate set: {qid}")
                    counts["owners_without_candidates"] += not ids
                    counts["owners_without_matches"] += not matched
                    write("candidate_pairs.tsv", qid + "\t" + ",".join(ids) + "\n")
                    write("matching_results.tsv", qid + "\t" + ",".join(matched) + "\n")
                    chunks[i] = None
        report = {"format": 1, "threshold": threshold, "counts": counts,
                  "inference_contract_sha256": sha256(inference / "inference_contract.json"),
                  "inference_complete_sha256": sha256(inference / "inference_complete.json"),
                  "raw_inputs": spec["raw_inputs"], "export_code_sha256": sha256(Path(__file__)),
                  "checks": ["complete_raw_roster_in_original_order", "all_score_shards_authenticated",
                             "exact_scored_candidate_set", "valid_target_ids_and_same_country",
                             "finite_scores", "no_duplicate_pairs_or_owner_rows",
                             "global_single_owner_per_target", "matches_subset_of_candidates",
                             "utf8_no_bom_lf_tabs_without_quoting"],
                  "files": {name: {"sha256": hashes[name].hexdigest(), "bytes": sizes[name]}
                            for name in OUTPUTS}}
        for name in OUTPUTS:
            os.replace(stage / name, out / name)
        write_json(out / "export_complete.json", report)
    return report


def verify_files(out: Path) -> dict:
    """Check that a completed output bundle still has its recorded bytes."""
    report = read_json(out / "export_complete.json")
    if set(report.get("files", {})) != set(OUTPUTS):
        raise ValueError("Invalid output manifest")
    for name, metadata in report["files"].items():
        path = out / name
        if metadata.get("bytes") != path.stat().st_size or metadata.get("sha256") != sha256(path):
            raise ValueError(f"Output file hash/size mismatch: {name}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        report = verify_files(args.out)
    else:
        if args.inference is None or args.test_dir is None:
            parser.error("--inference and --test-dir are required for assembly")
        report = export(args.inference, args.test_dir, args.out)
    print(report["counts"], flush=True)


if __name__ == "__main__":
    main()
