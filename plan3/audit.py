"""Evaluate immutable predictions with explicit claimant scope and protected Audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from plan1.decide import one_owner
from .artifacts import contract, load_prepared, parquet, read_json, sha256, write_json
from .decisions import error_ledger
from .diagnose import normalized_targets, link_slices, orphan_summary
from .export import inference_manifest, iter_score_shards

KEYS = ["s1_id", "target_id"]


def evaluation_population(roster, manifest, fresh, model, panel):
    if panel not in ("development", "fresh-audit"):
        raise ValueError("Unknown evaluation panel")
    joined = roster.join(manifest.select("s1_id", "role", "pool", expected_country="country"), on="s1_id", how="left")
    if joined["role"].null_count() or joined.filter(pl.col("country") != pl.col("expected_country")).height:
        raise ValueError("Claimant roster is outside the declared training population")
    if joined.filter(pl.col("role") == "fit").height:
        raise ValueError("Fit/dictionary training owners cannot be held-out claimants")
    if model == "support" and joined.filter(pl.col("pool") == "c_prob").height:
        raise ValueError("Support training pool cannot be held-out claimants")
    protected = fresh.select("s1_id")
    if panel == "development":
        if roster.join(protected, on="s1_id", how="semi").height:
            raise ValueError("Fresh Audit is excluded from development evaluation")
        return roster["s1_id"]
    if protected.join(roster.select("s1_id"), on="s1_id", how="anti").height:
        raise ValueError("Fresh Audit requires every reserved owner, including empty answers")
    return protected["s1_id"]


def clustered_intervals(per, repeats=2000):
    """Resample complete owners, keeping their link numerator/denominator together."""
    values = per.select("f3", "n_true", "n_retrieved").to_numpy()
    if not len(values) or repeats < 100:
        raise ValueError("Nonempty owners and at least 100 resamples are required")
    rng = np.random.default_rng(42)
    macro, recall = [], []
    for start in range(0, repeats, 25):
        selected = values[rng.integers(0, len(values), (min(25, repeats-start), len(values)))]
        macro.extend(selected[:, :, 0].mean(axis=1))
        denominator = selected[:, :, 1].sum(axis=1)
        numerator = selected[:, :, 2].sum(axis=1)
        recall.extend(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    valid = np.asarray(recall)[np.isfinite(recall)]
    return {"owner_bootstrap_repeats": repeats, "seed": 42,
            "macro_f05_95pct": np.quantile(macro, [.025, .975]).tolist(),
            "candidate_recall_95pct": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "scope": "Conditional on fixed models, candidate settings and decisions; resamples whole owners."}


def evaluate_inference(prepared: Path, inference: Path, out: Path, panel="development"):
    spec, complete, roster = inference_manifest(inference)
    if spec["split"] != "train":
        raise ValueError("No test labels are available for accuracy evaluation")
    meta = load_prepared(prepared)
    if spec["frozen"]["prepared_identity"] != meta["identity"]:
        raise ValueError("Predictions and labels belong to different prepared snapshots")
    if (sha256(prepared / "manifest.parquet") != meta["manifest_sha256"] or
            sha256(prepared / "fresh_audit.parquet") != meta["fresh_audit_sha256"]):
        raise ValueError("Prepared split or reserved Audit membership was modified")
    manifest = pl.read_parquet(prepared / "manifest.parquet")
    fresh = pl.read_parquet(prepared / "fresh_audit.parquet")
    ids = evaluation_population(roster, manifest, fresh, spec["frozen"]["model"], panel)
    run_spec = {"inference_contract_sha256": sha256(inference / "inference_contract.json"),
                "inference_complete_sha256": sha256(inference / "inference_complete.json"),
                "panel": panel, "owners": ids.len(), "evaluation_code": sha256(Path(__file__)),
                "truth_sha256": sha256(prepared / "truth.parquet")}
    contract(out / "evaluation_contract.json", run_spec)
    if panel == "fresh-audit":
        # A second evaluation can reproduce the same frozen experiment, but cannot
        # silently turn this reserved population into a model-selection loop.
        contract(prepared / "fresh_audit_use.json", {"frozen": spec["frozen"],
            "claimant_ids": spec["query_ids"], "fresh_audit_sha256": sha256(prepared / "fresh_audit.parquet")})
    truth_all = pl.read_parquet(prepared / "truth.parquet")
    truth = truth_all.filter(pl.col("s1_id").is_in(ids.implode()))
    query_countries = dict(roster.iter_rows())
    target_countries = {}
    for part in meta["partitions"]:
        if part["split"] == "train" and part["source"] != "S1":
            target_countries.update(dict.fromkeys(
                pl.read_parquet(prepared / part["file"], columns=["entity_id"])["entity_id"], part["country"]))
    threshold = spec["frozen"]["decision"]["threshold"]
    accepted_world, scored_panel = [], []
    for name, frame in iter_score_shards(inference, complete):
        for qid, tid, source, country in frame.select(*KEYS, "source", "country").iter_rows():
            if (query_countries.get(qid) != country or target_countries.get(tid) != country or
                    source not in ("S2", "S3") or not tid.startswith(source + "-")):
                raise ValueError(f"Unknown or cross-country pair in {name}")
        accepted_world.append(frame.filter(pl.col("p") >= threshold).select(*KEYS, "p"))
        scored_panel.append(frame.filter(pl.col("s1_id").is_in(ids.implode())).select(*KEYS, "p"))
    del target_countries
    accepted_world = pl.concat(accepted_world)
    final_world = one_owner(accepted_world)
    scored = pl.concat(scored_panel)
    accepted = accepted_world.filter(pl.col("s1_id").is_in(ids.implode()))
    final = final_world.filter(pl.col("s1_id").is_in(ids.implode()))
    report, errors, per = error_ledger(scored, scored, accepted, final, truth, ids)
    found = truth.join(scored.select(KEYS), on=KEYS, how="semi")
    per = (per.join(truth.group_by("s1_id").len(name="n_true"), on="s1_id", how="left")
           .join(found.group_by("s1_id").len(name="n_retrieved"), on="s1_id", how="left")
           .join(scored.group_by("s1_id").len(name="n_candidates"), on="s1_id", how="left")
           .with_columns(pl.col("n_true", "n_retrieved", "n_candidates").fill_null(0)))
    mean = per["n_candidates"].mean()
    p95 = per["n_candidates"].quantile(.95, interpolation="nearest")
    recall = found.height / truth.height if truth.height else None
    eligible = manifest.filter(pl.col("role") != "fit")
    if spec["frozen"]["model"] == "support":
        eligible = eligible.filter(pl.col("pool") != "c_prob")
    if panel == "development":
        eligible = eligible.join(fresh.select("s1_id"), on="s1_id", how="anti")
    report.update({"panel": panel, "evaluation_owners": ids.len(), "claimant_owners": roster.height,
        "eligible_held_out_owners": eligible.height, "claimant_coverage": roster.height / eligible.height,
        "ownership_scope": "Global across the declared inference roster; this roster is not automatically the full test claimant population.",
        "true_links": truth.height, "retrieved_true_links": found.height, "retrieval_misses": truth.height-found.height,
        "candidate_recall": recall, "mean_candidates": mean, "p95_candidates": p95,
        "max_candidates": per["n_candidates"].max(), "budget_met": mean <= 400 and p95 <= 500,
        "recall_target_met": recall is not None and recall >= .995,
        "uncertainty": clustered_intervals(per), "frozen": spec["frozen"]})
    targets = normalized_targets(prepared, meta, truth.select("target_id"))
    report["link_slices"], _ = link_slices(truth, scored, final, targets)
    labelled = scored.join(truth.with_columns(label=pl.lit(1)), on=KEYS, how="left").with_columns(pl.col("label").fill_null(0))
    total_targets = sum(p["rows"] for p in meta["partitions"] if p["split"] == "train" and p["source"] != "S1")
    report["orphans"] = orphan_summary(labelled, accepted.select(KEYS), truth_all, total_targets)
    report["world_unowned_target_share"] = 1 - truth_all.join(roster.select("s1_id"), on="s1_id", how="semi").height / total_targets
    parquet(out / "errors.parquet", errors)
    parquet(out / "per_owner.parquet", per)
    write_json(out / "report.json", report)
    print(f"{panel}: macro F0.5={report['after_ownership']['macro_f05']:.6f}; "
          f"{ids.len():,} evaluated owners, {roster.height:,} claimants; {truth.height-found.height:,} retrieval misses", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--panel", choices=("development", "fresh-audit"), default="development")
    a = p.parse_args()
    evaluate_inference(a.prepared, a.inference, a.out, a.panel)


if __name__ == "__main__":
    main()
