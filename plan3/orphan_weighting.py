"""Predeclared robustness ablation: double unowned-target negative training weights."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import lightgbm as lgb
import polars as pl

from .artifacts import ROOT, contract, read_json, sha256, versions, write_json
from .features import FEATURES
from .support import SUPPORT_FEATURES
from .train import _matrix, evaluate_model, predict_model, sample_negatives

NAME = "support_orphan2"


def reweight_unowned(sampled: pl.DataFrame, owned_targets: pl.Series):
    """Use existence of a labeled owner for training weights only, never features.

    The multiplier is fixed at 2.0. It does not estimate a test population prior
    or force any target or candidate-pair prevalence.
    """
    if sampled["weight"].null_count() or (sampled["weight"] <= 0).any():
        raise ValueError("Positive, non-null base sampling weights are required")
    # A join builds one lookup table. A large scalar list membership expression
    # can instead scan millions of targets separately for every sampled pair.
    owned = owned_targets.unique().to_frame("target_id").with_columns(_owned=pl.lit(True))
    flagged = (sampled.join(owned, on="target_id", how="left", maintain_order="left")
               .with_columns(_unowned=~pl.col("_owned").fill_null(False)).drop("_owned"))
    if flagged.filter((pl.col("label") == 1) & pl.col("_unowned")).height:
        raise ValueError("A positive training target is absent from complete truth")
    positive = flagged.filter(pl.col("label") == 1)
    negative = flagged.filter(pl.col("label") == 0)
    unowned = negative.filter(pl.col("_unowned"))
    owned = negative.filter(~pl.col("_unowned"))
    def mass(frame):
        return float(frame["weight"].cast(pl.Float64).sum())
    stats = {
        "sampled_positive_pairs": positive.height,
        "sampled_negative_pairs": negative.height,
        "sampled_unowned_negative_pairs": unowned.height,
        "sampled_owned_negative_pairs": owned.height,
        "sampled_unowned_negative_share": unowned.height / negative.height if negative.height else None,
        "base_positive_weight": mass(positive),
        "base_negative_weight": mass(negative),
        "base_unowned_negative_weight": mass(unowned),
        "base_owned_negative_weight": mass(owned),
        "adjusted_unowned_negative_weight": 2.0 * mass(unowned),
        "adjusted_owned_negative_weight": mass(owned),
        "adjusted_negative_weight": mass(owned) + 2.0 * mass(unowned),
    }
    adjusted = flagged.with_columns(
        weight=pl.when((pl.col("label") == 0) & pl.col("_unowned"))
        .then(pl.col("weight") * 2.0).otherwise(pl.col("weight")).cast(pl.Float32)
    ).drop("_unowned")
    return adjusted, stats


def validate_populations(samples: pl.DataFrame, manifest: pl.DataFrame, protected: pl.DataFrame):
    if samples["s1_id"].n_unique() != samples.height:
        raise ValueError("Sample owners must be unique across roles")
    if samples.join(protected.select("s1_id"), on="s1_id", how="semi").height:
        raise ValueError("Fresh Audit owners must remain excluded")
    assigned = samples.join(manifest.select("s1_id", "pool"), on="s1_id", how="left")
    if assigned.height != samples.height or assigned["pool"].null_count():
        raise ValueError("Every sample owner requires one manifest assignment")
    expected = {"fit": "fit", "support": "c_prob", "tune": "tune",
                "select": "c_select", "development": "audit"}
    for sample, pool in expected.items():
        selected = assigned.filter(pl.col("sample") == sample)
        if not selected.height or selected.filter(pl.col("pool") != pool).height:
            raise ValueError(f"Unexpected {sample} population")
    if assigned.filter(~pl.col("sample").is_in(list(expected))).height:
        raise ValueError("Unknown sample role")
    return assigned


def run_experiment(prepared: Path, run: Path, threads=4):
    features = FEATURES + SUPPORT_FEATURES
    parent = read_json(run / "support_meta.json")
    upstream = read_json(run / "direct_meta.json")
    feature_spec = read_json(run / "support_features_contract.json")
    for name, meta in (("support", parent), ("direct", upstream)):
        if sha256(run / (name + ".txt")) != meta["model_sha256"]:
            raise ValueError(f"Changed {name} parent model")
    if (feature_spec["parent_model_sha256"] != upstream["model_sha256"] or
            feature_spec["support_training_pool"] != "c_prob" or
            feature_spec["upstream_training_pool"] != "fit" or
            parent["sample"] != "support" or upstream["sample"] != "fit"):
        raise ValueError("Independent upstream scoring provenance is required")
    if feature_spec["features"] != features or parent["features"] != features:
        raise ValueError("Saved sibling feature definitions differ")
    train_sha = sha256(Path(__file__).with_name("train.py"))
    if train_sha != parent["train_code"]:
        raise ValueError("Baseline training helpers changed; sampling equality is not established")
    if versions() != parent["dependencies"]:
        raise ValueError("Baseline dependency versions changed")
    samples = pl.read_parquet(run / "samples.parquet")
    assigned = validate_populations(
        samples, pl.read_parquet(prepared / "manifest.parquet", columns=["s1_id", "pool"]),
        pl.read_parquet(prepared / "fresh_audit.parquet", columns=["s1_id"]))
    sample_sha = sha256(run / "samples.parquet")
    if any(m["training_ids_sha"] != sample_sha for m in (parent, upstream)):
        raise ValueError("Saved sample owners differ from parent models")
    files = sorted((run / "support_features").glob("*.parquet"))
    file_hashes = {p.name: sha256(p) for p in files}
    if not files or file_hashes != parent["feature_files"]:
        raise ValueError("Saved sibling feature shards changed")
    params = {**parent["params"], "num_threads": threads}
    spec = {
        "name": NAME, "features": features, "sample": "support", "threads": threads,
        "params": params, "max_rounds": parent["max_rounds"], "early_stopping_rounds": 200,
        "parent_model_sha256": parent["model_sha256"],
        "upstream_model_sha256": upstream["model_sha256"],
        "support_meta_sha256": sha256(run / "support_meta.json"),
        "support_feature_contract_sha256": sha256(run / "support_features_contract.json"),
        "support_feature_contract": feature_spec,
        "feature_files": file_hashes, "training_ids_sha256": sample_sha,
        "manifest_sha256": sha256(prepared / "manifest.parquet"),
        "fresh_audit_sha256": sha256(prepared / "fresh_audit.parquet"),
        "truth_sha256": sha256(prepared / "truth.parquet"),
        "source_sha256": sha256(Path(__file__)), "train_helpers_sha256": train_sha,
        "dependencies": versions(),
        "sampling": {"cap": 32, "hard": 8, "random": 24, "importance_weighted": True,
                     "unowned_negative_weight_multiplier": 2.0},
        "interpretation": "Fixed robustness ablation, not a test-prior estimate. Only absence "
            "from complete training target ownership changes sampled negative weights. Owner "
            "identities are never loaded for weights or supplied as model features. Saved "
            "feature artifacts and their original source contract remain unchanged.",
    }
    contract(run / (NAME + "_contract.json"), spec)
    model_path = run / (NAME + ".txt")
    meta_path = run / (NAME + "_meta.json")
    if model_path.exists() and meta_path.exists():
        if sha256(model_path) != read_json(meta_path)["model_sha256"]:
            raise ValueError("Cached ablation model changed")
        model = lgb.Booster(model_file=str(model_path))
    else:
        training, tuning = [], []
        support_ids = assigned.filter(pl.col("sample") == "support")["s1_id"]
        tune_ids = assigned.filter(pl.col("sample") == "tune")["s1_id"]
        for path in files:
            frame = pl.read_parquet(path)
            fit = frame.filter(pl.col("sample") == "support")
            if fit.height:
                if fit.select("s1_id").unique().join(support_ids.to_frame(), on="s1_id", how="anti").height:
                    raise ValueError("Feature shard has unexpected support owners")
                training.append(sample_negatives(fit).select("target_id", *features, "label", "weight"))
            tune = frame.filter(pl.col("sample") == "tune")
            if tune.height:
                if tune.select("s1_id").unique().join(tune_ids.to_frame(), on="s1_id", how="anti").height:
                    raise ValueError("Feature shard has unexpected Tune owners")
                tuning.append(tune.select(*features, "label"))
        fit = pl.concat(training)
        # Read only target existence. No true owner identifier enters training.
        owned_targets = pl.read_parquet(prepared / "truth.parquet", columns=["target_id"])["target_id"]
        fit, stats = reweight_unowned(fit, owned_targets)
        tune = pl.concat(tuning)
        if (fit.height != parent["training_pairs"] or fit["label"].sum() != parent["training_positives"] or
                tune.height != parent["tune_pairs"]):
            raise ValueError("Pair populations differ from the sibling baseline")
        dfit = lgb.Dataset(_matrix(fit, features), label=fit["label"].to_numpy(),
                           weight=fit["weight"].to_numpy(), feature_name=features, free_raw_data=True)
        dtune = lgb.Dataset(_matrix(tune, features), label=tune["label"].to_numpy(),
                            reference=dfit, feature_name=features, free_raw_data=True)
        nfit, ntune = fit.height, tune.height
        del fit, tune, owned_targets, training, tuning
        gc.collect()
        print(f"{NAME}: {len(support_ids):,} support owners; {nfit:,} sampled training pairs; "
              f"{stats['sampled_unowned_negative_pairs']:,} unowned negatives weighted x2", flush=True)
        start = time.monotonic()
        model = lgb.train(params, dfit, num_boost_round=parent["max_rounds"],
                          valid_sets=[dtune], valid_names=["tune"],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(250)])
        model.save_model(str(model_path))
        write_json(meta_path, {**spec, "training_owners": len(support_ids), "training_pairs": nfit,
            "training_positives": stats["sampled_positive_pairs"], "tune_pairs": ntune,
            "weighting_counts": stats, "best_iteration": model.best_iteration,
            "best_logloss": model.best_score["tune"]["binary_logloss"],
            "training_seconds": time.monotonic() - start, "model_sha256": sha256(model_path)})
    predict_model(run, model, run / "support_features", run / (NAME + "_scores"), features, threads)
    return evaluate_model(prepared, run, run / (NAME + "_scores"), NAME)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    run_experiment(args.prepared, args.run, args.threads)


if __name__ == "__main__":
    main()
