"""Ablate negative mining with independently produced direct-model scores."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import lightgbm as lgb
import polars as pl

from .artifacts import ROOT, contract, read_json, sha256, write_json
from .features import FEATURES
from .support import SUPPORT_FEATURES
from .train import _matrix, evaluate_model, predict_model


def mine(frame: pl.DataFrame, cap=32, hard=8):
    if not 0 <= hard < cap:
        raise ValueError("Require 0 <= hard < cap")
    if frame["p1"].is_null().any():
        raise ValueError("Independent upstream probabilities are required")
    positive = frame.filter(pl.col("label") == 1).with_columns(weight=pl.lit(1., pl.Float32))
    negative = (frame.filter(pl.col("label") == 0)
                .sort(["s1_id", "p1", "target_id"], descending=[False, True, False])
                .with_columns(_rank=pl.int_range(pl.len()).over("s1_id")))
    fixed = negative.filter(pl.col("_rank") < hard).with_columns(weight=pl.lit(1., pl.Float32))
    rest = (negative.filter(pl.col("_rank") >= hard)
        .with_columns(_total=pl.len().over("s1_id"), _hash=pl.struct("s1_id", "target_id").hash(seed=20260925))
        .sort(["s1_id", "_hash", "target_id"]).with_columns(_pick=pl.int_range(pl.len()).over("s1_id"))
        .filter(pl.col("_pick") < cap-hard)
        .with_columns(weight=(pl.col("_total")/pl.min_horizontal(pl.col("_total"), pl.lit(cap-hard))).cast(pl.Float32)))
    columns = frame.columns + ["weight"]
    return pl.concat([positive.select(columns), fixed.select(columns), rest.select(columns)])


def run_experiment(prepared: Path, run: Path, threads=4):
    features = FEATURES + SUPPORT_FEATURES
    name = "support_hardneg"
    parent = read_json(run / "support_meta.json")
    seed_spec = read_json(run / "support_features_contract.json")
    if seed_spec["upstream_training_pool"] != "fit" or seed_spec["support_training_pool"] != "c_prob":
        raise ValueError("Independent upstream scoring cannot be established")
    samples = pl.read_parquet(run / "samples.parquet")
    owners = samples.filter(pl.col("sample") == "support").join(
        pl.read_parquet(prepared / "manifest.parquet", columns=["s1_id", "pool"]), on="s1_id")
    if not owners.height or owners.filter(pl.col("pool") != "c_prob").height:
        raise ValueError("Wrong support training population")
    files = sorted((run / "support_features").glob("*.parquet"))
    spec = {"parent_model_sha": parent["model_sha256"], "features": features,
            "source_sha": sha256(Path(__file__)), "train_helpers_sha": sha256(Path(__file__).with_name("train.py")),
            "sampling": {"cap": 32, "hard": 8, "rank_by": "p1", "importance_weighted": True},
            "feature_files": {p.name: sha256(p) for p in files}, "threads": threads,
            "support_feature_contract_sha": sha256(run / "support_features_contract.json")}
    contract(run / (name + "_contract.json"), spec)
    model_path = run / (name + ".txt")
    if model_path.exists() and (run / (name + "_meta.json")).exists():
        model = lgb.Booster(model_file=str(model_path))
    else:
        training, tuning = [], []
        for path in files:
            frame = pl.read_parquet(path)
            fit = frame.filter(pl.col("sample") == "support")
            if fit.height:
                training.append(mine(fit).select(*features, "label", "weight"))
            tune = frame.filter(pl.col("sample") == "tune")
            if tune.height:
                tuning.append(tune.select(*features, "label"))
        fit, tune = pl.concat(training), pl.concat(tuning)
        dfit = lgb.Dataset(_matrix(fit, features), label=fit["label"].to_numpy(), weight=fit["weight"].to_numpy(),
                          feature_name=features, free_raw_data=True)
        dtune = lgb.Dataset(_matrix(tune, features), label=tune["label"].to_numpy(),
                           reference=dfit, feature_name=features, free_raw_data=True)
        nfit, ntune = fit.height, tune.height
        del fit, tune, training, tuning
        gc.collect()
        params = {**parent["params"], "num_threads": threads}
        print(f"{name}: {owners.height:,} support owners, {nfit:,} sampled training pairs", flush=True)
        start = time.monotonic()
        model = lgb.train(params, dfit, num_boost_round=parent["max_rounds"], valid_sets=[dtune], valid_names=["tune"],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(250)])
        model.save_model(str(model_path))
        write_json(run / (name + "_meta.json"), {**spec, "sample": "support", "params": params,
            "training_pairs": nfit, "tune_pairs": ntune, "best_iteration": model.best_iteration,
            "training_seconds": time.monotonic()-start, "model_sha256": sha256(model_path)})
    predict_model(run, model, run / "support_features", run / (name + "_scores"), features, threads)
    return evaluate_model(prepared, run, run / (name + "_scores"), name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    run_experiment(args.prepared, args.run, args.threads)


if __name__ == "__main__":
    main()
