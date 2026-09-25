"""Train changed models, with bounded negative samples and unsampled evaluation."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from plan1.decide import one_owner
from .artifacts import contract, parquet, read_json, sha256, versions, write_json
from .features import FEATURES
from .decisions import error_ledger, threshold_sweep


def sample_negatives(frame: pl.DataFrame, cap=32, hard=8) -> pl.DataFrame:
    if not 0 <= hard < cap:
        raise ValueError("Require 0 <= hard < cap")
    positive=frame.filter(pl.col("label")==1).with_columns(weight=pl.lit(1.0,pl.Float32))
    negative=(frame.filter(pl.col("label")==0).with_columns(
        _hardness=.5*pl.col("name_red_ratio").fill_null(0)+
                  .3*pl.col("addr_token_set").fill_null(0)+.2*pl.col("retrieval_score").fill_null(0))
        .sort(["s1_id","_hardness","target_id"],descending=[False,True,False])
        .with_columns(_rank=pl.int_range(pl.len()).over("s1_id")))
    certain=negative.filter(pl.col("_rank")<hard).with_columns(weight=pl.lit(1.0,pl.Float32))
    rest=(negative.filter(pl.col("_rank")>=hard)
          .with_columns(_total=pl.len().over("s1_id"),
                        _hash=pl.struct("s1_id","target_id").hash(seed=20260925))
          .sort(["s1_id","_hash","target_id"]).with_columns(_pick=pl.int_range(pl.len()).over("s1_id"))
          .filter(pl.col("_pick")<cap-hard)
          .with_columns(weight=(pl.col("_total")/pl.min_horizontal(pl.col("_total"),pl.lit(cap-hard))).cast(pl.Float32)))
    columns=frame.columns+["weight"]
    return pl.concat([positive.select(columns),certain.select(columns),rest.select(columns)])


def _matrix(frame,features):
    return frame.select(features).cast(pl.Float32).fill_null(np.nan).to_numpy()


def fit_model(prepared: Path,run: Path,features: list[str],input_dir: Path,model_name: str,
              train_sample: str,threads=4,max_rounds=6000):
    files=sorted(input_dir.glob("*.parquet"))
    if not files:
        raise ValueError("No feature files")
    manifest=pl.read_parquet(prepared/"manifest.parquet",columns=["s1_id","role","pool"])
    samples=pl.read_parquet(run/"samples.parquet")
    assigned=samples.filter(pl.col("sample")==train_sample).join(manifest,on="s1_id")
    required_pool="fit" if train_sample=="fit" else "c_prob"
    if assigned.height==0 or assigned.filter(pl.col("pool")!=required_pool).height:
        raise ValueError("Unexpected training population; upstream exclusion cannot be proved")
    meta_spec={"features":features,"sample":train_sample,"training_ids_sha":sha256(run/"samples.parquet"),
               "feature_contract":sha256(run/"feature_contract.json"),"threads":threads,
               "max_rounds":max_rounds,"negative_sampling":{"cap":32,"hard":8,"random":24,"importance_weighted":True},
               "dependencies":versions(),"train_code":sha256(Path(__file__)),
               "feature_files":{p.name:sha256(p) for p in files}}
    contract(run/f"{model_name}_contract.json",meta_spec)
    if (run/f"{model_name}.txt").exists() and (run/f"{model_name}_meta.json").exists():
        return lgb.Booster(model_file=str(run/f"{model_name}.txt"))
    training,tuning=[],[]
    for file in files:
        frame=pl.read_parquet(file)
        train=frame.filter(pl.col("sample")==train_sample)
        if train.height:
            training.append(sample_negatives(train).select(*features,"label","weight"))
        tune=frame.filter(pl.col("sample")=="tune")
        if tune.height:
            tuning.append(tune.select(*features,"label"))
    fit=pl.concat(training); tune=pl.concat(tuning)
    if fit["label"].n_unique()!=2 or tune["label"].n_unique()!=2:
        raise ValueError("Both label classes required")
    print(f"{model_name}: {assigned.height:,} training owners; {fit.height:,} sampled pairs; "
          f"{tune.height:,} unsampled Tune pairs; {len(features)} features",flush=True)
    dfit=lgb.Dataset(_matrix(fit,features),label=fit["label"].to_numpy(),weight=fit["weight"].to_numpy(),
                     feature_name=features,free_raw_data=True)
    dtune=lgb.Dataset(_matrix(tune,features),label=tune["label"].to_numpy(),
                      feature_name=features,reference=dfit,free_raw_data=True)
    nfit,positive,ntune=fit.height,fit["label"].sum(),tune.height
    del fit,tune,training,tuning
    gc.collect()
    params={"objective":"binary","metric":"binary_logloss","learning_rate":.05,"num_leaves":63,
            "max_depth":10,"min_data_in_leaf":100,"lambda_l2":5.0,"seed":42,
            "deterministic":True,"force_col_wise":True,"num_threads":threads,"verbosity":-1}
    start=time.monotonic()
    model=lgb.train(params,dfit,num_boost_round=max_rounds,valid_sets=[dtune],valid_names=["tune"],
                    callbacks=[lgb.early_stopping(200),lgb.log_evaluation(250)])
    model.save_model(str(run/f"{model_name}.txt"))
    write_json(run/f"{model_name}_meta.json",{
        **meta_spec,"params":params,"best_iteration":model.best_iteration,"training_pairs":nfit,
        "training_positives":positive,"tune_pairs":ntune,"best_logloss":model.best_score["tune"]["binary_logloss"],
        "cap_reached":model.current_iteration()>=max_rounds,"training_seconds":time.monotonic()-start,
        "model_sha256":sha256(run/f"{model_name}.txt")})
    return model


def predict_model(run: Path,model,input_dir: Path,out_dir: Path,features,threads=4):
    for file in sorted(input_dir.glob("*.parquet")):
        output=out_dir/file.name
        if output.exists():
            continue
        frame=pl.read_parquet(file).filter(pl.col("sample")!="fit")
        if not frame.height:
            continue
        # No in-sample Fit predictions enter support-model training.
        p=model.predict(_matrix(frame,features),num_threads=threads)
        parquet(output,frame.with_columns(p=pl.Series(p)))


def evaluate_model(prepared: Path,run: Path,scored_dir: Path,model_name: str):
    samples=pl.read_parquet(run/"samples.parquet")
    truth=pl.read_parquet(prepared/"truth.parquet").join(samples.select("s1_id"),on="s1_id",how="semi")
    scores=pl.scan_parquet(str(scored_dir/"*.parquet")).select("s1_id","target_id","source","sample","p").collect()
    select=scores.filter(pl.col("sample")=="select")
    roster=samples.filter(pl.col("sample")=="select")["s1_id"]
    best,table=threshold_sweep(select,truth,roster)
    parquet(run/f"{model_name}_threshold_sweep.parquet",table)
    write_json(run/f"{model_name}_decision.json",{"threshold":best["threshold"],"selected_on":"select",
                                                "selection_macro_f05":best["macro_f05"],
                                                "ownership":"highest score per target; applied after threshold"})
    reports={}
    for sample in ("tune","select","development"):
        roster=samples.filter(pl.col("sample")==sample)["s1_id"]
        p=scores.filter(pl.col("sample")==sample)
        accepted=p.filter(pl.col("p")>=best["threshold"])
        final=one_owner(accepted)
        report,errors,per=error_ledger(p,p,accepted,final,truth,roster)
        reports[sample]=report
        parquet(run/f"{model_name}_{sample}_errors.parquet",errors)
        parquet(run/f"{model_name}_{sample}_per_owner.parquet",per)
        pr=report["after_ownership"]
        print(f"{model_name}/{sample}: F0.5={pr['macro_f05']:.6f}, "
              f"P={pr['link_precision']}, R={pr['link_recall']}; panel-only ownership",flush=True)
    write_json(run/f"{model_name}_report.json",reports)
    return reports


def train_direct(prepared,run,threads=4,max_rounds=6000):
    if not (run/"features_complete.json").exists():
        raise ValueError("Feature generation is incomplete")
    model=fit_model(prepared,run,FEATURES,run/"features","direct","fit",threads,max_rounds)
    predict_model(run,model,run/"features",run/"direct_scores",FEATURES,threads)
    return evaluate_model(prepared,run,run/"direct_scores","direct")
