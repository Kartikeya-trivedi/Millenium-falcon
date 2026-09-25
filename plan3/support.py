"""Use independently scored sibling records as evidence; never propagate labels."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .artifacts import contract, parquet, read_json, sha256, write_json
from .features import FEATURES
from .train import fit_model, predict_model, evaluate_model

SUPPORT_FEATURES = ["p1","best_other_p","p1_gap_other"] + [
    f"{source}_support_{name}" for source in ("same","cross") for name in
    ("count","max_p","name_similarity","address_similarity","joint_similarity","number_conflict_share")]


def seeds_for(rows, threshold, exclude=None, source=None):
    seen=set(); selected=[]
    for row in sorted(rows,key=lambda r:(-r["p"],r["target_id"])):
        if row["p"]<threshold or row["target_id"]==exclude or (source is not None and row["source"]!=source):
            continue
        key=(row["t_name"],row["t_address"])
        if key in seen or not any(key):
            continue
        seen.add(key); selected.append(row)
        if len(selected)==2:
            break
    return selected


def _evidence(target,seeds):
    if not seeds:
        return [0,np.nan,np.nan,np.nan,np.nan,np.nan]
    names=[]; addresses=[]; joint=[]; conflicts=[]
    for s in seeds:
        n=fuzz.token_set_ratio(target["t_name"],s["t_name"])/100 if target["t_name"] and s["t_name"] else np.nan
        a=fuzz.token_set_ratio(target["t_address"],s["t_address"])/100 if target["t_address"] and s["t_address"] else np.nan
        names.append(n); addresses.append(a)
        if np.isfinite(n) and np.isfinite(a):
            joint.append((n+a)/2)
        if target["t_numbers"] and s["t_numbers"]:
            conflicts.append(float(not (set(target["t_numbers"]) & set(s["t_numbers"]))))
    def maximum(values):
        finite=[v for v in values if np.isfinite(v)]
        return max(finite) if finite else np.nan
    return [len(seeds),max(s["p"] for s in seeds),maximum(names),maximum(addresses),
            maximum(joint),float(np.mean(conflicts)) if conflicts else np.nan]


def support_features(frame: pl.DataFrame, threshold: float) -> pl.DataFrame:
    rows=[]
    fields=["s1_id","target_id","source","p","t_name","t_address","t_numbers"]
    for group in frame.select(fields).partition_by("s1_id",maintain_order=True):
        members=list(group.iter_rows(named=True))
        ordered=sorted(members,key=lambda r:(-r["p"],r["target_id"]))
        pools={}
        for source in ("S2","S3"):
            groups={}
            for candidate in ordered:
                key=(candidate["t_name"],candidate["t_address"])
                if candidate["source"]!=source or candidate["p"]<threshold or not any(key):
                    continue
                # Excluding one candidate can remove at most one evidence group.
                # Three groups and two representatives each preserve the exact top two.
                if key not in groups and len(groups)==3:
                    continue
                group_rows=groups.setdefault(key,[])
                if len(group_rows)<2:
                    group_rows.append(candidate)
            pools[source]=[r for group_rows in groups.values() for r in group_rows]
        for row in members:
            best=(ordered[0]["p"] if ordered[0]["target_id"]!=row["target_id"] else
                  ordered[1]["p"] if len(ordered)>1 else np.nan)
            values=[row["p"],best,row["p"]-best]
            other_source="S3" if row["source"]=="S2" else "S2"
            same=seeds_for(pools[row["source"]],threshold,row["target_id"],row["source"])
            cross=seeds_for(pools[other_source],threshold,row["target_id"],other_source)
            values+=_evidence(row,same)+_evidence(row,cross)
            rows.append((row["s1_id"],row["target_id"],*values))
    out=pl.DataFrame(rows,schema={"s1_id":pl.String,"target_id":pl.String,
                                 **{c:pl.Float32 for c in SUPPORT_FEATURES}},orient="row")
    return out.with_columns(pl.col(SUPPORT_FEATURES).fill_nan(None))


def competing_claims(scored: pl.DataFrame) -> pl.DataFrame:
    """Global aggregation over the supplied population; never independently per chunk.

    This diagnostic is deliberately not a learned pilot feature. Small-panel
    rival counts cannot stand in for the eventual full inference population.
    """
    if scored.select("s1_id","target_id").unique().height!=scored.height:
        raise ValueError("Duplicate pair scores")
    ranked=(scored.sort(["target_id","p","s1_id"],descending=[False,True,False])
            .with_columns(_rank=pl.int_range(pl.len()).over("target_id"),
                          claimant_count=pl.len().over("target_id"),
                          _best=pl.col("p").first().over("target_id")))
    second=ranked.filter(pl.col("_rank")==1).select("target_id",_second="p")
    return (ranked.join(second,on="target_id",how="left")
            .with_columns(best_other_claim=pl.when(pl.col("_rank")==0).then(pl.col("_second")).otherwise(pl.col("_best"))))


def choose_seed_threshold(run):
    tune=(pl.scan_parquet(str(run/"direct_scores/*.parquet")).filter(pl.col("sample")=="tune")
          .select("s1_id","target_id","source","p","label","t_name","t_address").collect())
    groups=[list(g.iter_rows(named=True)) for g in tune.partition_by("s1_id",maintain_order=True)]
    trials=[]
    for threshold in (.9,.95,.98,.99,.995,.999):
        chosen=[s for group in groups for source in ("S2","S3") for s in seeds_for(group,threshold,source=source)]
        correct=sum(s["label"] for s in chosen)
        trials.append({"threshold":threshold,"seed_links":len(chosen),"correct_seed_links":correct,
                       "observed_precision":correct/len(chosen) if chosen else None,
                       "owner_coverage":len({s["s1_id"] for s in chosen})/len(groups) if groups else 0})
    eligible=[r for r in trials if r["seed_links"]>=200 and r["observed_precision"]>=.995]
    threshold=eligible[0]["threshold"] if eligible else .999
    write_json(run/"seed_selection.json",{"threshold":threshold,"selected_on":"tune",
                                         "trials":trials,"precision_interpretation":"Observed seed precision, not a population lower confidence bound."})
    return threshold


def train_support(prepared: Path,run: Path,threads=4,max_rounds=6000):
    direct=read_json(run/"direct_meta.json")
    if direct["sample"]!="fit":
        raise ValueError("M0 was not trained exclusively on the declared Fit population")
    if read_json(prepared/"translit.json")["training_role"]!="fit":
        raise ValueError("Dictionary training role cannot exclude support-training owners")
    threshold=choose_seed_threshold(run)
    contract(run/"support_features_contract.json",{"parent_model_sha256":direct["model_sha256"],
             "support_code":sha256(Path(__file__)),"seed_threshold":threshold,
             "support_training_pool":"c_prob","upstream_training_pool":"fit",
             "features":FEATURES+SUPPORT_FEATURES,"competitor_probabilities_used_in_model":False})
    for path in sorted((run/"direct_scores").glob("*.parquet")):
        output=run/"support_features"/path.name
        if output.exists():
            continue
        frame=pl.read_parquet(path)
        if (frame["sample"]=="fit").any():
            raise ValueError("In-sample upstream predictions are forbidden for support training")
        extra=support_features(frame,threshold)
        parquet(output,frame.join(extra,on=["s1_id","target_id"],how="left",maintain_order="left"))
    features=FEATURES+SUPPORT_FEATURES
    model=fit_model(prepared,run,features,run/"support_features","support","support",threads,max_rounds)
    predict_model(run,model,run/"support_features",run/"support_scores",features,threads)
    return evaluate_model(prepared,run,run/"support_scores","support")
