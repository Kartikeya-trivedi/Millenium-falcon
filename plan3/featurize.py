"""Build complete-query features and label only after retrieval."""
from __future__ import annotations

import gc
from pathlib import Path
import polars as pl

from .artifacts import contract, fingerprint, load_prepared, parquet, read_json, sha256, write_json
from .budget import keep_candidates
from .features import FEATURES, rich_features


def query_batches(candidates, queries, targets, size=250):
    """Index complete owner groups once instead of rescanning a country per batch."""
    candidates=candidates.sort("s1_id","target_id")
    queries=queries.sort("entity_id")
    counts=candidates.group_by("s1_id").len().sort("s1_id")
    if counts["s1_id"].to_list()!=queries["entity_id"].to_list():
        raise ValueError("Query records do not match candidate groups")
    offsets=[0]+counts["len"].cum_sum().to_list()
    target_rows={key:i for i,key in enumerate(targets["entity_id"])}
    if len(target_rows)!=targets.height or queries["entity_id"].n_unique()!=queries.height:
        raise ValueError("Duplicate record IDs")
    for start in range(0,queries.height,size):
        end=min(start+size,queries.height)
        c=candidates.slice(offsets[start],offsets[end]-offsets[start])
        q=queries.slice(start,end-start)
        indices=[target_rows[key] for key in c["target_id"].unique(maintain_order=True)]
        t=targets[pl.Series(indices,dtype=pl.UInt32)]
        yield start,c,q,t


def featurize(prepared: Path, views: Path, indexes: Path, run: Path, query_batch=250,
              word_k: int|None=None, missing_k: int|None=None):
    meta = load_prepared(prepared)
    if not (views/"train_complete.json").exists():
        raise ValueError("Additional text views are incomplete")
    default=read_json(run/"budget.json")["chosen"]
    word_k=default["word_k"] if word_k is None else word_k
    missing_k=default["missing_k"] if missing_k is None else missing_k
    spec={"prepared_identity":meta["identity"], "word_k":word_k,"missing_k":missing_k,
          "features":FEATURES,"features_code":sha256(Path(__file__).with_name("features.py")),
          "batching_code":sha256(Path(__file__)),
          "views_contract":sha256(views/"train_contract.json"),
          "query_batch":query_batch,
          "reason":"Explicit bounded scoring experiment; compare with stored Tune budget frontier."}
    contract(run/"feature_contract.json",spec)
    sample=pl.read_parquet(run/"samples.parquet")
    truth=pl.read_parquet(prepared/"truth.parquet").join(sample.select("s1_id"),on="s1_id",how="semi")
    true_by_owner=dict(truth.group_by("s1_id").agg("target_id").iter_rows())
    sample_by_owner=dict(sample.select("s1_id","sample").iter_rows())
    freq=pl.read_parquet(views/"train_name_frequency.parquet")
    cands=pl.scan_parquet(str(run/"candidates/*.parquet"))
    for country in sorted(pl.read_parquet(run/"queries.parquet")["country"].unique().to_list()):
        country_cands=keep_candidates(cands.filter(pl.col("country")==country).collect(),word_k,missing_k)
        qids=country_cands.select(entity_id="s1_id").unique()
        tids=country_cands.select(entity_id="target_id").unique()
        records=[]
        for p in meta["partitions"]:
            if p["split"]!="train" or p["country"]!=country:
                continue
            ids=qids if p["source"]=="S1" else tids
            frame=(pl.scan_parquet(prepared/p["file"]).join(ids.lazy(),on="entity_id",how="semi")
                   .join(pl.scan_parquet(views/p["file"]),on="entity_id")
                   .join(freq.lazy(),on=["name_red","country"],how="left").collect(engine="streaming"))
            records.append(frame)
        qrecords=pl.concat([f for f in records if f["source"][0]=="S1"])
        trecords=pl.concat([f for f in records if f["source"][0]!="S1"])
        idfs={}
        for source in ("S2","S3"):
            d=indexes/"train"/f"{fingerprint(country)[:16]}-{source}"/"word"
            vocab=pl.read_parquet(d/"vocabulary.parquet")
            idfs[source]=(dict(zip(vocab["term"].to_list(),vocab["idf"].to_list())),read_json(d/"index.json")["rows"])
        country_ids=qrecords["entity_id"].sort()
        print(f"Feature batches: {country}, {country_ids.len():,} queries, {country_cands.height:,} pairs",flush=True)
        for start,c,q,t in query_batches(country_cands,qrecords,trecords,query_batch):
            output=run/"features"/f"{fingerprint(country)[:16]}-{start:08d}.parquet"
            if output.exists():
                continue
            ids=q["entity_id"].to_list()
            labels=pl.DataFrame([(qid,tid) for qid in ids for tid in true_by_owner.get(qid,[])],
                                schema={"s1_id":pl.String,"target_id":pl.String},orient="row")
            roles=pl.DataFrame({"s1_id":ids,"sample":[sample_by_owner[qid] for qid in ids]})
            f=rich_features(c,q,t,idfs)
            f=(f.join(labels.with_columns(label=pl.lit(1,pl.Int8)),on=["s1_id","target_id"],how="left",maintain_order="left")
               .with_columns(pl.col("label").fill_null(0)).join(roles,on="s1_id",how="left",maintain_order="left")
               .join(t.select(target_id="entity_id",t_name="name_red",t_address="addr_norm",
                              t_numbers="addr_nums"),on="target_id",how="left",maintain_order="left")
               .with_columns(country=pl.lit(country)))
            parquet(output,f)
            if (start//query_batch)%5==0:
                print(f"  {country}: {min(start+query_batch,country_ids.len()):,}/{country_ids.len():,}",flush=True)
            del c,q,t,f
        del records,qrecords,trecords,country_cands,idfs
        gc.collect()
    write_json(run/"features_complete.json",{"status":"featurized","feature_contract_sha":sha256(run/"feature_contract.json")})
