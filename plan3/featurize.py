"""Build complete-query features and label only after retrieval."""
from __future__ import annotations

import gc
from pathlib import Path
import polars as pl

from .artifacts import contract, fingerprint, load_prepared, parquet, read_json, sha256, write_json
from .budget import keep_candidates
from .features import FEATURES, rich_features


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
          "views_contract":sha256(views/"train_contract.json"),
          "query_batch":query_batch,
          "reason":"Explicit bounded scoring experiment; compare with stored Tune budget frontier."}
    contract(run/"feature_contract.json",spec)
    sample=pl.read_parquet(run/"samples.parquet")
    truth=pl.read_parquet(prepared/"truth.parquet").join(sample.select("s1_id"),on="s1_id",how="semi")
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
        for start in range(0,country_ids.len(),query_batch):
            output=run/"features"/f"{fingerprint(country)[:16]}-{start:08d}.parquet"
            if output.exists():
                continue
            ids=country_ids.slice(start,query_batch)
            c=country_cands.join(pl.DataFrame({"s1_id":ids}),on="s1_id",how="semi").sort("s1_id","target_id")
            q=qrecords.join(pl.DataFrame({"entity_id":ids}),on="entity_id",how="semi")
            t=trecords.join(c.select(entity_id="target_id").unique(),on="entity_id",how="semi")
            f=rich_features(c,q,t,idfs)
            f=(f.join(truth.with_columns(label=pl.lit(1,pl.Int8)),on=["s1_id","target_id"],how="left",maintain_order="left")
               .with_columns(pl.col("label").fill_null(0)).join(sample,on="s1_id",how="left",maintain_order="left")
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
