"""Threshold selection, global ownership and complete error accounting."""
from __future__ import annotations

import numpy as np
import polars as pl

from plan1.metric import per_s1
from plan1.decide import one_owner
from .evaluate import score


def threshold_sweep(scored: pl.DataFrame, truth: pl.DataFrame, roster: pl.Series):
    lookup=pl.DataFrame({"s1_id":roster}).with_row_index("_q")
    labels=truth.select("s1_id","target_id").unique().with_columns(_true=pl.lit(1))
    p=(scored.select("s1_id","target_id","p").join(lookup,on="s1_id")
       .join(labels,on=["s1_id","target_id"],how="left").with_columns(pl.col("_true").fill_null(0)))
    if p.height!=scored.height:
        raise ValueError("Scored query outside threshold roster")
    g=(lookup.join(truth.group_by("s1_id").len(name="g"),on="s1_id",how="left")
       .sort("_q")["g"].fill_null(0).to_numpy().astype(np.float64))
    idx,prob,correct=p["_q"].to_numpy(),p["p"].to_numpy(),p["_true"].to_numpy()
    if not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any():
        raise ValueError("Invalid probabilities")
    rows=[]
    for threshold in np.r_[np.arange(.05,1,.005),1.000001]:
        mask=prob>=threshold
        k=np.bincount(idx[mask],minlength=roster.len())
        t=np.bincount(idx[mask],weights=correct[mask],minlength=roster.len())
        f=np.ones(roster.len(),np.float64)
        denominator=.25*g+k
        np.divide(1.25*t,denominator,out=f,where=denominator>0)
        rows.append({"threshold":float(threshold),"macro_f05":float(f.mean())})
    table=pl.DataFrame(rows)
    best=table.sort(["macro_f05","threshold"],descending=[True,True]).row(0,named=True)
    return best,table


def error_ledger(candidates,scored,accepted,final,truth,roster):
    """Ordered counterfactual loss decomposition; ownership contribution is signed."""
    keys=["s1_id","target_id"]
    uni=pl.DataFrame({"s1_id":roster})
    truth=truth.join(uni,on="s1_id",how="semi").select(keys).unique()
    c=candidates.select(keys).unique()
    a=accepted.select(keys).unique()
    z=final.select(keys).unique()
    if a.join(c,on=keys,how="anti").height or z.join(a,on=keys,how="anti").height:
        raise ValueError("Decision output is outside scored candidate/accepted sets")
    retrieved_true=truth.join(c,on=keys,how="semi")
    accepted_true=truth.join(a,on=keys,how="semi")
    classes=[
        truth.join(c,on=keys,how="anti").with_columns(error_class=pl.lit("retrieval_miss")),
        retrieved_true.join(a,on=keys,how="anti").with_columns(error_class=pl.lit("true_candidate_rejected")),
        a.join(truth,on=keys,how="anti").with_columns(error_class=pl.lit("false_acceptance")),
        accepted_true.join(z,on=keys,how="anti").with_columns(error_class=pl.lit("true_link_lost_at_ownership")),
    ]
    errors=pl.concat(classes).join(scored.select(*keys,"p"),on=keys,how="left")
    errors=errors.join(z.with_columns(in_final=pl.lit(True)),on=keys,how="left").with_columns(pl.col("in_final").fill_null(False))
    stages=[retrieved_true,accepted_true,a,z]
    scores=[per_s1(s,truth,roster).select("s1_id",pl.col("f").alias(f"f{i}")) for i,s in enumerate(stages)]
    per=scores[0]
    for s in scores[1:]:
        per=per.join(s,on="s1_id")
    per=per.with_columns(retrieval_loss=1-pl.col("f0"),rejection_loss=pl.col("f0")-pl.col("f1"),
                         false_acceptance_loss=pl.col("f1")-pl.col("f2"),ownership_loss=pl.col("f2")-pl.col("f3"))
    columns=["retrieval_loss","rejection_loss","false_acceptance_loss","ownership_loss"]
    loss={col:per[col].mean() for col in columns}
    if abs(sum(loss.values())-(1-per["f3"].mean()))>1e-10:
        raise ValueError("Loss decomposition does not sum to final score loss")
    return {"before_ownership":score(a,truth,roster),"after_ownership":score(z,truth,roster),
            "loss_decomposition":loss,"ownership_scope":"This evaluated query population only; not a full-population ownership estimate.",
            "error_counts":{r["error_class"]:r["len"] for r in errors.group_by("error_class").len().iter_rows(named=True)}},errors,per
