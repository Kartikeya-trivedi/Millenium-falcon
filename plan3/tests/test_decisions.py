import polars as pl
import pytest
from plan3.decisions import error_ledger, threshold_sweep
from plan3.train import sample_negatives


def pairs(rows):
    return pl.DataFrame(rows,schema={"s1_id":pl.String,"target_id":pl.String},orient="row")


def test_four_losses_include_rejections_and_ownership_and_sum():
    truth=pairs([("a","x"),("a","y"),("b","z"),("c","w")])
    c=pairs([("a","x"),("a","bad"),("b","z"),("c","w")])
    a=pairs([("a","x"),("a","bad"),("b","z")])
    final=pairs([("a","x")])
    scored=c.with_columns(p=pl.lit(.8))
    report,errors,per=error_ledger(c,scored,a,final,truth,pl.Series(["a","b","c","d"]))
    assert set(errors["error_class"])=={"retrieval_miss","true_candidate_rejected","false_acceptance","true_link_lost_at_ownership"}
    assert sum(report["loss_decomposition"].values())==pytest.approx(1-report["after_ownership"]["macro_f05"])
    assert per.filter(pl.col("s1_id")=="d")["f3"][0]==1


def test_empty_answer_can_win_threshold_selection():
    truth=pairs([])
    s=pairs([("a","x")]).with_columns(p=pl.lit(.9))
    best,_=threshold_sweep(s,truth,pl.Series(["a","b"]))
    assert best["threshold"]>.9 and best["macro_f05"]==1


def test_negative_sampling_preserves_positives_and_weight_mass():
    f=pl.DataFrame({"s1_id":["q"]*102,"target_id":[f"t{i}" for i in range(102)],
                    "label":[1,1]+[0]*100,"name_red_ratio":[.5]*102,
                    "addr_token_set":[.5]*102,"retrieval_score":[.5]*102})
    selected=sample_negatives(f)
    assert selected["label"].sum()==2
    assert selected.filter(pl.col("label")==0).height==32
    assert selected["weight"].sum()==pytest.approx(102)
    assert selected.equals(sample_negatives(f))
