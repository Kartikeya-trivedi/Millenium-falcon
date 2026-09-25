import polars as pl
from plan3.support import support_features, competing_claims


def test_self_support_excluded_and_duplicate_votes_deduplicated():
    frame=pl.DataFrame({
        "s1_id":["q"]*4,"target_id":["a","b","c","d"],"source":["S2","S2","S2","S3"],
        "p":[.99,.99,.99,.99],"t_name":["alpha"]*4,"t_address":["12 road"]*4,
        "t_numbers":[["12"]]*4})
    f=support_features(frame,.95)
    assert (f["same_support_count"]<=1).all()
    assert f.filter(pl.col("target_id")=="a")["cross_support_count"][0]==1
    solo=support_features(frame.head(1),.95)
    assert solo["same_support_count"][0]==0 and solo["cross_support_count"][0]==0
    assert solo["best_other_p"][0] is None


def test_global_claims_excludes_self_and_handles_top_score_ties():
    s=pl.DataFrame({"s1_id":["q1","q2","q3"],"target_id":["x"]*3,"p":[.9,.9,.8]})
    r=competing_claims(s)
    assert r["best_other_claim"].to_list()==[.9,.9,.9]
    assert r["claimant_count"].to_list()==[3,3,3]
    one=competing_claims(s.head(1))
    assert one["best_other_claim"][0] is None


def test_support_features_and_model_have_separate_contracts(tmp_path, monkeypatch):
    from plan3 import support
    from plan3.artifacts import contract, read_json, write_json
    run, prepared = tmp_path / "run", tmp_path / "prepared"
    write_json(run / "direct_meta.json", {"sample": "fit", "model_sha256": "parent"})
    write_json(prepared / "translit.json", {"training_role": "fit"})
    monkeypatch.setattr(support, "choose_seed_threshold", lambda _: .95)
    def fit(*args):
        contract(run / "support_contract.json", {"sample": "support", "kind": "model"})
        return object()
    monkeypatch.setattr(support, "fit_model", fit)
    monkeypatch.setattr(support, "predict_model", lambda *args: None)
    monkeypatch.setattr(support, "evaluate_model", lambda *args: {"status": "fixture"})
    assert support.train_support(prepared, run) == {"status": "fixture"}
    assert read_json(run / "support_features_contract.json")["parent_model_sha256"] == "parent"
    assert read_json(run / "support_contract.json")["kind"] == "model"
    assert support.train_support(prepared, run) == {"status": "fixture"}
