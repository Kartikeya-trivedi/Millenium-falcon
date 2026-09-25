"""Enrich the four error classes and measure target orphan populations separately."""
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl

from plan1.decide import one_owner
from .artifacts import ROOT, load_prepared, parquet, read_json, records, sha256, write_json

KEYS = ["s1_id", "target_id"]


def orphan_summary(scored, accepted, ownership, total_targets):
    """Owner labels are diagnostic only; never return them as matcher features."""
    if ownership["target_id"].n_unique() != ownership.height:
        raise ValueError("A target has multiple true owners")
    if total_targets < ownership.height:
        raise ValueError("Target population smaller than labeled targets")
    joined = scored.join(ownership.rename({"s1_id": "known_owner"}), on="target_id", how="left")
    negative = joined.filter(pl.col("label") == 0)
    false = accepted.join(joined.select(*KEYS, "label", "known_owner"), on=KEYS).filter(pl.col("label") == 0)
    def counts(frame):
        unowned = frame.filter(pl.col("known_owner").is_null())
        return {"pairs": frame.height, "unowned_pairs": unowned.height,
                "unowned_pair_share": unowned.height / frame.height if frame.height else None,
                "distinct_targets": frame["target_id"].n_unique(),
                "distinct_unowned_targets": unowned["target_id"].n_unique()}
    return {"full_target_corpus": {"targets": total_targets, "unowned_targets": total_targets-ownership.height,
                                  "unowned_share": (total_targets-ownership.height)/total_targets},
            "retrieved_negative_pairs": counts(negative), "false_accepted_pairs_before_ownership": counts(false)}


def normalized_targets(prepared, meta, ids):
    wanted = ids.rename({"target_id": "entity_id"}).unique().lazy()
    parts = []
    for part in meta["partitions"]:
        if part["split"] == "train" and part["source"] != "S1":
            parts.append(pl.scan_parquet(prepared / part["file"]).join(wanted, on="entity_id", how="semi")
                         .select(target_id="entity_id", target_country="country", target_source="source",
                                 target_name="name_red", target_address="addr_norm",
                                 target_nonlatin="name_nonlatin", target_missing_address="addr_missing",
                                 target_numbers="addr_nums").collect(engine="streaming"))
    return pl.concat(parts)


def link_slices(true, candidates, final, target_meta):
    t = true.join(target_meta, on="target_id", how="left")
    t = (t.join(candidates.select(KEYS).with_columns(retrieved=pl.lit(True)), on=KEYS, how="left")
         .join(final.select(KEYS).with_columns(accepted=pl.lit(True)), on=KEYS, how="left")
         .with_columns(pl.col("retrieved", "accepted").fill_null(False)))
    result = {}
    for field in ("target_country", "target_source", "target_nonlatin", "target_missing_address"):
        result[field] = t.group_by(field).agg(true_links=pl.len(), retrieved=pl.col("retrieved").sum(),
                                            accepted=pl.col("accepted").sum()).with_columns(
            candidate_recall=pl.col("retrieved")/pl.col("true_links"),
            accepted_recall=pl.col("accepted")/pl.col("true_links")).sort(field).to_dicts()
    return result, t


def diagnose(prepared: Path, run: Path, model="support", sample="tune"):
    meta = load_prepared(prepared)
    samples = pl.read_parquet(run / "samples.parquet").filter(pl.col("sample") == sample)
    if not samples.height:
        raise ValueError("Empty diagnostic sample")
    truth_all = pl.read_parquet(prepared / "truth.parquet")
    truth = truth_all.join(samples.select("s1_id"), on="s1_id", how="semi")
    scores = pl.scan_parquet(str(run / (model + "_scores") / "*.parquet")).filter(pl.col("sample") == sample)
    available = scores.collect_schema().names()
    wanted = KEYS + ["p", "label", "country", "source", "retrieval_score", "retrieval_rank", "num_conflict",
                    "cand_name_nonlatin", "cand_addr_missing", "target_indic_unknown_share", "target_indic_token_count",
                    "name_coverage_query", "name_coverage_target", "address_coverage_query", "address_coverage_target"]
    wanted += [c for c in available if "support_" in c or c in ("p1", "best_other_p", "p1_gap_other")]
    scored = scores.select(wanted).collect()
    decision = read_json(run / (model + "_decision.json"))
    accepted = scored.filter(pl.col("p") >= decision["threshold"])
    final = one_owner(accepted)
    target_total = sum(p["rows"] for p in meta["partitions"] if p["split"] == "train" and p["source"] != "S1")
    orphan = orphan_summary(scored, accepted.select(KEYS), truth_all, target_total)
    errors = pl.read_parquet(run / f"{model}_{sample}_errors.parquet")
    targets = normalized_targets(prepared, meta, pl.concat([truth.select("target_id"), errors.select("target_id")]))
    slices, true_flags = link_slices(truth, scored, final, targets)
    owner_flags = true_flags.group_by("s1_id").agg(
        true_link_count=pl.len(), has_nonlatin_true=pl.col("target_nonlatin").any(),
        has_missing_address_true=pl.col("target_missing_address").any())
    countries = pl.read_parquet(prepared / "manifest.parquet", columns=["s1_id", "country"])
    per = (pl.read_parquet(run / f"{model}_{sample}_per_owner.parquet").join(countries, on="s1_id")
           .join(owner_flags, on="s1_id", how="left").with_columns(
               pl.col("true_link_count").fill_null(0),
               pl.col("has_nonlatin_true", "has_missing_address_true").fill_null(False))
           .with_columns(singleton=pl.col("true_link_count") == 0))
    owner_slices = {}
    loss_cols = ["retrieval_loss", "rejection_loss", "false_acceptance_loss", "ownership_loss"]
    for field in ("country", "singleton", "has_nonlatin_true", "has_missing_address_true"):
        owner_slices[field] = per.group_by(field).agg(owners=pl.len(), macro_f05=pl.col("f3").mean(),
            *[pl.col(c).mean() for c in loss_cols]).sort(field).to_dicts()

    # All four error classes receive raw/normalized fields and candidate context.
    # Missing true links can exist in the wider proposal pool but not C_final.
    context = scored.drop("p")
    enriched = errors.join(context, on=KEYS, how="left").join(targets, on="target_id", how="left")
    proposal = (pl.scan_parquet(str(run / "candidates/*.parquet"))
                .join(errors.select(KEYS).lazy(), on=KEYS, how="semi")
                .select(*KEYS, "word_rank", "missing_rank", "char_rank", "name_rank").collect())
    enriched = enriched.join(proposal, on=KEYS, how="left").with_columns(
        decision_threshold=pl.lit(decision["threshold"]),
        proposal_found=pl.any_horizontal(pl.col(c).is_not_null() for c in ("word_rank", "missing_rank", "char_rank", "name_rank")))
    owner_best = scored.group_by("s1_id").agg(owner_best_p=pl.col("p").max())
    enriched = enriched.join(owner_best, on="s1_id", how="left").join(
        truth_all.rename({"s1_id": "known_owner"}), on="target_id", how="left")
    winners = final.select("target_id", winning_claimant="s1_id", winning_score="p")
    enriched = enriched.join(winners, on="target_id", how="left").with_columns(
        winner_margin=pl.col("winning_score")-pl.col("p"))
    raw_targets = []
    dataset = Path(meta["dataset"])
    for source in (2, 3):
        raw_targets.append(records(dataset / "train" / f"train_source{source}.tsv")
            .join(errors.select(entity_id="target_id").unique().lazy(), on="entity_id", how="semi")
            .select(target_id="entity_id", raw_target_name="business_name", raw_target_address="business_address")
            .collect(engine="streaming"))
    enriched = enriched.join(pl.concat(raw_targets), on="target_id", how="left")
    raw_queries = (records(dataset / "train/train_source1.tsv")
        .join(errors.select(entity_id="s1_id").unique().lazy(), on="entity_id", how="semi")
        .select(s1_id="entity_id", raw_query_name="business_name", raw_query_address="business_address")
        .collect(engine="streaming"))
    queries = pl.read_parquet(run / "queries.parquet").select(
        s1_id="entity_id", query_name="name_red", query_address="addr_norm", query_numbers="addr_nums")
    enriched = enriched.join(raw_queries, on="s1_id", how="left").join(queries, on="s1_id", how="left")
    if enriched.height != errors.height or enriched["raw_target_name"].is_null().any():
        raise ValueError("Error context lookup lost or multiplied rows")
    removed = accepted.select(*KEYS, "p", "label").join(final.select(KEYS), on=KEYS, how="anti")
    out = run / "diagnostics" / model / sample
    parquet(out / "errors_context.parquet", enriched)
    parquet(out / "ownership_removed.parquet", removed)
    parquet(out / "owners.parquet", per)
    report = {"model": model, "sample": sample, "owners": samples.height,
              "source_sha256": sha256(Path(__file__)), "model_sha256": sha256(run / (model + ".txt")),
              "decision": decision, "orphan_populations": orphan, "true_link_slices": slices,
              "owner_slices": owner_slices,
              "interpretation": "Owner flags use true-link metadata only for diagnostics. No diagnostic label enters model features. "
              "Ownership is panel-only; orphan-negative shares use candidate pairs, not all targets."}
    write_json(out / "report.json", report)
    print(f"{model}/{sample}: {enriched.height:,} error records enriched.", flush=True)
    for name, counts in orphan.items():
        print(f"  {name}: {counts}", flush=True)
    for field, rows in slices.items():
        print(f"  {field}: {rows}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", default="support")
    p.add_argument("--sample", choices=("tune", "select", "development"), default="tune")
    args = p.parse_args()
    if not args.model.replace("_", "").isalnum():
        p.error("Invalid model name")
    diagnose(args.prepared, args.run, args.model, args.sample)


if __name__ == "__main__":
    main()
