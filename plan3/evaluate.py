"""Full-roster metrics and retrieval ablations. Labels never affect retrieval."""
from __future__ import annotations

import numpy as np
import polars as pl

from plan1.metric import per_s1, summarize


def score(pred: pl.DataFrame, truth: pl.DataFrame, roster: pl.Series) -> dict:
    if roster.n_unique() != roster.len() or not roster.len():
        raise ValueError("Evaluation roster must be nonempty and unique")
    result = summarize(per_s1(pred, truth, roster))
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in result.items()}


def retrieval_report(candidates: pl.DataFrame, truth: pl.DataFrame, roster: pl.Series,
                     target_info: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    uni = pl.DataFrame({"s1_id": roster})
    true = truth.join(uni, on="s1_id", how="semi")
    if candidates.select("s1_id", "target_id").unique().height != candidates.height:
        raise ValueError("Candidate pairs must be unique")
    if candidates.join(uni, on="s1_id", how="anti").height:
        raise ValueError("Candidate outside evaluation roster")
    filters = {
        "W50": pl.col("word_rank") <= 50,
        "W_all": pl.col("word_rank").is_not_null(),
        "W+C": pl.col("word_rank").is_not_null() | pl.col("char_rank").is_not_null(),
        "W+N": pl.col("word_rank").is_not_null() | pl.col("name_rank").is_not_null(),
        "W+M": pl.col("word_rank").is_not_null() | pl.col("missing_rank").is_not_null(),
        "union": pl.lit(True),
    }
    report = {}
    for name, expr in filters.items():
        c = candidates.filter(expr.fill_null(False)).select("s1_id", "target_id")
        found = true.join(c, on=["s1_id", "target_id"], how="semi")
        counts = uni.join(c.group_by("s1_id").len(name="n"), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
        report[name] = {
            "queries": roster.len(), "true_links": true.height, "found_links": found.height,
            "missed_links": true.height - found.height,
            "candidate_recall": found.height / true.height if true.height else None,
            "candidate_oracle_macro_f05": score(found, true, roster)["macro_f05"],
            "candidates": c.height, "mean_candidates": counts["n"].mean(),
            "p95_candidates": counts["n"].quantile(.95, interpolation="nearest"),
            "max_candidates": counts["n"].max(), "zero_candidate_queries": counts.filter(pl.col("n") == 0).height,
        }
    membership = candidates.select("s1_id", "target_id", *[f"{c}_rank" for c in ("word", "char", "name", "missing")])
    joined = true.join(membership, on=["s1_id", "target_id"], how="left").with_columns(
        found=pl.any_horizontal([pl.col(f"{c}_rank").is_not_null() for c in ("word", "char", "name", "missing")]))
    report["unique_channel_recoveries"] = {}
    for channel in ("word", "char", "name", "missing"):
        only = pl.col(channel + "_rank").is_not_null() & pl.all_horizontal(
            [pl.col(c + "_rank").is_null() for c in ("word", "char", "name", "missing") if c != channel])
        report["unique_channel_recoveries"][channel] = joined.filter(only).height
    joined = joined.join(target_info, on="target_id", how="left")
    if joined["source"].is_null().any():
        raise ValueError("Target metadata missing for some true links")
    report["slices"] = {}
    for column in ("country", "source", "name_nonlatin", "addr_missing"):
        groups = joined.group_by(column).agg(links=pl.len(), found=pl.col("found").sum()).sort(column)
        report["slices"][column] = [{**r, "recall": r["found"] / r["links"]} for r in groups.iter_rows(named=True)]
    report["interpretation"] = "Development queries; full target partitions including orphan targets. Not fresh Audit or test recall."
    return report, joined.filter(~pl.col("found"))
