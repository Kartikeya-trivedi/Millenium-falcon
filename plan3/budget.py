"""Choose a bounded candidate configuration on Tune; never use fresh Audit."""
from __future__ import annotations

from pathlib import Path
import polars as pl

from .artifacts import contract, parquet, read_json, sha256, write_json
from .evaluate import score


def keep_candidates(candidates: pl.DataFrame, word_k: int, missing_k: int) -> pl.DataFrame:
    return candidates.filter(((pl.col("word_rank") <= word_k).fill_null(False)) |
                             ((pl.col("missing_rank") <= missing_k).fill_null(False)))


def select_budget(prepared: Path, run: Path, mean_limit=400, p95_limit=500, target_recall=.995) -> dict:
    if not (run / "retrieval_complete.json").exists():
        raise ValueError("Retrieval is incomplete")
    samples = pl.read_parquet(run / "samples.parquet")
    roster = samples.filter(pl.col("sample") == "tune")["s1_id"]
    uni = pl.DataFrame({"s1_id": roster})
    truth = pl.read_parquet(prepared / "truth.parquet").join(uni, on="s1_id", how="semi")
    candidates = pl.scan_parquet(str(run / "candidates/*.parquet")).filter(pl.col("sample") == "tune").collect()
    rows = []
    max_k = read_json(run / "retrieval_contract.json")["budgets"]
    for word in (50, 100, 200):
        if word > max_k["word"]:
            continue
        for missing in (0, 25, 50, 100):
            if missing > max_k.get("missing", 0):
                continue
            c = keep_candidates(candidates, word, missing)
            found = truth.join(c.select("s1_id","target_id"), on=["s1_id","target_id"], how="semi")
            counts = uni.join(c.group_by("s1_id").len(name="n"), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
            mean, p95 = counts["n"].mean(), counts["n"].quantile(.95, interpolation="nearest")
            rows.append({"word_k":word, "missing_k":missing, "mean_candidates":mean, "p95_candidates":p95,
                         "eligible":mean<=mean_limit and p95<=p95_limit,
                         "recall":found.height/truth.height, "misses":truth.height-found.height,
                         "oracle_f05":score(found,truth,roster)["macro_f05"]})
    table = pl.DataFrame(rows)
    parquet(run / "budget_grid.parquet", table)
    table.write_csv(run / "budget_grid.tsv", separator="\t")
    eligible = table.filter(pl.col("eligible"))
    if not eligible.height:
        raise ValueError("No candidate configuration meets the declared budget")
    at_target = eligible.filter(pl.col("recall") >= target_recall)
    if at_target.height:
        chosen = at_target.sort(["mean_candidates","oracle_f05"],descending=[False,True]).row(0,named=True)
        reason = "Smallest eligible candidate set reaching the Tune recall target."
    else:
        chosen = eligible.sort(["oracle_f05","recall","mean_candidates"],descending=[True,True,False]).row(0,named=True)
        reason = "Recall target not met. Best measured candidate oracle within declared budget; final score remains to be tested."
    result = {"selection_population":"tune", "queries":roster.len(), "true_links":truth.height,
              "target_recall":target_recall, "mean_limit":mean_limit, "p95_limit":p95_limit,
              "chosen":chosen, "reason":reason, "grid_sha256":sha256(run/"budget_grid.tsv")}
    contract(run / "budget.json", result)
    print(table, flush=True)
    print(reason, chosen, flush=True)
    return result
