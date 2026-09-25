"""Bounded two-threshold experiment for address-present and address-missing targets."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import polars as pl

from plan1.decide import one_owner
from .artifacts import ROOT, parquet, read_json, sha256, write_json
from .decisions import error_ledger


def select_thresholds(scored, truth, roster, reference_threshold, grid=None):
    if roster.len() == 0 or roster.n_unique() != roster.len():
        raise ValueError("Owner roster must be unique and nonempty")
    lookup = pl.DataFrame({"s1_id": roster}).with_row_index("_q")
    joined = (scored.join(lookup, on="s1_id")
              .join(truth.select("s1_id", "target_id").unique().with_columns(_true=pl.lit(1)),
                    on=["s1_id", "target_id"], how="left"))
    if joined.height != scored.height:
        raise ValueError("Scored owner outside policy population")
    if grid is None:
        grid = np.unique(np.r_[np.round(np.arange(.05, 1, .01), 8), reference_threshold, 1.000001])
    g = (lookup.join(truth.group_by("s1_id").len(name="g"), on="s1_id", how="left")
         .sort("_q")["g"].fill_null(0).to_numpy())
    idx, prob = joined["_q"].to_numpy(), joined["p"].to_numpy()
    actual = joined["_true"].fill_null(0).to_numpy()
    missing = joined["cand_addr_missing"].to_numpy() == 1
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Invalid model probability")
    counts, correct = [], []
    for field in (False, True):
        k, t = [], []
        for threshold in grid:
            mask = (missing == field) & (prob >= threshold)
            k.append(np.bincount(idx[mask], minlength=roster.len()))
            t.append(np.bincount(idx[mask], weights=actual[mask], minlength=roster.len()))
        counts.append(np.asarray(k)); correct.append(np.asarray(t))
    rows = []
    for i, a in enumerate(grid):
        total = counts[0][i] + counts[1]
        tp = correct[0][i] + correct[1]
        denominator = .25*g + total
        f = np.ones_like(denominator, dtype=float)
        np.divide(1.25*tp, denominator, out=f, where=denominator > 0)
        for j, b in enumerate(grid):
            rows.append({"address_threshold": float(a), "missing_threshold": float(b),
                         "macro_f05": float(f[j].mean()), "one_threshold": bool(a == b)})
    table = pl.DataFrame(rows)
    best = table.sort(["macro_f05", "one_threshold", "address_threshold", "missing_threshold"],
                      descending=[True, True, True, True]).row(0, named=True)
    return best, table


def apply_policy(scored, decision):
    threshold = pl.when(pl.col("cand_addr_missing") == 1).then(
        pl.lit(decision["missing_threshold"])).otherwise(pl.lit(decision["address_threshold"]))
    return scored.filter(pl.col("p") >= threshold)


def run_policy(prepared: Path, run: Path, model="support"):
    samples = pl.read_parquet(run / "samples.parquet")
    truth = pl.read_parquet(prepared / "truth.parquet").join(samples.select("s1_id"), on="s1_id", how="semi")
    scores = (pl.scan_parquet(str(run / (model + "_scores") / "*.parquet"))
              .select("s1_id", "target_id", "p", "sample", "cand_addr_missing").collect())
    reference = read_json(run / (model + "_decision.json"))["threshold"]
    choose = scores.filter(pl.col("sample") == "select")
    roster = samples.filter(pl.col("sample") == "select")["s1_id"]
    best, table = select_thresholds(choose, truth, roster, reference)
    name = model + "_missing_policy"
    parquet(run / (name + "_sweep.parquet"), table)
    write_json(run / (name + "_decision.json"), {**best, "selected_on": "select",
        "model": model, "model_sha256": sha256(run / (model + ".txt")), "source_sha256": sha256(Path(__file__)),
        "status": "experimental; compare Tune and held-out development before promotion"})
    report = {}
    for sample in ("tune", "select", "development"):
        p = scores.filter(pl.col("sample") == sample)
        roster = samples.filter(pl.col("sample") == sample)["s1_id"]
        accepted = apply_policy(p, best)
        final = one_owner(accepted)
        result, errors, per = error_ledger(p, p, accepted, final, truth, roster)
        parquet(run / f"{name}_{sample}_errors.parquet", errors)
        parquet(run / f"{name}_{sample}_per_owner.parquet", per)
        report[sample] = result
        print(f"{name}/{sample}: {result['after_ownership']}", flush=True)
    write_json(run / (name + "_report.json"), report)
    print(f"Selected thresholds: {best}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", default="support")
    args = p.parse_args()
    if not args.model.replace("_", "").isalnum():
        p.error("Invalid model name")
    run_policy(args.prepared, args.run, args.model)


if __name__ == "__main__":
    main()
