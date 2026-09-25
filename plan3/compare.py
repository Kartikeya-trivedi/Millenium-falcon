"""Paired owner-level model comparisons, including clustered uncertainty."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import polars as pl

from .artifacts import ROOT, parquet, sha256, write_json


def paired_summary(before: pl.DataFrame, after: pl.DataFrame, repeats=2000, seed=42):
    for frame in (before, after):
        if frame["s1_id"].n_unique() != frame.height or not frame.height:
            raise ValueError("Per-owner scores must be unique and nonempty")
    if (before.height != after.height or
            before.select("s1_id").join(after.select("s1_id"), on="s1_id", how="anti").height):
        raise ValueError("Paired comparison requires identical owner populations")
    pairs = (before.select("s1_id", before_f="f3")
             .join(after.select("s1_id", after_f="f3"), on="s1_id")
             .sort("s1_id").with_columns(delta=pl.col("after_f")-pl.col("before_f")))
    delta = pairs["delta"].to_numpy()
    if not np.isfinite(delta).all() or repeats < 100:
        raise ValueError("Finite scores and at least 100 resamples are required")
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    # Resample whole S1 groups; multiple links for one owner are not independent.
    for start in range(0, repeats, 50):
        n = min(50, repeats-start)
        means[start:start+n] = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    report = {"owners": len(delta), "before_macro_f05": pairs["before_f"].mean(),
              "after_macro_f05": pairs["after_f"].mean(), "paired_delta": delta.mean(),
              "improved_owners": int((delta > 1e-12).sum()),
              "worsened_owners": int((delta < -1e-12).sum()),
              "unchanged_owners": int((np.abs(delta) <= 1e-12).sum()),
              "bootstrap_repeats": repeats, "seed": seed,
              "owner_bootstrap_95pct": [float(low), float(high)],
              "interpretation": "Conditional on these fitted models, fixed decisions and sampled owners. "
              "Does not include training, threshold-selection, label-noise or distribution-shift uncertainty."}
    return report, pairs


def compare(prepared: Path, run: Path, before: str, after: str):
    manifest = pl.read_parquet(prepared / "manifest.parquet", columns=["s1_id", "country"])
    out = run / "comparisons" / (before + "-vs-" + after)
    report = {}
    for sample in ("tune", "select", "development"):
        a = run / f"{before}_{sample}_per_owner.parquet"
        b = run / f"{after}_{sample}_per_owner.parquet"
        result, pairs = paired_summary(pl.read_parquet(a), pl.read_parquet(b))
        result["artifacts"] = {"before_sha256": sha256(a), "after_sha256": sha256(b)}
        pairs = pairs.join(manifest, on="s1_id")
        result["by_country"] = pairs.group_by("country").agg(
            owners=pl.len(), delta=pl.col("delta").mean(), before=pl.col("before_f").mean(),
            after=pl.col("after_f").mean()).sort("country").to_dicts()
        parquet(out / (sample + "_per_owner.parquet"), pairs)
        report[sample] = result
        print(f"{before} -> {after}/{sample}: delta={result['paired_delta']:.6f}, "
              f"owner bootstrap 95%={result['owner_bootstrap_95pct']}", flush=True)
    write_json(out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--before", default="direct")
    p.add_argument("--after", default="support")
    args = p.parse_args()
    if any(not value.replace("_", "").isalnum() for value in (args.before, args.after)):
        p.error("Model names must contain only letters, digits and underscores")
    compare(args.prepared, args.run, args.before, args.after)


if __name__ == "__main__":
    main()
