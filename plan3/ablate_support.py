"""Control for second-stage retraining before attributing gains to sibling evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ROOT, read_json
from .features import FEATURES
from .train import fit_model, predict_model, evaluate_model


def ablate(prepared: Path, run: Path, threads=4, max_rounds=6000):
    spec = read_json(run / "support_features_contract.json")
    direct = read_json(run / "direct_meta.json")
    if spec["parent_model_sha256"] != direct["model_sha256"]:
        raise ValueError("Support features refer to a different upstream model")
    if spec["support_training_pool"] != "c_prob" or direct["sample"] != "fit":
        raise ValueError("Disjoint upstream scoring provenance is required")
    # Same training owners, candidates, sampling, Tune and boosting settings as M1.
    # Only the extra relationship columns are omitted.
    features = FEATURES + ["p1"]
    name = "support_control"
    model = fit_model(prepared, run, features, run / "support_features", name,
                      "support", threads, max_rounds)
    predict_model(run, model, run / "support_features", run / (name + "_scores"), features, threads)
    return evaluate_model(prepared, run, run / (name + "_scores"), name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=ROOT / "work/plan3/prepared")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=6000)
    args = p.parse_args()
    ablate(args.prepared, args.run, args.threads, args.max_rounds)


if __name__ == "__main__":
    main()
