# Millenium Falcon

Business record matching across three sources. A reference record may match zero, one or many records in the other two sources. Each target may have at most one accepted owner.

## Current build

The imported first-generation source is in plan1/. The new implementation in plan3/ includes deterministic data preparation, Indic transliteration, full-corpus sparse retrieval, candidate-budget selection, richer direct features and regression tests. Model training and score-aware relationship features are the next implementation steps. No new performance result is claimed by this initial code release.

The reference report records an external F0.5 score of 0.9388. That is supplied reference evidence, not a result reproduced by this repository. The development target is at least 99.5% candidate recall with mean at most 400 and p95 at most 500 candidates per query. Final selection uses macro F0.5, including empty answers.

## Setup

Python 3.12 is used by the current implementation.

    python -m pip install -r plan3/requirements.txt
    python -m pytest plan3/tests -q

Place the supplied TSV data under dataset/train/ and dataset/test/. Source columns are entity_id, business_name, business_address, country. Labels use source1_entity_id, matched_entity_ids. Inputs are tab-separated, with ID lists separated by commas.

    python -m plan3 prepare --dataset dataset --include-test
    python -m plan3 views
    python -m plan3 retrieve --run work/plan3/pilot --profile pilot
    python -m plan3 budget --run work/plan3/pilot

Indexes use all relevant target records, including unowned records. The pilot limits query owners, not the target corpus. The full retrieval profile uses a larger query population; it is not automatically a final validated model.

The preparation and cache contracts reject changed inputs. Choose new artifact directories when changing preprocessing or index definitions. Preserve complete candidate groups for every reference record when computing relative features.

## Text and matching evidence

Raw text is preserved. Fit-only dictionary transliteration and a generic fallback provide an additional Latin view of Indic text. Number handling distinguishes values such as 12B, 12/3 and A-5.

The richer direct scorer retains the existing 28 pair features and adds distinctive-token coverage, address-number roles, alias/domain comparisons, transliteration coverage and retrieval-channel provenance. Further work adds sibling and competing-owner evidence using predictions generated independently of the corresponding training labels.

## Evaluation discipline

Tune selects model and candidate settings. A separate selection population chooses decision thresholds. The previously unused probability pool is explicitly assigned to second-stage training in this implementation; it must not also calibrate that model. A fresh 50,000-owner Audit panel is reserved outside the previously exposed panel.

See [the execution plan](docs/PLAN.md) for the staged experiments, error accounting and acceptance criteria. Baseline source provenance is recorded in [UPSTREAM.md](plan3/UPSTREAM.md).
