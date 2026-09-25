# Millenium Falcon

Business record matching across three sources. A reference record may match zero, one or many records in the other two sources. Each target may have at most one accepted owner.

## Current build

Start with [the server commands and handoff guide](docs/SERVER_HANDOFF.md). The next run trains the new direct matcher at a larger scale; it does not repeat first-generation model training.

The imported reference source is in plan1/. The new implementation in plan3/ includes deterministic data preparation, Indic transliteration, full-corpus sparse retrieval, candidate-budget selection, 67-column direct features, LightGBM training, threshold selection and four-class error accounting. A second model adds sibling and cross-source evidence. The terminal runner records stage logs and resumes compatible artifacts. The plan3-cloud branch adds persistent Modal CPU execution through the optional compute dependency.

The direct-model pilot scores 0.951625 macro F0.5 on a 2,000-owner exposed development subset; sibling support scores 0.960921. A control trained on the same second-stage owners without relationship features scores 0.952165. Tune candidate recall is 99.3065% at mean 237.6 candidates per owner. These are local development measurements, not an external result or a fresh-Audit evaluation. See [the measured pilot results](docs/PILOT_RESULTS.md).

The reference report records an external F0.5 score of 0.9388. That is supplied reference evidence, not a result reproduced by this repository. The development target is at least 99.5% candidate recall with mean at most 400 and p95 at most 500 candidates per query. Final selection uses macro F0.5, including empty answers.

## Setup

Use uv with the committed Python 3.12 pin and dependency lockfile.

    uv python install 3.12
    uv sync --locked
    uv run --locked python -m pytest plan3/tests -q

Place the supplied TSV data under dataset/train/ and dataset/test/. Source columns are entity_id, business_name, business_address, country. Labels use source1_entity_id, matched_entity_ids. Inputs are tab-separated, with ID lists separated by commas.

    uv run --locked python -m plan3.server --dataset /path/to/dataset --work-root /path/to/run-storage/p3-v1 --profile full --threads 16 --check-only
    uv run --locked python -m plan3.server --dataset /path/to/dataset --work-root /path/to/run-storage/p3-v1 --profile full --threads 16 --stop-after train

Use Python 3.12 and the pinned dependencies. Training requires the four train TSVs only. Raw data and generated artifacts are transferred separately from Git. No GPU is required. Full-profile memory planning and resume/return commands are in the handoff guide.

Indexes use all relevant target records, including unowned records. The pilot limits query owners, not the target corpus. The full retrieval profile uses a larger query population; it is not automatically a final validated model.

The preparation and cache contracts reject changed inputs. Choose new artifact directories when changing preprocessing or index definitions. Preserve complete candidate groups for every reference record when computing relative features.

## Text and matching evidence

Raw text is preserved. Fit-only dictionary transliteration and a generic fallback provide an additional Latin view of Indic text. Number handling distinguishes values such as 12B, 12/3 and A-5.

The richer direct scorer retains the existing 28 pair features and adds distinctive-token coverage, address-number roles, alias/domain comparisons, transliteration coverage and retrieval-channel provenance. Sibling evidence uses predictions generated independently of the corresponding training labels. Learned full-population competing-owner evidence remains a later experiment.

## Evaluation discipline

Tune selects model and candidate settings. A separate selection population chooses decision thresholds. The previously unused probability pool is explicitly assigned to second-stage training in this implementation; it must not also calibrate that model. A fresh 50,000-owner Audit panel is reserved outside the previously exposed panel.

See [the execution plan](docs/PLAN.md) for the staged experiments, error accounting and acceptance criteria. Baseline source provenance is recorded in [UPSTREAM.md](plan3/UPSTREAM.md).
