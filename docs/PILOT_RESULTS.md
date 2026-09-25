# Rich direct-matcher pilot: 25 September 2026

These are measured local development results from the new pipeline. They are not an external result or a paired improvement over the supplied first-generation report. The old exposed Audit pool supplies the development subset; the fresh 50,000-owner Audit panel was not scored.

## Population and preprocessing

- 10,000 Fit owners for the direct model; 3,000 disjoint c_prob owners reserved for support-model training.
- 2,000 Tune owners, 2,000 threshold-selection owners and 2,000 exposed development owners.
- Retrieval uses all 10,320,219 training targets in their corresponding country/source partitions, including unowned targets.
- Raw multilingual fields are retained. The Fit-only 1,205-word transliteration dictionary and generic fallback are carried forward.
- Train/test preparation completed locally; only train-derived labels entered development. Test has no measured accuracy.

## Candidate budget on Tune

There are 6,921 true Tune links. K values below are per target source, before deduplicating the channel union. Means and p95 cover the complete 2,000-owner Tune roster.

| Word K | Missing-address K | Mean | p95 | Recall | Misses | Candidate-oracle macro F0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0 | 100.0 | 100 | 98.4684% | 106 | 0.995554 |
| 100 | 25 | 237.6 | 250 | 99.3065% | 48 | 0.997584 |
| 100 | 100 | 379.5 | 400 | 99.3498% | 45 | 0.997737 |
| 200 | 25 | 433.0 | 450 | 99.4943% | 35 | 0.998130 |
| 200 | 100 | 566.7 | 600 | 99.5376% | 32 | 0.998282 |

The stated target is at least 99.5% recall at mean at most 400 and p95 at most 500. No tested configuration meets all three. The automatic best eligible oracle is W100/M100. The actual scoring experiment explicitly uses **W100/M25**: another 141.9 candidates per owner recovered only three extra links on this panel. That choice is frozen in feature_contract.json rather than hidden in the automatic budget result.

## Direct model

The feature matrix has 67 columns: the 28 reference features plus distinctive-token coverage, typed address numbers, alias/domain comparisons, transliteration coverage, name ambiguity and channel provenance. Some inactive channel columns are constant. The pilot generated 4,516,827 scored candidate pairs across all 19,000 owners.

Training retains every retrieved positive plus up to eight hard and 24 deterministically sampled negative pairs per Fit owner. Inverse inclusion weights preserve the remaining negative mass. This gives 354,368 training pairs, including 34,368 positives; Tune evaluates all 475,234 candidate pairs without negative downsampling.

LightGBM uses learning rate 0.05, 63 leaves, maximum depth 10, minimum leaf size 100, L2 penalty 5, seed 42 and four CPU threads. Early stopping chooses iteration 823 from a cap of 6,000; Tune log loss is 0.00385653. The boosting fit took 60.2 seconds after data/index/features were ready. That time excludes the substantially longer preparation, retrieval and feature stages.

The acceptance threshold is 0.81, chosen on the separate selection panel. Ownership is resolved only among each evaluated panel's query owners.

| Panel | Macro F0.5 | Link precision | Link recall |
| --- | ---: | ---: | ---: |
| Tune | 0.956041 | 98.6072% | 91.0418% |
| Select | 0.954140 | 98.5911% | 90.8671% |
| Development | 0.951625 | 98.4221% | 90.7127% |

## What the error ledger says

Tune has 48 missed true candidates, 572 retrieved true candidates rejected by the decision rule and 89 false accepts. There were zero true links lost to ownership in this small panel; this does not establish a zero loss rate under a full claimant population.

Ordered macro-score loss: retrieval 0.00241603, true-candidate rejection 0.03189619, false acceptance 0.00964637 and panel ownership 0. These contributions sum to 1 minus the final Tune score. The largest measured gap remains rejection of true candidates, so the next bounded experiment adds sibling evidence on the same candidate set.

## Next measurement

Scale the changed direct model to the full training profile, then compare sibling support on the same evaluation owners. Keep full-world rival evidence and missing-owner stress as separate experiments. Do not infer a gain from the richer feature count alone, from candidate-oracle scores, or from comparing a 2,000-owner development subset with the teammate's older 20,000-owner result.
