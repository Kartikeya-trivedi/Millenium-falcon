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

## Sibling experiment and controls

The sibling model is now measured. It trains on 3,000 c_prob owners that were excluded from the upstream direct model and transliteration dictionary. It uses 82 feature columns, including 15 score/relationship columns, and stops at 146 boosting rounds. Its threshold is 0.81, selected on the separate selection panel.

| Model | Tune macro F0.5 | Select macro F0.5 | Development macro F0.5 |
| --- | ---: | ---: | ---: |
| Direct matcher | 0.956041 | 0.954140 | 0.951625 |
| Second-stage control: direct features plus upstream score | 0.956669 | 0.955433 | 0.952165 |
| Sibling support | 0.960309 | 0.963231 | 0.960921 |
| Sibling support with score-mined negatives | 0.962271 | 0.964921 | 0.960532 |
| Sibling support with separate missing-address threshold | 0.960385 | 0.963615 | 0.960860 |
| Sibling support with unowned-negative weights doubled | 0.959231 | 0.959354 | 0.958175 |

The relationship features add 0.008756 macro F0.5 over the retraining control on the same development owners. A paired bootstrap resampling whole owners gives a 95% interval of [0.005817, 0.012047]. This interval is conditional on the fitted models and decisions; it does not cover training, threshold-selection, label-noise or distribution-shift uncertainty. No fresh Audit was used.

Keep the sibling model as the next larger comparison. Neither the score-mined negatives nor the extra missing-address threshold earns default status from this pilot: both have weaker development macro score than the simpler sibling model. The score-mined variant improves Tune/Select and may merit a larger independent check, but is not an established improvement.

The predeclared unowned-negative weighting ablation also fails. The same 106,325 sampled training pairs are retained, with 17,770 unowned negatives receiving twice their original sampling weight. No owner-status field is a model input, positives are unchanged, and no pair prevalence is forced. Development macro F0.5 changes by -0.002746, with paired owner-bootstrap 95% interval [-0.005065, -0.000842]. Tune unowned false accepts decrease from 40 to 37, but total false accepts rise from 75 to 76 and rejected true candidates rise from 494 to 503. Keep the original weighting.

## Residual errors and one-hop probe

With sibling support, Tune has 48 retrieval misses, 494 rejected true candidates and 75 false accepts. There are 2,681,854 unowned records among all 10,320,219 training targets (25.99%). They make up 15.52% of retrieved negative pairs but 40/75 false accepts (53.33%). These are different denominators; this does not justify forcing an arbitrary 40% negative-pair training mix.

Of 322 true links with missing target addresses, 311 are retrieved and only 148 accepted. Of 496 true links with non-Latin names, 488 are retrieved and 454 accepted. The non-Latin flag covers any non-Latin script and should not be read as a Hindi-only statistic. Raw text and the existing Indic transliteration views remain intact.

A single full-target word search from up to two predicted seed records per owner adds 11,443 candidates across Tune, recovering eight missed links. Recall moves from 99.3065% to 99.4220%; mean candidates move from 237.617 to 243.339, p95 to 273. The candidate oracle improves only from 0.997584 to 0.997782. This is a retrieval probe, not a measured matcher improvement. New candidates need a separately trained matcher and are not automatically accepted.

The sibling feature implementation was optimized after this run. On a saved 29,298-pair shard, every feature value remained bit-for-bit equal, while runtime fell from 9.234 to 1.641 seconds (5.63x for that shard). This is not a full-training speed estimate.

The new streamed inference path was checked on 20 existing pilot owners: all 4,932 candidate pairs and direct-model probabilities matched exactly, with maximum probability difference zero. This checks implementation consistency, not model accuracy on a new population. The output writer passes 22 focused tests and the supplied validator on an exported fixture with ID checks. Full test inference and full-data output validation are still pending.

## Next measurement

Scale the changed direct model to the full training profile, then compare sibling support and its retraining control on the same evaluation owners. Keep full-world rival evidence and missing-owner stress as separate experiments. Do not infer a gain from the richer feature count alone, from candidate-oracle scores, or from comparing a 2,000-owner development subset with the teammate's older 20,000-owner result.
