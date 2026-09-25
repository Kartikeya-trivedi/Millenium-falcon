# Plan 3 — The strongest justified solution

**Execution update:** [Plan 3 final](../docs/PLAN.md) incorporates the teammate's completed 0.9388 Plan 1 result and the subsequent review. Reuse that baseline without local reproduction. Prioritize a complete error ledger and existing-candidate sibling evidence ahead of broad retrieval expansion. This document remains the detailed reference for a later one-hop experiment.

**Objective:** build the strongest solution supported by the supplied data, while removing components that add complexity without a dependable score gain.

**Resource assumption:** compute is plentiful and there is no deadline constraint. The limiting factors are evidence, error propagation and the number of interacting mechanisms we can validate.

**Position:** run concurrently with [Final Plan 1](../plans/PLAN_1_FINAL.md) and [Final Plan 2](../plans/PLAN_2_FINAL.md). Build and measure a strong direct-tree baseline inside this run, then add the components below where its residual errors support them. Plan 3 does not wait for another plan's checkpoint. Its final implementation may retain fewer components than this experiment plan proposes.

**Shared comparison, separate artifacts:** use the immutable split and shared Audit comparison panel in Final Plan 1, Section 2. Keep this run's artifacts under work/plan3/RUN_ID/ and output/plan3/RUN_ID/. The [earlier direct-tree blueprint](../docs/DIRECT_MATCHER_REFERENCE.md) remains the technical reference for this baseline. Final Plan 2 is now a separate neural branch; its encoder is not automatically part of Plan 3.

**No score promise:** the EDA does not establish a final achievable F0.5, French accuracy or a guaranteed winning architecture. This plan is intended to maximize our chance of winning with defensible evidence.

## 1. My recommended architecture

Build a strong LightGBM direct matcher using the earlier direct-tree blueprint within this run. Add a **single, controlled relationship pass**:

    This run's direct-matcher candidate set C0
        → direct matcher M0
        → very reliable seed matches
        → one-hop retrieval of related S2/S3 records
        → expanded candidate set Cfinal
        → direct features + independently generated support features
        → final matcher M1
        → calibrated set decisions and competing-owner resolution
        → exact candidate ledger and final matches

Then inspect what is still wrong. Allocate one optional specialist slot to the best-supported remaining problem: an additional retrieval model, a different pair model, or a compact neural pair matcher. Do not add all three as a standard recipe.

The distinctive idea is **using another record of a business as additional evidence without treating that record’s prediction as truth**. S1 remains the reference. Every final S1-to-target link is scored and may be rejected.

## 2. Measure where the next score gain can come from

Before building a relationship index or downloading weights, produce an error ledger for this run's frozen direct-matcher checkpoint.

For every held-out business, record true count, retrieved true count, predicted count, TP/FP/FN, macro-score contribution and whether an ownership decision changed the result. Assign failures to overlapping diagnostic tags, then to one primary cause for accounting:

| Primary cause | Next intervention |
| --- | --- |
| True record absent from C0 | Better retrieval or one-hop expansion |
| True record in C0 but rejected | Better features, support evidence or a scoring specialist |
| Wrong record accepted | Hard negatives, calibration, conflict evidence or better pair discrimination |
| Target assigned to the wrong S1 | Competing-owner retrieval/evidence and decision policy |
| Singleton receives any record | High-confidence false-match analysis and set-level rejection |
| Observable fields almost indistinguishable despite different labels | Quantify the ambiguous tail; avoid memorizing IDs or inventing extra data |

Compute the **oracle macro F0.5 for C0** as well as ordinary link recall. If the oracle is 0.997, perfect extra retrieval can add at most 0.003 to the ideal score for that population. A large retrieval project then needs an unusually convincing justification.

That bounds the pure recall-ceiling gain. New candidates could also expose a competing owner or provide better context and prevent false positives; measure those decision gains separately.

Likewise, do not multiply the notebook’s “94% of its candidate oracle” by the new oracle to forecast a final score. It came from a different, weaker candidate population and optimistically selected thresholds.

Rank proposed work by plausible recoverable **macro-score loss**, affected-business count, false-positive risk and maintenance cost. A large error count on businesses that already have many correct links may matter less than fewer singleton false merges.

## 3. Add one-hop record relationships safely

The [consolidated findings](https://github.com/sohambuilds/icmliclrneuripsaaaiemlnlpmaxxing/blob/main/EDA_FINDINGS_ALL.md) report that 75.9% of sampled hard-name true links had an easier same-business record with a strong name or address resemblance. That calculation used the labels to know the owner. It identifies an opportunity, not a deployable 75.9% rescue rate.

The same findings show that same-source records sharing a cleaned name and address have the same owner only about 97–98% of the time. That is too weak for unconditional propagation under F0.5.

### 3.1 Build a label-free neighbor index

Construct candidate neighbor lists among S2/S3 records within the same country:

1. Exact conservative normalized name-plus-address keys as a cheap starting channel.
2. Bounded character/word similarity search for near duplicates, emphasizing distinctive address tokens and retaining name/number evidence.
3. Same-source neighbors first; add cross-source neighbors as a separate ablation.

Store raw and normalized similarities, shared distinctive tokens, numeric conflicts, field completeness, source and duplicate-group size. Labels can evaluate these lists and train the final matcher; they must not determine which records are neighbors at inference.

Do not cluster a whole connected component and call it one business. A single mistaken edge could merge many unrelated businesses. Common short addresses and common-name groups need strict degree limits.

Start with at most 20 neighbors per seed record. Assess neighbor recall and false-neighbor rates on held-out owners. Larger degrees are experiments, not an assumption that more context is better.

### 3.2 Select seeds from direct predictions

Use M0’s direct scores with thresholds validated on held-out development predictions, without any ground-truth lookup:

- Require a very high direct-match score.
- Require a clear competing-owner margin.
- Prefer distinctive name/address evidence and reliable number agreement when available.
- Use at most two seeds from each source for one S1.

Select the seed threshold on held-out development predictions. A starting target is a **lower confidence bound on seed precision of at least 99.5%**, with slice counts reported. This is a measured target, not a guarantee that every seed is correct.

Track seed coverage as well as precision. If a conservative threshold produces almost no seeds for Indic or missing-address cases, it may not address the intended failures.

### 3.3 Expand once, and keep candidate accounting exact

For each S1, retrieve neighbors of its selected seeds and propose new S1-to-target pairs. Initial ceiling: **40 new candidates per source**, beyond C0.

    Cfinal = C0 ∪ proposed neighbor targets

Retain all C0 candidates. Do not accept a neighbor merely because it was proposed. Compute direct S1-to-target features and an M0 score for every new pair, then score all Cfinal pairs with M1. Give expansion-only pairs explicit missing-rank/origin flags rather than inventing original retrieval ranks.

If a later retrieval specialist is retained, its proposals also enter Cfinal and receive the same required scoring. They do not become additional propagation seeds.

No new accepted neighbor becomes another seed in this run. There is exactly one expansion hop. If no seed exists, the S1 retains its original direct-matcher candidates.

Persist proposal provenance, including seed IDs and neighbor-channel evidence, for debugging. IDs are bookkeeping, not model features.

### 3.4 Use support as features

For a candidate record x and an S1 business, add:

| Feature | Meaning |
| --- | --- |
| Best support score | Strongest similarity between x and a reliable seed for that S1 |
| Support provenance | Same-source versus cross-source support; exact versus fuzzy neighbor |
| Direct/support disagreement | Strong neighbor evidence but weak S1-to-x evidence, or the reverse |
| Number/address consistency | Whether x, the seed and S1 agree on reliable details |
| Distinct support groups | Capped count of different evidence groups, not raw duplicate votes |
| Opposing support | Whether another S1 has a stronger seed supporting x |
| Missingness/coverage | Whether the support fills a field absent or uninformative in x |

Exclude x from its own supporting-seed set. Count identical normalized records as one evidence group so repeated copies do not manufacture independent confidence. Do not multiply seed probabilities as though their errors were independent.

### 3.5 Train one final matcher

Use a second compact LightGBM model M1 on the direct features, M0 score and support features. Its task is still binary S1-to-target matching.

Include expansion candidates that are wrong, especially:

- Exact-looking duplicates owned by different S1 businesses.
- Distractors near a mistaken seed.
- Common-address neighbors with unrelated names.
- Same-name branches with conflicting numbers.
- Newly proposed records also strongly supported by another S1.

Tune M1 to preserve strong direct matches while using support to rescue genuinely harder records. Missing support is a normal feature value; unsupported candidates remain eligible.

**Two separate ablations are mandatory:**

1. Expansion with the direct matcher only: how much does merely finding extra candidates help?
2. Support features with the same Cfinal: how much does learning from relationships help?

If expansion helps but M1 does not, ship expansion plus the simpler matcher. If support helps only on existing C0 candidates, avoid the extra neighbor-retrieval stage where possible.

## 4. Prevent leakage through the extra stage

This is the main implementation risk in Plan 3.

Never train M1 on seeds or M0 scores generated in sample. The model could otherwise learn from unrealistically clean, nearly memorized first-stage predictions.

Use grouped cross-fitting:

1. Split Fit owners into three inner groups.
2. Fit M0 and its supervised dictionary on two groups.
3. Score the held-out group, select seeds, build its expanded candidates and compute support features.
4. Repeat until every Fit owner has out-of-fold first-stage outputs.
5. Train M1 on those out-of-fold features and the actual labels.

Generate support and competing claims jointly for the held-out world so every participating query is unseen by that first-stage fit. Do not mix training-owner predictions into a validation world as “reliable” rivals.

For an honest development-fold score of the whole two-stage pipeline, the outer held-out owners must also be excluded from **all inner fitting and stage-two fitting**. Inner cross-fitting creates M1 training data; outer held-out inference evaluates the complete pipeline. A single set of out-of-fold M0 predictions does not automatically make an in-sample M1 score valid.

At final fitting, train M0 on the chosen Fit population, retain M1 trained from out-of-fold inputs, and calibrate the resulting complete pipeline on C-prob. Select seed/acceptance/ambiguity settings on the designated development/selection populations. Keep the prospectively locked Audit set for the final frozen configuration.

Use raw M0 scores and empirically validated seed thresholds for this extra stage. Reserve C-prob for calibrating the final output; do not first use its labels to calibrate M0 and then treat the same rows as independent final-calibration examples. If calibrated upstream scores are needed, their calibration must also be fitted on separate or cross-fitted data.

Fold-specific dictionaries change normalized target text and neighbor indexes. Cache them under separate hashes. A globally learned Indic dictionary cannot be reused for every validation fold.

## 5. Spend the specialist budget on the remaining bottleneck

Run this section only after evaluating the relationship pass. First compare a lightweight pilot on the residual slices **and on a representative complete query population**. Improvement on a hand-picked hard subset is not sufficient.

### Option A: complementary sparse/dense retrieval

Choose this if meaningful macro loss still comes from records absent from Cfinal.

First try the smallest training-derived repair: better ambiguous-token alignments, a bounded alternate transliteration view, or missing distinctive words excluded by the sparse vocabulary. These can be more effective than a new encoder.

If the residual remains material, test a single multilingual encoder on **name plus address**. A concrete starting candidate is [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md): its model card lists MIT licensing, 12 layers and 384-dimensional embeddings. Verify the exact downloaded weight revision, license and parameter count before use; the task requires at most 8B parameters.

Use one fixed serialization of the raw and normalized name/address fields. Apply the model card’s prefix and pooling conventions; for symmetric record similarity, start with the query prefix for both sides and normalized mean-pooled vectors.

Two eligible deployments:

- Search all relevant same-country targets for queries identified as difficult by observable features.
- Build a target subset using observable problems such as poor Indic dictionary coverage, opaque short names or heavily incomplete addresses, and search that subset for all S1 queries.

Do not identify eligible records using validation labels at inference. Do not route only businesses with no existing match: a confident business can still be missing one difficult extra record.

Use a country-partitioned approximate nearest-neighbor index, with exact-neighbor checks on a diagnostic subset. Report encoder retrieval loss separately from approximate-index loss. Compare against sparse retrieval using the **same target population, query population and final candidate budget**.

For scale, ten million 384-dimensional float16 vectors alone occupy about 7.68 GB; index structures and metadata are additional. This is a storage calculation, not a measured throughput or RAM claim.

Keep the channel only if it adds unique true candidates and improves final macro F0.5 after rescoring. The existing small-pool embedding experiment cannot establish this benefit.

### Option B: a compact neural pair matcher

Choose this if the correct records are already present but the tree model consistently mishandles a material, explainable class of pairs.

Use a permissively licensed multilingual encoder with a binary classification head, fine-tuned only on provided training pairs. The encoder from Option A is a concrete starting point; this is a **new supervised matching experiment**, not a claim that its retrieval checkpoint already resolves businesses.

Feed both businesses’ names and addresses with explicit field separators. Preserve raw spelling plus a bounded normalized view, including numbers. Start with a 256-token limit, verify truncation rates, and increase only when important content is lost.

A reasonable first training run uses two epochs, learning rate around 2e-5, mixed precision and early stopping on held-out loss and final score. Treat these as pilot settings. Use true retrieved hard negatives, same-name branches and convincing singleton negatives.

Initially score only a frozen, observable disagreement band: uncertain direct scores, inconsistent name/address evidence, competing owners or conflicting relationship support. Verify the band’s coverage of real errors, including overconfident errors.

Add the neural score and a “specialist ran” flag to the final scorer; do not let the neural model directly override the output. Produce its training features out of fold, and recalibrate the complete system afterwards.

It earns a place only through additional macro-score gain at the same candidate set. Extra GPU availability by itself is not a reason to include it.

### Option C: modest tree-model diversity

Choose this when a cheaper second tree model provides complementary mistakes.

Train one [Apache-2.0-licensed CatBoost](https://github.com/catboost/catboost/blob/master/LICENSE) alternative on the same direct pair population. Compare the best single model with a two-model blend using a small grid of blend weights.

Require out-of-fold evidence that the combination corrects errors beyond what the better individual model fixes. Calibrate the selected blend on separate held-out predictions. If one model dominates, retain it alone.

Do not average extra models merely to smooth their outputs. Correlated models can preserve the same confident false matches.

### Specialist selection rule

Default to one retained specialist family. A second family needs a separate ablation showing a material gain **on top of the already selected system**, not against the older Plan 1 baseline. Record its retraining, inference and packaging burden.

## 6. Finish with conservative set and ownership decisions

Evaluate the earlier direct-tree blueprint's decision-policy family within this run, applied to its final calibrated scores:

- Several true records per source remain allowed.
- Empty answers remain valid.
- One target record can belong to at most one S1.
- Close competing claims may be left unresolved when validation supports abstention.
- A different legal form or uncertain first number is evidence, not an automatic veto.

Tune the decision policy after the final candidate set, scorer and calibration are fixed. Reusing thresholds from a different candidate or scoring pipeline is not a valid ablation.

A **separate singleton gate** is one final optional experiment if no-match false merges remain a large share of score loss. Train it on out-of-fold entity summaries such as best final score, competing-owner margin, distinctive evidence, support consistency and candidate completeness. Compare it with the existing start/link thresholds and measure the damage to genuine one-match businesses.

Do not build a general graph neural network, a large integer-programming assignment, or an expected-match-count system by default. They are unnecessary to express the observed one-owner constraint and introduce additional assumptions.

Do not force a match because another source has one, and do not propagate an owner repeatedly until a graph becomes consistent. Repeated propagation can make a wrong seed appear increasingly well supported.

## 7. Validate robustness and perform the final removal pass

Preserve the earlier direct-tree blueprint's three distinct evaluation views:

1. Full-target-pool candidate recall, oracle score and direct matching.
2. Controlled complete ownership worlds with measured approximately 26%, 40% and 50% distractors.
3. Missing-owner worlds, where targets remain but some possible S1 claimants are hidden.

Use business-grouped validation, name-family holdouts and US-to-India/India-to-US tests. For every learned dictionary, seed generator, specialist and calibrator, record precisely which labels it could access.

For France, check deterministic parsing, field/truncation coverage, candidate counts, competing-claim rates and stability under the observed formatting variations. Treat these as unlabeled diagnostics. Do not report pseudo-label agreement as French accuracy.

### Final ablation matrix

| Remove from the selected final system | What it tests |
| --- | --- |
| Learned Indic dictionary | Whether a specialist made it redundant, or it remains essential |
| Each retrieval channel separately | Unique recovery at the chosen candidate cap |
| Typed number/distinctive-address features | Protection against same-name and same-address impostors |
| Hard-negative enrichment | Whether it improves difficult discrimination without miscalibration |
| One-hop expansion only | Value of additional candidates |
| Support features only, holding candidates fixed | Value of relational evidence |
| Cross-source support | Whether it helps beyond same-source neighbors |
| Optional specialist/blend | Its marginal contribution to the final system |
| Ambiguity abstention/ownership policy | Effect on macro score, singletons and contested records |
| Optional singleton gate | Benefit beyond ordinary set thresholds |

Retrain affected downstream stages. An input group removed from a model trained to depend on it is not a fair removal ablation.

Use paired confidence intervals on the same held-out query outcomes and collision-group resampling for shared-record decisions. Confirm important gains across more than one development fold/world. Keep the locked audit out of the removal loop.

Suggested predeclared retention floors:

- Small deterministic addition: +0.0005 macro F0.5 with a positive paired confidence interval, or a clear operational improvement at equivalent accuracy.
- New learned stage/index: +0.0015 macro F0.5, positive paired confidence interval, and replicated benefit.
- Several interacting additions: demonstrate the joint gain and the contribution of each, or simplify the group.

These are complexity-control choices, not mathematical guarantees. Do not retain a component whose confidence interval includes no benefit simply because considerable work has already gone into it.

Apply those score gates to optional modeling and retrieval machinery. Output correctness, reproducibility and open-country handling are required capabilities, and French-format support remains an explicitly unvalidated generalization assumption rather than a claimed French score gain.

**Stopping rule:** stop adding mechanisms when the remaining recoverable score loss is small, two well-targeted alternatives fail the retention rule, or gains disappear under hard-negative/ownership stress. More compute should buy confidence and larger representative experiments before it buys more architecture.

## 8. Implementation and release sequence

| Stage | Deliverable | Decision |
| --- | --- | --- |
| 1 | This run's frozen direct-matcher checkpoint and per-business error ledger | Identify the largest plausible macro-score gains |
| 2 | Label-free neighbor lists and held-out seed precision/coverage | Continue only if useful neighbors are discoverable without owner labels |
| 3 | One-hop expanded candidates scored by the direct matcher | Decide whether expansion itself helps |
| 4 | Cross-fitted support features and final matcher M1 | Decide whether the extra learned stage helps |
| 5 | One residual specialist pilot selected from the ledger | Keep only a complementary gain |
| 6 | Final calibration, thresholds and ownership policy | Select on the designated held-out selection population |
| 7 | Final removal ablations and robustness report | Delete redundant or harmful components |
| 8 | Freeze artifacts; run the prospectively locked audit | Report all limitations without retuning on the audit |
| 9 | Full test inference, verification and package reproduction | Deliver both exact TSVs and the reproducible system |

Parallelize independent data partitions and the selected experiments where resources permit. Within one experiment, respect the dependency order: fold dictionary → normalization → indexes → candidates → model outputs → support → final model → calibration → decisions.

The pipeline needs a small, explicit experiment manifest, not an orchestration platform. Record inputs, split IDs, seeds, model revisions, feature/candidate versions and artifact hashes. Reuse an artifact only when all upstream fingerprints match.

Produce a compact evidence table for every retained component: additional TP/FP/FN, change in macro F0.5, confidence interval, singleton effect, country/slice effects, candidate growth, fit/inference time and storage.

## 9. Exact output accounting and reproducibility

The [evaluation specification](../student_resource/README.md) requires candidate_pairs.tsv to contain the exact candidate population presented to the matching model, not an early or selectively cleaned-up approximation.

For this architecture:

1. C0 contains the initial candidate population scored before expansion.
2. Neighbor and optional retrieval-specialist proposals add records but never remove C0 from the final ledger.
3. Every Cfinal record receives final matcher inference.
4. candidate_pairs.tsv serializes Cfinal.
5. matching_results.tsv contains only accepted records from Cfinal.

If another learned filtering stage is introduced, retain a complete scored-pair audit and ensure the final export still follows the official definition. Do not hide rejected model-scored candidates to make blocking look better.

Resolve ownership across all test S1 businesses, then serialize one row per S1 in both files, including France and empty lists. Verify every target ID, all country equalities, no duplicate IDs, candidate-subset membership and one final owner per target.

Run the official validator with --check-ids and enforce candidate-subset membership as an internal hard failure; the official checker only warns about that discrepancy.

Package runnable source, pinned dependencies, fitted model/dictionary/calibration artifacts or deterministic regeneration steps, model licenses and parameter counts, configuration, both TSVs and the filled methodology template. Reproduce the result in a clean environment with no network-dependent business lookup or hosted conversion service.

An all-label refit is optional. First validate that the larger training regime helps under cross-validation; regenerate calibration from held-out/out-of-fold predictions of that regime. Do not keep thresholds from the old smaller fit by assumption, and do not call the previous audit an independent score of a model subsequently trained on those audit labels.

## 10. What “reasonably complete” means here

The final solution should address reliable normalization, Indic spelling, high-recall retrieval, difficult negatives, unfamiliar-country behavior, multiple matches, singletons, competing owners and the exploitable part of duplicate relationships.

It does not need every possible embedding model, repeated pseudo-labeling, unrestricted graph closure, internet enrichment, a 7B generative model, or handcrafted exceptions for individual business IDs.

My preferred endpoint is **a strong direct-tree baseline plus a validated one-hop relationship pass, with at most one clearly useful specialist initially**. If that pass or specialist does not earn its place, the stronger engineering decision is to ship the simpler validated configuration. The winning criterion is the measured final set output, not the number of techniques in the architecture.
