# Plan 2 — The Pareto solution

**Superseded as the Plan 2 execution plan:** use [Final Plan 2 — Direct multilingual matching](../plans/PLAN_2_FINAL.md). This earlier direct-tree design is retained as a technical reference for Plan 3's baseline; it is not the current Plan 2 branch.

**Recommendation:** this is the default destination. It addresses the main observed failure modes while keeping one direct pair matcher and a small, understandable decision layer.

**Resource assumption:** ample compute; no deadline-driven shortcuts. Complexity must still earn its place through measurable accuracy or operational value.

**Concurrent execution:** start this run independently of Plans 1 and 3. Use the immutable split and shared Audit comparison panel in [Final Plan 1, Section 2](../plans/PLAN_1_FINAL.md#2-data-and-the-shared-evaluation-contract), but fit this plan's own normalization, indexes, models and policies. Keep artifacts under work/plan2/RUN_ID/ and output/plan2/RUN_ID/. Compare frozen results when available; no other plan's completion is a prerequisite.

**Status:** execution plan. “Pareto” is the design objective, not a claim that this configuration has already been proved optimal.

| Level | What it delivers |
| --- | --- |
| [Plan 1](../plans/PLAN_1_FINAL.md) | One word retriever, 12 pair features, one matcher and threshold; no transliteration or dictionary |
| **Plan 2: this document** | Better candidate coverage and numeric evidence, representative hard negatives, calibrated set decisions and ambiguity handling |
| [Plan 3](../docs/RELATIONSHIP_REFERENCE.md) | One-hop record relationships and carefully selected residual-error improvements |

Share raw data and the split, metric and export contracts. Build this plan's richer pipeline within its own run; reuse compatible code where available, without requiring Plan 1 artifacts.

## 1. The intended final system

    Records + Fit-only normalization artifacts
        → combined word/character search
        → bounded missing-address and special-name retrieval
        → fused candidate ledger
        → richer direct pair features
        → one LightGBM classifier
        → held-out calibration
        → multiple-match thresholds + competing-owner checks
        → two verified TSV outputs

The four main improvements over Plan 1 are:

1. Recover records that combined-text search misses, particularly Indic names with incomplete addresses.
2. Separate true matches from same-name or same-address impostors using more precise evidence.
3. Train and calibrate on the difficult candidates the deployed search actually produces.
4. Make decisions using competing claims and the real macro F0.5 objective.

Do not add duplicate propagation or a neural model by default. Their upside has not yet been measured with a real retrieval-and-decision pipeline.

## 2. Establish a trustworthy comparison

Retain Plan 1’s Fit/Tune/Calibrate/Audit allocation of 60/10/10/20 by S1 owner. Reuse the prospectively locked Audit set and do not tune on it. For Plan 2, divide Calibrate deterministically into two equal groups:

- **C-prob:** fit probability calibration.
- **C-select:** choose the final set-decision thresholds and other small policy parameters.

Train dictionaries, alignment rules and models only on Fit; use Tune for iteration and early stopping. Run three business-grouped development folds for the shortlisted changes, rebuilding supervised normalization inside each fold. Full cross-validation is unnecessary for obviously losing ideas.

All recorded scores must use the exact per-business rule:

    F0.5 = 1.25 × true_positives / (0.25 × true_count + predicted_count)

Empty truth plus empty prediction scores 1. Macro-average across all queries. Optimize after ownership resolution and every rejection step.

### Full pool and realistic ownership worlds serve different purposes

Continue to measure candidate recall and direct matching on a fixed query cohort against **all training S2/S3 records**. Increase this cohort to 50,000–100,000 queries for close comparisons and rare slices. Report the complete candidate-stage recall, not just pre-filter retrieval recall.

For global ownership and distribution-shift experiments, also create **controlled held-out worlds**:

1. Select a large, complete group of held-out S1 owners.
2. Include all their true targets as the world’s matched population.
3. Add original unmatched targets, stratified by country and source, to produce a measured distractor fraction.
4. Retrieve normally. Labels define the world and score it; they must never insert records into an individual query’s candidate set.
5. Score every S1 in that world together, including singletons, so all its potential competing owners participate.

These worlds are smaller and label-constructed. Label them as controlled diagnostics and corroborate retrieval improvements on the full target pool. Do not present them as full-scale or French validation.

### Correct the distractor-shift experiment

The training distractor share is 25.99%. Approximately 40% on test is an estimate assuming a similar match-count distribution, not a measured test label statistic.

For a world with P matched targets and desired distractor fraction d, include:

    D = d × P / (1 − d) unrelated target records

Test approximately 26%, 40% and 50% distractor worlds **with the same active query set**, increasing the target pool. Use actual unrelated records from the provided data, not duplicates of existing negatives. Log the achieved fraction per country/source and actual candidate-negative counts.

Separately, test missing owners: remove some S1 queries but retain their targets. Starting at d0 = 0.2599, hiding a fraction h of owners gives approximately:

    d1 = d0 + h × (1 − d0)
    h ≈ 0.1893 to move from 25.99% to 40%

That equation assumes hidden owners have representative match counts. Measure the achieved result.

**Important distinction:** hiding queries with an unchanged target pool does not make a fixed pairwise matcher’s predictions harder for the retained queries. It mainly changes the set of competing owners. The added-target experiment and missing-owner experiment therefore test different failure modes.

Do not automatically adjust class probabilities using the 40% estimate. Target-row distractor prevalence is not the same quantity as match prevalence among retrieved candidate pairs.

### Generalization checks

- Train on US and evaluate on India, then reverse the direction. Decompose changes into retrieval loss, dictionary coverage, matching errors and calibration drift.
- Run a smaller name-family holdout, grouping repeated cleaned S1 names together, to test behavior on unfamiliar names.
- Stress held-out records with training-supported corruption patterns: missing address parts, word reordering, accent changes, legal-form changes and number formatting.

A cross-country score drop above 0.03 is an investigation trigger, not proof that a specific feature is bad. Unseen scripts and different noise distributions can explain it.

## 3. Improve normalization where it changes decisions

Start from the basic text views specified in Final Plan 1 and build the following additional representations in this run:

| Area | Implementation |
| --- | --- |
| Trade names | Split names at dba, aka, fka, formerly, t/a and “doing business as”; compare each nonempty side |
| Websites | Preserve domain stems and joined-token views; support common compound endings such as .co.in |
| Indic names | Build a Fit-only token dictionary with generic transliteration fallback and distinct-owner alignment evidence; store mapping support, agreement, runner-up and unknown-token fraction |
| OCR-like noise | Add a separate view for supported letter/digit confusions inside words; leave the conservative text and address numbers intact |
| Address numbers | Distinguish likely house, unit/floor, compound number and possible postal-code evidence |
| Geography | Canonicalize observed state aliases in address context; retain French region/department strings without treating them as equivalent identities |

Do not turn token alignment into unconstrained phrase translation. Support a small number of recurring multi-token forms only when the training evidence is clear. For an ambiguous Indic token, preserve the generic fallback and expose uncertainty; at most two alternatives may be tested as bounded retrieval views.

Make number-parser uncertainty explicit. Preserve “47 bis”, “47 ter”, “576B” and “12/3”. Recognize that address components can move. A mismatch between two uncertain first-number estimates is weaker evidence than a conflict between two confidently parsed house numbers.

Treat postal-code candidates as optional evidence. The findings report no six-digit postal codes in Indian S1 and very few in France, so postal blocking cannot be a core dependency.

### France handling

Apply the already observed legal forms and abbreviation patterns through deterministic text views: SARL/SAS/SASU/EURL/SCI, rue/r, avenue/av, allée, n° and bis/ter. Keep accents in the original view and tolerate missing/split initial letters through character similarity.

Do not equate region and department tokens through external lookup. Do not infer French thresholds from same-name “likely pairs”: they are unlabeled. Use the shared model and general missingness, script, numeric and ambiguity features. Report French output coverage and score distributions only as diagnostics.

## 4. Strengthen retrieval without multiplying indexes indiscriminately

Build word and character searches on combined name plus address. Final Plan 1 contains only the word channel. Start the character channel with the existing benchmark's 2–4 character grams and weighting settings, then measure its contribution within this run. Evaluate the following additions separately:

| Channel | Query and target population | Purpose |
| --- | --- | --- |
| Missing-address name search | Every S1 name against only targets whose address is missing | Retrieve invisible missing-address matches even when other matches are easy |
| Alias/domain retrieval | Conservative and reduced alias/domain views | Recover website and trade-name variants |
| Address-only search | Initially every development query against same-country targets | Test recovery of opaque aliases and weak-name cases |

Start with top 20 per source for the missing-address channel and top 10 for each special channel. These are proposals to benchmark, not known optimal budgets.

**Do not trigger missing-address search only when the S1 address is empty.** S1 addresses are present; it is the unknown target’s address that is missing. A dedicated target-side subset makes the condition usable at inference.

Address-only search is lower priority: the old full-pool combined search already recovered 98.86% of its weak-name/strong-address slice. Retain a new channel only for unique useful recovery. Likewise, do not assume global name-only search helps just because missing-address search helps.

### Candidate fusion and budgets

Store candidate ID, source, channel, rank and score, then deduplicate by S1/target. Fuse ranks using reciprocal-rank fusion, starting with:

    fused_score = sum over channels of 1 / (60 + channel_rank)

Protect a small quota from genuinely complementary channels before filling the remaining budget. Otherwise a useful missing-address result can be crowded out by redundant combined-text results. Validate the quota itself.

Start at **100 final candidates per source**, including all channels. Compare 50, 75, 100 and 150 per source on the same queries. Report raw-union and final-capped recall and the marginal gain per additional million scored pairs.

Adaptive budgets are a later refinement: expand queries with common names, weak retrieval agreement or poor distinctive-token coverage. Check that the routing preserves difficult *additional* matches for businesses already having easy ones; a high top score alone does not mean retrieval is complete.

Targets for investigation are at least 99% overall link recall, 98% in each labeled country, and substantially better Indic/missing-address recall than Plan 1. These are engineering targets. Selection still uses measured oracle macro F0.5 and final score, and there is no French recall target measurable from the available labels.

## 5. Use stronger evidence in one direct matcher

Extend the Plan 1 features with:

| Group | Important additions |
| --- | --- |
| Name detail | Best alias similarity; domain similarity; rare-name-token overlap; conservative versus aggressive disagreement |
| Address detail | Shared IDF weight, missing distinctive-token weight, coverage both ways, estimated street/locality overlap |
| Number detail | Typed agreement/conflict, parser confidence, unit conflict and compound-number preservation |
| Reliability | Indic mapping coverage/support; short-address flag; number of distinctive tokens |
| Ambiguity | Same-name S1 frequency; candidate counts; retrieval margin to the best competing S1 |
| Channel evidence | Channel-specific ranks/scores, exact-view support and independent-channel agreement |

Compute rarity as a relative corpus statistic, per country/source where appropriate. A fixed “document frequency below 20,000” threshold has different meanings for countries of different sizes.

For two token sets A and B with IDF weights, useful address features are:

    shared_weight = sum of IDF weights in A ∩ B
    coverage_A = shared_weight / sum of IDF weights in A
    coverage_B = shared_weight / sum of IDF weights in B

Handle empty denominators explicitly. Add unshared distinctive weight rather than relying on token-set similarity alone, which can be high for a very short contained address.

Reverse-claim features at this stage should use label-free retrieval/text evidence. If model-derived features are introduced, generate them out of fold; a model must not provide its own in-sample “confidence” as a training feature.

### Training population and hard negatives

Run a learning curve at approximately 100,000 Fit owners, 500,000 and the remaining Fit population if the gain continues. Keep all candidates for each selected owner where practical.

Use three development folds to mine mistakes from predictions made on owners excluded from that fold’s model. Deliberately retain:

- Same cleaned name at another address.
- Same street/city with a conflicting house or unit number.
- Very similar address and almost-identical name with a negative label.
- Singletons whose strongest candidate looks convincing.
- Candidates strongly claimed by a different S1.

Keep the supplied labels, including convincing negative examples such as Ivenent Metal. Do not hand-correct them based on appearance.

Hard mining should enrich a representative training sample, not replace all normal candidates. If negatives are downsampled, save the selection policy/probabilities and validate calibration on unsampled candidate populations.

Try a small LightGBM parameter sweep around Plan 1: 31/63/127 leaves and minimum leaf sizes 100/300, holding other settings fixed initially. Select on Tune macro behavior and loss; report the score after the complete decision policy. Do not launch a large architecture search before analyzing the remaining errors.

## 6. Calibrate and decide at the right level

Fit a simple sigmoid calibration map on C-prob predictions generated by a model that did not train on those owners. Examine reliability especially in the high-score acceptance region. Use one shared calibrator first; source-specific calibration is an ablation.

Compare these small policies on C-select:

| Policy | Behavior |
| --- | --- |
| P1 | One threshold for every candidate |
| Two thresholds | Require a sufficiently strong best candidate to start a set; permit additional candidates above a separately tuned link threshold |
| Source thresholds | Different S2/S3 thresholds only if stable gains justify them |

For the two-threshold policy:

    If best candidate score < start_threshold: return empty
    Otherwise: retain every candidate score ≥ link_threshold

Tune the thresholds jointly with ownership decisions. Do not fix them at 0.5, force an expected number of matches, require a constant gap from the best candidate, or create a separate France threshold.

### Competing owners

For each target record, gather all proposed claims across the complete query population:

1. Compare the best and runner-up calibrated claim scores.
2. Accept the best only if it passes the chosen acceptance rule.
3. Test abstention when the best claim is only marginally stronger than another plausible claim.
4. Reapply the entity start rule if its supporting best claim was removed.

Measure whether ambiguous abstention actually improves macro F0.5. A high-scoring wrong claimant is possible, so “always choose the winner” is not sufficient evidence.

A target has capacity one; an S1 has no fixed capacity. The basic problem therefore needs a reduction by target ID, not a Hungarian one-to-one assignment across businesses.

Compute scores, ownership margins and output decisions in separate passes so partition boundaries cannot hide rivals. Never use known training ownership to settle a validation claim.

## 7. Execution sequence and ablations

| Order | Experiment | Required comparison |
| --- | --- | --- |
| 1 | Build this run's direct-matcher baseline and shared evaluation worlds | Save a reproducible internal control; compare with Plan 1 when its frozen result is available |
| 2 | Better alias, number and Indic representations | Retrain matcher; isolate normalization from extra candidate budget |
| 3 | Add missing-address search | Incremental true links, oracle gain and final F0.5 |
| 4 | Try alias/domain and address-only channels individually | Same budget and larger-budget operating points |
| 5 | Add richer features and representative hard-negative mining | Same candidate set; precision and singleton effects |
| 6 | Scale Fit owners along the learning curve | Gain versus more rows and training time |
| 7 | Calibration, two thresholds and ambiguity abstention | Same raw model predictions; full end-to-end score |
| 8 | Cross-country, name-family and distractor stress checks | Explain regressions before selecting the final configuration |
| 9 | Freeze, audit and run test inference | No parameter changes from test patterns or audit score |

For each added feature group or retrieval channel, also remove it from the **final Plan 2 configuration**. Something useful early can become redundant after another improvement.

Proposed retention rules, fixed before the experiment:

- Cheap deterministic changes: at least +0.0005 macro F0.5 with a positive paired confidence interval, or equivalent accuracy with a clear operational benefit.
- A new index or learned component: at least +0.0015 macro F0.5, positive paired confidence interval and gains in more than one development fold/world.
- Investigate material regressions in singleton behavior, either labeled country or the higher-distractor worlds. Do not hide them inside a pooled average.

Use paired resampling of the same S1 outcomes. For ownership-policy comparisons, additionally resample collision groups to assess dependence between businesses sharing candidate records. These floors are engineering choices, not properties of the metric.

The score floors govern optional complexity. Required output correctness, Unicode preservation and open-country coverage remain part of the contract. Deterministic French-format support is a documented generalization assumption checked with parsing/robustness examples; absent French labels, do not claim it has passed a French accuracy ablation.

## 8. Completion, packaging and what stays out

**Plan 2 is complete when** its frozen configuration has been measured under the shared comparison protocol, each retained addition has a reason to remain, and the full inference package reproduces the selected result. Claim an improvement over Plan 1 only after both frozen results are available and the comparison supports it.

Store input/split hashes, dictionary and feature versions, candidate ledgers, model, calibrator, decision settings, per-entity predictions and metrics. Preserve the exact direct-matcher candidate population in candidate_pairs.tsv. All final matches must be a subset.

Export one row for every test S1, including France and empty predictions. Run the official validator with --check-ids plus internal hard checks for candidate-subset membership, country equality, ID uniqueness and ownership. Fill the methodology template and pin dependencies. The [official specification](../student_resource/README.md) governs the package.

The baseline model is [MIT-licensed LightGBM](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE). No external record enrichment, geocoding, business lookup or hosted conversion of the records is part of this plan.

The following remain outside the default Plan 2:

- Duplicate expansion: the reported 75.9% rescue opportunity was found using true-owner labels, and cannot be treated as realized recall.
- Separate no-match classifier: first measure what calibrated set thresholds already solve.
- Model ensembles: first establish that another model makes different useful errors.
- Dense retrieval and neural reranking: the existing embedding experiment did not include a comparable full-pool sparse baseline.
- Pseudo-labeling and iterative graph propagation: their errors can reinforce one another.

These omissions are deliberate. Plan 3 provides bounded experiments for the remaining high-value gaps.

Evidence for the priorities and limitations comes from the [consolidated findings](https://github.com/sohambuilds/icmliclrneuripsaaaiemlnlpmaxxing/blob/main/EDA_FINDINGS_ALL.md), the [full-pool retrieval report](../eda/full_sparse_20260925/RESULTS.md), and the [independent evaluation scope](../eda/independent_20260925/README.md).
