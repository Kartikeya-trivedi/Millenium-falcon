# Plan 3: build from the 0.9388 submission

25 September 2026. This is the current execution plan. The earlier Plan 3 document remains a reference for the later relationship experiment.

## Objective and evidence

Improve the evaluation's **macro F0.5 across every S1**, including singletons. The initial retrieval operating target is **at least 99.5% candidate recall, mean at most 400 and p95 at most 500 candidates per S1**. Keep **99.99% as a stretch aspiration**, not the first delivery requirement. Candidate recall is the fraction of true links available to the matcher; accepted-link recall is the fraction finally emitted. They are different targets.

The supplied Plan 1 report gives:

| Metric | Reported v2 result |
| --- | ---: |
| Public external evaluation F0.5 | 0.9388 |
| Old Audit macro F0.5 | 0.9523 |
| Candidate recall | 98.5% |
| Candidate-oracle macro F0.5 | 0.995 |
| Accepted-link precision / recall | 98.5% / 89.9% |
| Missing-address candidate recall | 85.8% |
| Indic-name candidate recall | 98.0% |

These are the teammate's completed measurements, accepted as the Plan 1 baseline. The source is [the supplied repository](https://github.com/sohambuilds/icmliclrneuripsaaaiemlnlpmaxxing), pinned locally at commit 2691386. **Do not reproduce Plan 1 locally.** The user explicitly ruled out that redundant run. The report, source and supplied datasets are sufficient to choose and implement Plan 3 experiments.

Reuse the teammate's saved models, candidates and predictions if they become available. Their absence must not block Plan 3 implementation or trigger an automatic baseline rebuild. Record new experiments' own populations and metrics, and distinguish comparisons against the reported baseline from paired comparisons using the same owners and saved predictions.

Perfect retrieval alone would improve the reported candidate oracle by at most 0.005 on that old panel. Most remaining score loss is below that ceiling: correct candidates rejected, wrong candidates accepted, or competing owners resolved incorrectly. Plan 3 therefore improves retrieval and decisions together.

99.99% means at most one missed link per 10,000 true links in the measured population. A small Tune panel cannot substantiate such precise population performance. Always report miss counts, candidate budgets, full-corpus coverage, and held-out population sizes alongside the percentage. If no configuration meets the 99.5%/400/500 operating target, report that failure and the measured frontier; do not silently inflate the budget or conceal missed links.

## Compact implementation contract

- **Decision:** a target may match zero or one S1; an S1 may match zero, one or many targets.
- **Primary metric:** macro F0.5 on all evaluation S1s with a frozen decision rule.
- **Guardrails:** link precision/recall, singleton false-positive rate, per-country and missing-address/Indic performance, mean/p95/max candidate count, candidate oracle and runtime.
- **Mistake budget:** there is no license to trade arbitrary false merges for recall. Retain additions only when held-out score improves without an unexplained collapse in a important slice. Report uncertainty rather than inventing a guaranteed improvement threshold.
- **Data:** only supplied business records and labels. No external business lookup. IDs and countries are bookkeeping/filter keys, not learned identity shortcuts.
- **Splits:** reproduce the upstream salted Fit/Tune/C-select/Audit roles. Use Tune for retrieval/model development; C-select for thresholds. The old Audit panel is exposed development data. Reserve a new Audit panel from the remaining Audit role, excluding the old panel, with a separate sampling salt.
- **Baseline:** upstream v2 normalization, Fit-role transliteration dictionary, W50 per source, 28 features, one LightGBM and one-owner output.
- **First experiment:** use the reported v2 error profile to prioritize improved matching and a bounded W200 plus missing-address retrieval experiment. Implement the four-class ledger for saved teammate predictions when available and for every new Plan 3 scored run. A missing baseline prediction artifact is not a prerequisite for starting.
- **Fallback:** retain the measured v2 configuration if new channels/models do not help. Do not overwrite Plan 1 or Plan 2 artifacts.

## 1. Retain the richer Plan 3 architecture

The original Plan 3 remains the architecture we are building toward. The recent review changes execution priorities and narrows the first retrieval experiment; it does not remove the richer matching feature families or the planned second-stage matcher.

    Plan 1 v2 text treatment, transliteration and 28 existing features
      -> richer direct evidence and a first-stage matcher
      -> independently generated first-stage scores
      -> sibling, cross-source and competing-owner evidence
      -> second-stage matcher
      -> validated acceptance and ownership decisions

The earlier direct-tree blueprint in DIRECT_MATCHER_REFERENCE.md supplies the direct-feature families. RELATIONSHIP_REFERENCE.md supplies the support model and controlled one-hop design. Those documents are references; the execution priorities and boundaries in this final plan take precedence.

| Retained feature family | Evidence added beyond the actual v2 baseline |
| --- | --- |
| Name detail | Trade-name/alias and domain comparisons; overlap on rare name words; disagreement between conservative and reduced text |
| Address detail | Shared distinctive-word weight, unmatched distinctive words and coverage in both directions; distinguish a short contained address from a fully supported match |
| Number detail | House versus unit/floor evidence where parsing is reliable, compound-number agreement, conflicts and parser uncertainty |
| Transliteration reliability | Dictionary coverage and mapping evidence; unknown-token fraction; comparison with conservative/raw text when conversion is uncertain |
| Retrieval evidence | Separate score/rank and presence for each retained channel; agreement across complementary channels |
| Sibling and cross-source support | Evidence from other reliable records for the same S1, separately for the same source and the other source; distinct support groups and contradictions |
| Competing-owner evidence | Strongest other claimant, score/text margins and ambiguity across the declared claimant population |

V2 already has 28 features, including several name similarities, relative ranks and basic anchor similarities. Carry those forward and add genuinely different evidence. Do not present an existing v2 feature as a new Plan 3 contribution.

Implement and evaluate coherent feature groups separately so their incremental value remains visible. The second-stage matcher is a planned Plan 3 experiment, not an automatic promise that two models will be retained in the final submission. One-hop candidate expansion and a neural/dense specialist remain later experiments, justified by remaining errors.

## 2. Carry forward the completed v2 baseline, including transliteration

Import Plan 1 source unchanged and record its file hashes. Reuse the learned Indic dictionary procedure with Fit-role labels only, number-marker handling, state/phrase replacements, raw missingness flags, 28 pair features, global ownership constraint and portal row ordering.

Transliteration is already part of the working v2 baseline. It is not an optional feature to rediscover, and its use in Plan 3 does not depend on the original minimal Plan 1 scope.

| Text treatment | Exact first action |
| --- | --- |
| Raw fields | Preserve original names and addresses; never overwrite them with Latin text |
| Indic word dictionary | Rebuild from Fit-role matching pairs only; freeze the mapping before Tune/C-select/Audit inference |
| Unknown Indic words | Retain v2's AnyAscii fallback and documented suffix/spelling fixes |
| Unicode | Preserve Indic vowel signs; remove invisible joiners without splitting words; retain v2 Latin accent handling |
| Legal forms and states | Reuse v2 phrase/dotted-form replacements and state/address normalization |
| Number evidence | Preserve meaningful 12B, 12/3 and A-5 distinctions; handle No.5 and N°49 as v2 does |
| Unknown countries | Accept their country strings and apply the same field-based pipeline; France must be represented |

The local preparation already produced **1,205 dictionary words**, matching the report. Reuse that prepared data for Plan 3. No additional Plan 1 reproduction is required.

Carry forward the source implementation of W50, the 28 features and the reported threshold/score as reference evidence. New training is for an explicitly changed Plan 3 experiment, with its changed features, candidate set or training schedule recorded. Do not run the original 300,000-owner training recipe merely to reconfirm the teammate's score.

Inspect the upstream evaluation boundary: report pre-ownership and post-ownership scores separately rather than comparing one to the other. Do not tune on old Audit to force the published number or claim a paired improvement from evaluations on different owner populations.

The v2 changes were bundled. Their individual gains have not been isolated. Feature importance does not establish causal contribution; two external evaluation submissions do not establish a universal external evaluation correction.

Preprocessing runs identically during train and inference. Cache keys include raw input content hashes, preprocessing source, dictionary, retrieval parameters and query membership. A conflicting cache must fail, never silently load.

## 3. Build the complete error ledger alongside Plan 3

The aggregate report already justifies working on rejected true candidates, false acceptance, missing addresses and ownership. Start those bounded experiments now. Use the detailed ledger to refine subsequent choices; do not wait for a redundant full baseline run to produce it.

Save these four error classes with S1/target IDs, raw and normalized fields, script/missingness, retrieval rank, model probability, threshold, sibling support and claimant evidence when available:

1. **Retrieval miss:** a true link never entered the candidate set.
2. **Retrieved true link rejected:** a true candidate failed the scoring or acceptance decision.
3. **False acceptance:** an incorrect candidate passed the decision, whether or not ownership later removed it.
4. **True link lost at ownership:** a correct accepted link was removed during conflict resolution. Record the winning claimant and score margin.

Also retain a separate log of incorrect links successfully removed by ownership. Removing them is a benefit, not an error.

Attribute score loss per S1 with an explicit ordered counterfactual:

    all true links (F = 1)
      -> only retrieved true links                     : retrieval loss
      -> only true links accepted before ownership    : rejection loss
      -> all links accepted before ownership          : false-acceptance penalty
      -> final output after ownership                 : signed ownership loss/gain

Average the differences across the entire Tune roster, including zero-match businesses. The four signed contributions sum to final macro-score loss. This is a diagnostic decomposition under this ordering, not a causal experiment proving what a new feature will fix.

For each class, show owner counts and macro-score loss by country, Indic/Latin target name, missing address, singleton/non-singleton, source and number-conflict status. Use it to choose the next intervention. The reported 0.043 matcher/decision gap takes priority over chasing the final 0.005 of oracle loss.

## 4. Improve decisions on existing candidates first

Keep the candidate set fixed while testing these individually:

Begin by adding the direct-feature families in Section 1 as measured groups to the existing 28-feature implementation. This produces the richer first-stage scorer. Then evaluate the following additions, using independently generated first-stage predictions for the second-stage model:

1. **Training headroom:** retain the 28-feature reference; allow up to 6,000 boosting rounds with early stopping. The original cap was reached, but longer training is retained only if Tune/C-select behavior improves.
2. **Score-aware sibling support:** compare each candidate with other high-confidence candidate records for that S1. Add best supporting score, name/address agreement, compound-number conflict, distinct supporting groups and direct/support disagreement. Use no new retrieval in this experiment.
3. **Competitor evidence and abstention:** add the strongest *other* S1 claimant's score, winner/runner-up gap, claim count and contradictory field evidence. Compare a validated rejection rule with the baseline highest-score owner.

V2 already has anchor name/address similarities. The sibling experiment must improve on those existing features, not duplicate and rename them. The EDA's roughly three-quarters sibling opportunity uses labels to identify siblings; it is not a measured deployable rescue rate.

A candidate cannot support itself. Exact duplicate evidence counts as one group. No score becomes a truth label, and no support prediction automatically becomes an accepted match.

**Leakage contract:** training support features must come from models and learned dictionaries that excluded the corresponding support-training owners. A small disjoint fit/score/support-training experiment can test the idea first; full fold-based regeneration is justified only if it helps. Tune, C-select and fresh Audit labels never create support features. Save the exact upstream training ownership for every score artifact.

Competitor features require a declared claimant population. A 20,000-query panel omits most rivals. Before promoting these features or an ownership rule, obtain predictions across a complete held-out query world or a clean cross-fitted claimant population. Small-panel collision counts cannot stand in for full-test evaluation.

## 5. Limited retrieval experiment: W200 plus missing-address search

Every search uses all targets of the query's country and source, including targets without a labeled owner. Country handling accepts arbitrary strings, including France. No label is used to inject a missing positive.

Generate two proposal channels on Tune:

| Channel | Initial maximum per source | Purpose |
| --- | ---: | --- |
| W: v2 word TF-IDF, name plus address | 200 | Preserve W50 and measure whether larger K recovers the tail |
| M: character name search among missing-address targets | 100 | Give the weakest observed slice its own retrieval budget |

The diagnostic proposal pool has at most 600 candidates per S1 before overlap. This is not automatically the set used to train the final model. From cached rankings compare W in {50, 100, 200} and M in {0, 25, 50, 100}, per source. Measure the **deduplicated final union** across all queries, including queries with no candidates.

Only configurations with **mean <=400 and p95 <=500 candidates per S1** are eligible for the first expanded-matcher experiment. Prefer the smallest eligible set that reaches 99.5% recall. If none does, report recall, miss counts and oracle gain for the eligible frontier and decide using final F0.5, not recall alone. Preserve W50 for a paired control.

Do not add C (character name plus address) or N (name-only search across all targets) in the initial run. They remain optional code paths for a later residual-error experiment. Wider retrieval increases ambiguity and training cost; a fixed v2 threshold or unretuned matcher is not a valid expanded-pool comparison.

Use exact sparse top-K within target shards with a common full-partition vocabulary and IDF. Merge shard top-K globally. This keeps corpus coverage real while bounding working memory. Use a fixed target ordering for repeatable ties; keep source and query identifiers as strings without truncation. Cache channel results and provenance separately.

The stopping rule is evidence: enumerate remaining missed true links and their macro-score cost. More transliteration work is justified only when that ledger shows actual conversion failures. Test dictionary coverage, incorrect mappings and fallback errors separately; do not use Tune/Audit labels to edit an inference dictionary. A richer transliteration model, C/N channel, address-only index or dense index must beat simpler fixes on untouched evaluation data.

## 6. Retrain for any retained candidate expansion

Expanded retrieval changes the negative population and relative ranks. Never feed new retrieval scores into a v2 model and assume they mean the same thing.

Retain v2 direct similarities. Keep actual word cosine as word cosine for every candidate, even when a different channel retrieved it. Preserve per-channel scores/ranks, absence flags and W50 membership. Recompute relative/anchor features over the complete candidate set of each S1; do not split an S1 across feature batches. Train a new direct LightGBM on that distribution.

Include all retrieved positives; missing positives remain retrieval misses in evaluation. Include difficult retrieved negatives, particularly same-name branches, missing-address distractors, incorrect near duplicates and owner competitors. If training negatives must be sampled, record the scheme and weights; tune decisions on the unsampled candidate population.

The v2 learner hit its 2,000-round cap while validation loss was improving. Permit up to 6,000 rounds with early stopping on Tune, log best iteration, and warn if the new cap is hit. This is a hypothesis, not proof that more rounds improve F0.5. Select the final threshold on C-select using the actual evaluation metric.

First establish the richer direct scorer, then train the planned sibling/competitor second-stage matcher on clean upstream predictions. Compare both on the same held-out population. Additional neural specialists, ensembles or calibration stages require their own incremental evidence.

## 7. Treat unowned targets as a training and robustness concern

The full S2/S3 corpus is always used during retrieval, including targets with no labeled S1 owner. Therefore the reference already encounters retrieved unowned targets; they must not be removed from training.

Measure three different populations separately: the fraction of all target records with no labeled owner, their share among retrieved candidate negatives, and their share among false acceptances. The reported or estimated roughly 40% test orphan share does not imply that 40% of training pairs should be sampled as orphan negatives.

Keep hard retrieved unowned negatives in training. For a stronger mismatch test, build a controlled missing-owner world: remove selected owners from the S1 query population while retaining their S2/S3 records as distractors. Selection must not use their model scores or evaluation errors. Measure candidate mix and false merges as the owner-removal rate changes, including a world whose target-level unmatched share is near the reported test estimate.

Use these retrieved distractors in a separately recorded training ablation or weighting experiment. Do not relabel a correct pair as negative while its owner is still a query. Do not impose an arbitrary 40% pair ratio, use orphan-status labels as model inputs, or select a policy using unavailable test labels.

## 8. One-hop expansion only if existing-candidate support leaves useful misses

This is later than the sibling-support experiment in Section 4. It changes the candidate set; the earlier experiment only changes evidence on existing candidates. If measured retrieval misses have reliable predicted siblings, test the one-hop design in the earlier Plan 3:

1. Select highly reliable direct matches as seeds using held-out predictions.
2. Search for similar S2/S3 records near those seeds, once.
3. Add proposed targets as candidates, never as automatic matches.
4. Add direct/support disagreement, number conflicts, distinct support groups and opposing-owner evidence to a final model.

Do not use a candidate as its own supporting seed, duplicate records as independent votes, training labels as inference edges, or unrestricted connected-component propagation. Fold-trained dictionaries, seed scores and support features must exclude held-out owners. Freeze the direct baseline first so the relationship contribution can be measured.

Initial caps remain two seeds per source, twenty neighbors per seed and forty added candidates per source. These are tunable budgets. Build this pass only after the measured direct error ledger justifies it; it is not part of the first retrieval milestone.

## 9. Ownership and empty answers

Keep the useful one-owner constraint. Resolve ownership over all inference query chunks together; doing it separately in each chunk is incorrect. Keep a deterministic tie break.

A contested target can have no correct claimant. Winner margin and weak absolute evidence can justify abstention after validation. Do not impose arbitrary rules such as rejecting every target claimed by ten businesses.

A small sampled panel understates global evaluation. Its post-ownership result is labeled **panel-only ownership**, not a faithful estimate of full-test assignment. Before promoting a new ownership policy, evaluate a complete held-out query world with full targets and missing-owner distractors, or a clean cross-fitted claimant population. Do not use in-sample competitors to claim held-out performance.

## 10. Honest evaluation and release

Use development probes against the full target corpus, not a target subset containing known answers. Increasing query samples changes confidence, not corpus coverage. Tiny smoke runs verify execution only.

For each measured run, save:

- Per-query counts, F0.5 and the four-class ledger from Section 3, including rejected true candidates and true links removed by ownership.
- Recall, oracle and candidate budgets for each retrieval ablation.
- Exact model/features/preprocessing/splits, dependencies, input hashes and selected threshold.
- Before/after ownership scores with claimant-population limitations.
- A run status that distinguishes prepared, retrieved, trained, development-evaluated and fresh-Audit-evaluated.

The new Audit is used after choices are frozen, not as another tuning loop. Near-perfect observed recall on a small query sample is insufficient evidence of 99.99% population recall. Confidence calculations must respect that links belonging to one S1 are correlated.

Final output contains every test S1 exactly once **in original test_source1.tsv order**, even when its candidate or match list is empty. Write UTF-8 without BOM, tab separators, no quoting and LF newlines. Matching IDs must be a subset of the exact candidates scored, IDs must exist, country/source contracts must hold and targets must have at most one emitted owner. Run both internal checks and the supplied official validator.

## Execution order

| Stage | Action | Required output before advancing |
| --- | --- | --- |
| A. Reuse | Accept the teammate's completed v2 result; reuse its source, text treatment and already prepared data | Recorded reported baseline and experiment contracts; no repeat Plan 1 run |
| B. Diagnose while building | Start from the reported error profile; add the four-class ledger to available saved scores and new Plan 3 runs | Macro-score loss by class/slice as predictions become available |
| C. Improve decisions | Richer direct-feature groups, training headroom, then the planned sibling/competitor second-stage matcher and abstention experiments | Paired results on fixed candidates; clean upstream prediction provenance |
| D. Repair retrieval | Test W200+M only where the ledger warrants it; choose budget-eligible ranks | Recall/miss/oracle table, mean <=400 and p95 <=500; retrained matcher comparison |
| E. Address residuals | One-hop retrieval or one specialist only for an observed remaining problem | Incremental score gain beyond the chosen simpler pipeline |
| F. Freeze and release | Choose thresholds on C-select, evaluate fresh Audit and full-world ownership, then infer test | Validated ordered TSVs, exact candidate ledger and reproducible package |

Stages C and D can proceed from the already reported evidence while B is implemented. Their results are combined only after isolated comparisons. Plenty of compute enables better measurements; it does not justify repeating the teammate's completed run or installing every proposed component.

### Current implementation status

- Imported the upstream v2 logic; the published copy has non-functional wording edits.
- Reconstructed all 2,206,821 training owners' split roles and the 1,205-word dictionary.
- Prepared all train and test source/country partitions with input fingerprints.
- Reserved 50,000 fresh Audit owners outside the old 20,000-owner panel.
- Stopped the redundant local v2 reference retrieval at the user's request. Its partial artifacts are not a completed or reproduced baseline and do not gate Plan 3.
- Measured full-target W/M retrieval on the 2,000-owner Tune panel. W100/M25 recalls 99.3065% of 6,921 true links at mean 237.6 and p95 250 candidates. No measured setting reaches 99.5% within both declared budgets.
- Trained the richer 67-column direct model on 10,000 Fit owners. Its exposed 2,000-owner development subset scores 0.951625 macro F0.5; this is not a paired or external gain over v2.
- Saved the four-class error ledger. Tune has 48 retrieval misses, 572 rejected true candidates and 89 false accepts; no ownership loss was observed in this small panel.
- Trained independently scored sibling features with 3,000 c_prob owners for separate second-stage training. That pool is repurposed and must not also be used for probability calibration. Development macro F0.5 improves to 0.960921 versus 0.951625 direct and 0.952165 for the retraining control without relationship features. Full-world competitor probabilities are not yet learned model features.
- Built a server runner, stage logs, result packaging and uv-based environment management. Launched the full profile on Modal on 25 September 2026: 300,000 direct-training owners, 50,000 separate support-training owners and 20,000 per evaluation panel. All 36 tests passed on its Linux worker before preparation. Training results are pending.
- Measured score-mined negatives and a separate missing-address threshold; neither improves the sibling pilot's development macro score. A bounded one-hop probe recovers eight of 48 Tune retrieval misses but adds only 0.000198 to the candidate oracle. These variants remain separate experiments.
- Deferred C/N retrieval channels and retained full-world ownership validation and missing-owner stress training as outstanding work before final promotion. Fresh Audit remains unused.
- Completed label-free test views and all W/M indexes on Modal, including France. Implemented checkpointed frozen-model inference, guarded Audit evaluation and ordered output with global ownership. The locked environment passes 67 tests; a real 20-owner inference consistency check matches all 4,932 saved candidate probabilities exactly. Full-model selection, fresh Audit, complete claimant-world evaluation and final test inference are pending.
- Rejected the separate unowned-negative weighting ablation: doubling those weights reduces pilot development macro F0.5 from 0.960921 to 0.958175. Unowned negatives remain present under the original weighted sampling scheme.

The completed teammate baseline is accepted without local reproduction. The candidate budget frontier, changed direct matcher and sibling comparison are measured on the pilot; the recall target is not yet met and any final-score improvement remains unmeasured. PILOT_RESULTS.md records the exact populations and limits.

No new external evaluation score, 99.99% recall, or completed submission is claimed until that particular artifact has actually been measured or produced.
