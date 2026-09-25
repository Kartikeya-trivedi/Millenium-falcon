# Server handoff: train the richer record matcher

This release is ready for the next direct-matcher training experiment. It also contains a separately selectable sibling-support experiment. Use the terminal commands below. The first-generation model is not retrained by these commands.

## What to run now

Clone this repository, install the pinned environment, attach the four training TSVs, and run the **full profile through train**. Return the compact result bundle. No GPU is required by either LightGBM model.

The full profile has 300,000 direct-model training owners, 50,000 separate support-model training owners, 20,000 Tune owners, 20,000 threshold-selection owners and 20,000 exposed development owners. Every query searches all relevant targets in its country and source, including unowned targets. The 50,000 fresh Audit owners remain reserved.

Hardware planning estimate: 16-32 CPU cores, 128 GB RAM minimum planning target, preferably 256 GB for headroom, and at least 300 GB free SSD space for the full profile. The current implementation materializes some large country/source candidate tables; it is not constant-memory. These are capacity estimates, not a measured full-run peak. The smaller pilot ran on a roughly 26 GB workstation; a 64 GB server is suitable for that profile. Start a pilot if a full-sized machine is unavailable. Do not silently reduce the target corpus to fit memory.

The default scoring set is W100 plus M25 per target source: word retrieval proposes 100 targets and missing-address name retrieval adds up to 25. Initial retrieval saves W200/M100 results so a budget comparison is available. The scoring choice is explicit; the automatic budget recommendation is recorded separately and does not override it.

## 1. Install on a Linux server

Use uv from the repository root. It installs Python 3.12 and creates the project environment from the committed uv.lock. Linux LightGBM also needs the system OpenMP runtime, commonly libgomp1. There is no CUDA setup for this release.

```bash
git clone https://github.com/Kartikeya-trivedi/Millenium-falcon.git
cd Millenium-falcon
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --locked
POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run --locked python -m pytest plan3/tests -q
git rev-parse HEAD
```

Skip the installer if uv is already available. The default sync includes the development group used by the tests and preflight. Do not pass --no-dev. Dependency versions match the measured pipeline; uv adds reproducible dependency resolution and cached downloads. See [uv project management](https://docs.astral.sh/uv/guides/projects/) and [installation](https://docs.astral.sh/uv/getting-started/installation/) for the upstream instructions.

Record that commit with the run. Keep the checkout unchanged while a run is active. Do not run git pull in the middle of it: source and dependency hashes are part of cache validation.

## 2. Attach the supplied data

Raw data is transferred separately and is not in Git. Set DATASET to the directory containing train/, not to train/ itself. For example:

```text
/mnt/data/records/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
```

Each source TSV has these columns, in this order:

```text
entity_id<TAB>business_name<TAB>business_address<TAB>country
```

The label TSV has source1_entity_id and matched_entity_ids, separated by a tab; matched IDs are comma-separated. Preserve empty fields and IDs exactly. The supplied training corpus has 2,206,821 reference rows, 10,320,219 target rows and 7,638,365 true links.

Test files are not required for training. To prepare them in the same snapshot, supply test/test_source1.tsv through test_source3.tsv and add --include-test from the first run onward. Changing that option later requires a new work root. The default omits test preparation to get training started sooner.

## 3. Start training

Replace both /mnt paths with real server paths. Use persistent SSD storage for WORK, with enough free space. A single work root belongs to one fixed code/data/configuration snapshot.

```bash
DATASET=/mnt/data/records
WORK=/mnt/record-runs/p3-v1

uv run --locked python -m plan3.server --dataset "$DATASET" --work-root "$WORK" --profile full --threads 16 --check-only
uv run --locked python -m plan3.server --dataset "$DATASET" --work-root "$WORK" --profile full --threads 16 --stop-after train
```

Run the second command inside your existing job scheduler allocation or a persistent terminal such as tmux. Request one CPU node; this release does not distribute one job across machines. The runner streams progress, saves a log per stage, stops on the first failed stage and records the active stage in server_status.json. Changing --threads or --max-rounds within an existing run changes its contract; keep the same values when resuming.

The run directory is:

```text
$WORK/runs/full-w100-m25/
```

For a smaller server, replace --profile full with --profile pilot in every command. A pilot uses 10,000 Fit, 3,000 support and 2,000 owners in each evaluation panel; it still searches the full target corpus. Its run directory ends in pilot-w100-m25.

## 4. Check progress and resume

```bash
cat "$WORK/runs/full-w100-m25/server_status.json"
tail -f "$WORK/runs/full-w100-m25/server_logs/retrieve.log"
```

Use the log matching the current stage. Retrieve and features can take much longer than the final boosting fit. There is no measured full-server runtime estimate yet.

After an interruption, reissue the original command with the same configuration. Completed, compatible artifacts are reused. An interrupted channel search restarts that channel; completed feature shards are retained. A boosting fit interrupted before its final model is saved restarts the fit. To start directly at a known unfinished stage:

```bash
uv run --locked python -m plan3.server --dataset "$DATASET" --work-root "$WORK" --profile full --threads 16 --start-at train --stop-after train
```

For an out-of-memory exit, keep the logs and report the stage and machine RAM; do not remove targets or change labels. For a contract mismatch, use a new work root for changed code/data/settings rather than editing hashes. A force-killed runner may leave server_running.lock in the run directory. Verify that its process and child training process have stopped before removing that one lock file. Never launch two jobs writing the same run directory.

## 5. Optional sibling-support comparison

After direct training succeeds, the same prepared data and features can support the second model:

```bash
uv run --locked python -m plan3.server --dataset "$DATASET" --work-root "$WORK" --profile full --threads 16 --start-at support --stop-after support
```

It adds independently scored sibling and cross-source evidence. Its training owners come from the previously unused c_prob pool and were excluded from the direct model and learned transliteration dictionary. This pool is not also used for probability calibration. Predictions are evidence, never replacement truth labels. The runner keeps the direct model and does not automatically promote the support model. The local pilot improves development macro F0.5 from 0.951625 to 0.960921; repeat this comparison on the larger run.

## Modal CPU execution

The project includes an optional compute dependency and a persistent Modal job. Choose your own authenticated Modal profile explicitly. The app requests 16 physical CPU cores and 128 GiB RAM, with a 192 GiB hard limit, one active container and a 24-hour timeout. It uses a persistent falcon-record-runs volume for data, logs and checkpoints. It runs the full direct model, sibling model and retraining control, then packages their reports. Fresh Audit and final inference are separate stages. Keep an active job on its recorded code revision; use a separate checkout for updates.

```bash
uv sync --locked --extra compute
export MODAL_PROFILE=your-profile
uv run --locked --extra compute python -m plan3.modal_client upload --dataset /mnt/data/records
uv run --locked --extra compute modal deploy -m plan3.modal_app
uv run --locked --extra compute python -m plan3.modal_client submit --dataset-id DATASET_ID_FROM_UPLOAD --run-id full-rich-v1
uv run --locked --extra compute python -m plan3.modal_client status --job work/modal/full-rich-v1.json
uv run --locked --extra compute python -m plan3.modal_client download --job work/modal/full-rich-v1.json --out work/modal/full-rich-v1-results.zip
```

After the parent preparation stage completes, test views and indexes can run independently while training continues:

```bash
uv run --locked --extra compute modal deploy -m plan3.modal_followup
uv run --locked --extra compute python -m plan3.modal_client prepare-test --job work/modal/full-rich-v1.json
uv run --locked --extra compute python -m plan3.modal_client status --job work/modal/full-rich-v1-test-assets.json
```

This second job uses 8 CPU cores, 32 GiB reserved RAM and a 64 GiB limit. It writes only test assets under the same persistent work root. It does not choose a model or evaluate labels.

Queue the remaining stages while the full training run is active. Supply the exact original validator file; it is hash-checked and uploaded only to the private volume.

```bash
uv run --locked --extra compute python -m plan3.modal_client finish --job work/modal/full-rich-v1.json --test-assets-job work/modal/full-rich-v1-test-assets.json --validator /path/to/supplied/utils/validate_submission.py
uv run --locked --extra compute python -m plan3.modal_client status --job work/modal/full-rich-v1-finish.json
uv run --locked --extra compute python -m plan3.modal_client download --job work/modal/full-rich-v1-finish.json --out work/modal/full-rich-v1-outputs.zip
```

The coordinator waits for successful training and test preparation, then submits a separate 16-core worker with its own 24-hour timeout. Status follows that worker; completion of the waiting job alone is not completion of the outputs. Failed dependencies stop the queue. Do not redeploy the followup app while it has active work.

The support model is retained only if Select macro F0.5 improves, development paired-owner confidence intervals have positive lower bounds against both direct and the retraining control, and neither development country regresses. Otherwise direct is retained. This rule is fixed before reading fresh Audit. No threshold is retuned using Audit.

The finishing worker scores every eligible held-out training owner, excluding Fit/dictionary owners and, for support, its second-stage training pool. Ownership is resolved across that entire declared world, and accuracy is reported on the reserved 50,000-owner Audit panel. It then scores every test owner, resolves test ownership globally, assembles both TSV files and runs the supplied validator with --check-ids. The private output ZIP contains the files, validation metadata, model-choice evidence and Audit report; its download verifies the recorded archive hash. Failed export attempts stay in separate temporary directories. Compatible inference checkpoints resume; changed committed files are rejected.

The Modal worker serializes finishing jobs. Local CLI workers must not run on different hosts against the same work root. Same-host finishing uses an OS lock that releases on process termination, so an interrupted finishing process does not require deletion of a persistent lock file.

The archive contains exactly the seven supplied TSV files and their hashes. It is extracted into a private project volume, not placed in Git. Do not submit another job with the same run ID while it is running. See the official [Modal volume](https://modal.com/docs/guide/volumes) and [CPU resource](https://modal.com/docs/guide/resources) documentation for the storage and resource semantics.

## 6. Send back the result bundle

```bash
uv run --locked python -m plan3.collect --run "$WORK/runs/full-w100-m25" --out "$WORK/full-v1-results.zip"
```

Send the ZIP privately to the team with the commit SHA and machine CPU/RAM. It includes model files, run metadata, metrics, candidate budget table, threshold sweeps, error ledgers and stage logs. It excludes the large raw/feature/candidate caches. It can also package an incomplete run after a failure. Use a different ZIP filename for a second collection; existing bundles are not overwritten. Error ledgers contain record IDs and some retrieved-miss text, so do not commit the bundle to the public repository. Keep the full work directory on the server for later analysis.

The useful checkpoints are direct_report.json, direct_meta.json, direct_decision.json, budget_grid.tsv, direct_tune_errors.parquet and, if run, support_report.json and seed_selection.json. Reports separate retrieval misses, rejected true candidates, false accepts and true links lost during ownership resolution.

## What these results establish

The direct-model pilot development score was 0.951625 on 2,000 owners. Its candidate recall on Tune was 99.306% with 237.6 mean candidates per owner; it did not meet the 99.5% target. This is a development experiment, not an external score or a fresh-Audit result. The supplied reference score of 0.9388 is accepted without reproducing that run.

The next server run expands training and evaluation of the new direct matcher. It does not yet establish full-population ownership, missing-owner robustness or unseen-country accuracy. A doubled unowned-negative weighting ablation worsened the pilot and was rejected. See PILOT_RESULTS.md and PLAN.md for the measured evidence and remaining experiments.

## Frozen-model inference, evaluation and output

Use these commands after selecting the model from development evidence. The inference contract freezes model hashes, the saved Select threshold, proposal budgets, retained candidate cuts and preprocessing. Current choices are direct and support. Inference deliberately queries the original W200/M100 proposals before keeping the trained candidate budget, because channel ranks and presence beyond that final cut are part of the trained features.

```bash
RUN="$WORK/runs/full-w100-m25"
uv run --locked python -m plan3.inference --prepared "$WORK/prepared" --views "$WORK/views" --indexes "$WORK/indexes" --model-run "$RUN" --out "$WORK/fresh-audit" --model support --split train --queries "$WORK/prepared/fresh_audit.parquet" --threads 16
uv run --locked python -m plan3.audit --prepared "$WORK/prepared" --inference "$WORK/fresh-audit" --out "$WORK/fresh-audit/report" --panel fresh-audit
uv run --locked python -m plan3 views --prepared "$WORK/prepared" --out "$WORK/views" --split test
uv run --locked python -m plan3.inference --prepared "$WORK/prepared" --views "$WORK/views" --indexes "$WORK/indexes" --model-run "$RUN" --out "$WORK/test-inference" --model support --split test --threads 16
uv run --locked python -m plan3.export --inference "$WORK/test-inference" --test-dir "$DATASET/test" --out "$WORK/output"
```

The Audit evaluator reports the declared claimant population and forbids Fit/dictionary or support-training owners as held-out rivals. Scoring only the reserved Audit owners gives ownership across that panel, not a full-test ownership estimate. The first fresh-Audit evaluation records the frozen configuration and refuses a different model/decision on that same reserved population. Development evaluation uses --panel development and excludes reserved Audit owners.

Output includes every raw test S1 in original order, including empty candidates and matches. Ownership is resolved globally across score shards. Inputs, scores and output bytes are authenticated, and altered committed checkpoints fail. The exporter validates target existence, countries, duplicate pairs, probability ranges and exact candidate membership. Run the supplied validator separately on the completed outputs with its --check-ids option. Fixture validation and the 20-owner inference consistency check do not establish full-data output success or fresh-Audit accuracy.

### 64-core continuation

For the stopped full run with all eight retrieval searches already saved, the fast app reserves 64 physical CPU cores and 192 GiB RAM, with a 256 GiB memory limit. It reuses preparation, transliteration, indexes and channel searches. It requires the original training call and finishing queue to be terminated before it can write to that run. It records a separate accelerated execution contract rather than changing the old contracts.

```bash
uv run --locked --extra compute modal deploy -m plan3.modal_fast
uv run --locked --extra compute python -m plan3.modal_client accelerate --job work/modal/full-rich-v1.json --finish-job work/modal/full-rich-v1-finish.json
uv run --locked --extra compute python -m plan3.modal_client status --job work/modal/full-rich-v1-fast.json
uv run --locked --extra compute python -m plan3.modal_client download --job work/modal/full-rich-v1-fast.json --out work/modal/full-rich-v1-fast-outputs.zip
```

Feature construction uses 64 independent processes with one native thread each and a bounded work queue. A complete owner's candidates always stay together, preserving relative features and sibling evidence. Model fitting uses 64 threads. After the comparisons succeed, the job starts a separate 64-core finishing worker for the fixed selection rule, protected Audit, test inference and validation. Each worker has its own 24-hour timeout. No speedup or final score is assumed before measurement.
