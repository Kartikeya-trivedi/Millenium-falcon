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

It adds independently scored sibling and cross-source evidence. Its training owners come from the previously unused c_prob pool and were excluded from the direct model and learned transliteration dictionary. This pool is not also used for probability calibration. Predictions are evidence, never replacement truth labels. The runner keeps the direct model and does not automatically promote the support model. Its local feature generation completed, but its final model comparison is still pending; the direct run is the immediate handoff priority.

## 6. Send back the result bundle

```bash
uv run --locked python -m plan3.collect --run "$WORK/runs/full-w100-m25" --out "$WORK/full-v1-results.zip"
```

Send the ZIP privately to the team with the commit SHA and machine CPU/RAM. It includes model files, run metadata, metrics, candidate budget table, threshold sweeps, error ledgers and stage logs. It excludes the large raw/feature/candidate caches. It can also package an incomplete run after a failure. Use a different ZIP filename for a second collection; existing bundles are not overwritten. Error ledgers contain record IDs and some retrieved-miss text, so do not commit the bundle to the public repository. Keep the full work directory on the server for later analysis.

The useful checkpoints are direct_report.json, direct_meta.json, direct_decision.json, budget_grid.tsv, direct_tune_errors.parquet and, if run, support_report.json and seed_selection.json. Reports separate retrieval misses, rejected true candidates, false accepts and true links lost during ownership resolution.

## What these results establish

The direct-model pilot development score was 0.951625 on 2,000 owners. Its candidate recall on Tune was 99.306% with 237.6 mean candidates per owner; it did not meet the 99.5% target. This is a development experiment, not an external score or a fresh-Audit result. The supplied reference score of 0.9388 is accepted without reproducing that run.

The next server run expands training and evaluation of the new direct matcher. It does not yet establish full-population ownership, the test orphan-rate stress ablation, unseen-country accuracy or final inference/export readiness. Fresh Audit evaluation and final output generation are later, separate steps. See PILOT_RESULTS.md and PLAN.md for the measured evidence and remaining experiments.
