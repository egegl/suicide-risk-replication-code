# R9 organizer inference and replication code

The adopted R9 system is a two-seed, real-only Gemma-4-31B-IT LoRA ensemble. For organizer evaluation on unseen examples, use `predict.py`; synthetic-data generation and cross-validation are not required.

## Run R9 on unseen examples

Use Python 3.11 or newer on Linux with two CUDA GPUs supported by vLLM. Create the environment and point `GEMMA_MODEL` to the locally downloaded Gemma-4-31B-IT snapshot:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-gpu.txt
export GEMMA_MODEL=/path/to/gemma-4-31B-IT
```

Place the two separately supplied LoRA adapters as described in [`adapters/README.md`](adapters/README.md). The required members are:

- `adapters/seed1-step160`
- `adapters/seed2-step120`

If the adapters are not supplied separately, they can be rebuilt from the organizer training files after completing the development setup below:

```bash
.venv/bin/python run_experiment.py specs
.venv/bin/python run_experiment.py preflight --arms real_only
.venv/bin/python run_experiment.py train --run-id real_only-full-s1
.venv/bin/python run_experiment.py train --run-id real_only-full-s2
```

Then copy `runs/real_only-full-s1/adapters/step160` and `runs/real_only-full-s2/adapters/step120` to the two adapter locations above.

The unseen input may be `.xlsx` or `.csv` and must contain exactly these columns:

```text
row_id, anon_user_id, post_id, post
```

Within each user, `post_id` must be consecutive from zero. Run:

```bash
.venv/bin/python predict.py \
  --input organizer_hidden.xlsx \
  --output predictions.csv
```

The command builds the same limited timeline context used for R9, runs both adapters with the registered retry/fallback behavior, merges their risk and factor probabilities, applies the frozen R9 bagged thresholds in [`artifacts/r9_thresholds.json`](artifacts/r9_thresholds.json), aligns evidence to verbatim substrings, and writes validated columns `row_id,risk_level,evidence,factors`.

Public-leaderboard-specific known-row overrides are deliberately not applied to unseen examples.

For a single-GPU vLLM configuration, pass `--tensor-parallel-size 1` if the local device has enough memory. The R9 serving configuration uses two GPUs by default.

## Development replication setup

Native Linux and Windows support data preparation, scoring, and tests. GPU training uses vLLM, so Windows users need WSL2.

Linux or WSL2:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Place the organizer files at `data/raw/train.xlsx` and `data/raw/leaderboard.xlsx`, then run:

```bash
python build_dataset.py
python -m pytest -q
```

## Synthetic training files (not needed for R9 inference)

Generation uses the OpenAI Batch API and costs real money!

```bash
python generate_synthetic.py bundles
python generate_synthetic.py requests --preset mini
python generate_synthetic.py submit --backend openai --yes
python generate_synthetic.py poll --watch
python generate_synthetic.py fetch
python generate_synthetic.py run --until emit
python generate_synthetic.py check-realism
python generate_synthetic.py assemble
```

This creates the `synthetic` and size-matched `repeated_real` training arms.

## Training and local scoring (not needed when adapters are supplied)

Install the GPU dependencies in the Linux or WSL2 environment and point `GEMMA_MODEL` to the Gemma-4-31B-IT snapshot.

```bash
python -m pip install -r requirements-gpu.txt
export GEMMA_MODEL=/path/to/gemma-4-31B-IT
python run_experiment.py specs
python run_experiment.py preflight
python run_experiment.py indices --stage train
```

Run `python run_experiment.py train --index N` for every printed index. Then run:

```bash
python run_experiment.py eval --arm real_only
python run_experiment.py eval --arm synthetic
python run_experiment.py eval --arm repeated_real
```

For each arm and each seed in `{1, 2}`, run:

```bash
python run_experiment.py pool --arm ARM --seed SEED
python run_experiment.py thresholds --arm ARM --seed SEED
```

Finally:

```bash
python run_experiment.py compare
python run_experiment.py utility
python final_ensemble.py
```

The main outputs are `runs/sweep_table.json`, `runs/utility_report.json`, and `runs/reanalysis/final_ensemble.json`. Compare them with `expected_results.json`.
