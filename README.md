## Setup

Use Python 3.11 or newer. Native Linux and Windows support data preparation, scoring, and tests. GPU training uses vLLM, so Windows users need WSL2.

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

## Synthetic training files

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

## Training and scoring

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
