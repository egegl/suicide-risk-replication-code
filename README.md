# Evaluating LLM-Generated Training Data for Rare Psychosocial Factors in Suicide-Risk Detection

Replication code for the ICHI 2027 paper of this title. Within each split of five-fold user-grouped cross-validation, Claude Sonnet 5.5 and Gemma-4-31B-IT each wrote 6,384 training posts for six rare factors of the IEEE BigData 2026 Cup suicide-risk task. Adding them to a fine-tuned Gemma-4-31B-IT model lowered the composite score by about 0.013 for both writers, although the posts passed model and blinded human label checks.

`results/frozen.json` holds every number the paper and its supplement report. Competition posts come from the organizers under their data agreement. Generated posts, run predictions, and adapters are not released.

| Path | Contents |
|---|---|
| `build_dataset.py`, `src/prep` | Data preparation and the five user-grouped folds |
| `run_experiment.py`, `src/train` | LoRA training, evaluation, pooling, nested thresholds |
| `src/synth/pkgc.py` | Generated data: fold-local cases, writer prompt (`SYSTEM_PROMPTS["pkgc-v2"]`, `render_user_prompt`), filters (`screen`), assembly, audit sample and consensus |
| `scripts/e2_controls.py` | Repeated-real and rare-factor-weighting files and the control run specs |
| `scripts/e2_data.py` | CPU steps of `pkgc.py` |
| `scripts/pkgc_claude.py`, `scripts/pkgc_local.py` | Claude writers (Message Batches API); local writers and the fold labeler (vLLM) |
| `audit/` | Blinded annotation tool and guidelines |
| `scripts/e2_confirm.py`, `scripts/freeze.py` | The registered analysis and every reported number |

## Steps

Run from the repository root on Linux with CUDA GPUs supported by vLLM. Place the organizer files at `data/raw/train.xlsx` and `data/raw/leaderboard.xlsx`.

```bash
python -m pip install -r requirements.txt -r requirements-gpu.txt
export GEMMA_MODEL=/path/to/gemma-4-31B-it MODELS_DIR=/path/to/models
python build_dataset.py
```

**Real-data conditions** (real only `baseline`, repeated real `qcontrol`, rare-factor weighting `rareweight`). `build` reports whether each file matches the paper's.

```bash
python scripts/e2_controls.py build
python scripts/e2_controls.py specs
python run_experiment.py --runs-dir runs31 train --index N         # N = 0..19
python run_experiment.py --runs-dir runs31-rare train --index N    # N = 0..9
python run_experiment.py --runs-dir runs31 eval --run-id RUN_ID    # every run in runs31/specs.json, likewise runs31-rare
```

**Generated data**, for each held-out fold K in 0..4 (Sonnet costs about $15 and needs `ANTHROPIC_API_KEY`; submit after all five folds are prepared):

```bash
python scripts/e2_data.py prepare --fold K
PYTHONPATH=src python scripts/pkgc_claude.py submit-full
PYTHONPATH=src python scripts/pkgc_claude.py collect-full
PYTHONPATH=src python scripts/pkgc_local.py --fold K generate --arm gemma31
PYTHONPATH=src python scripts/pkgc_local.py --fold K first-draft --arm gemma31
PYTHONPATH=src python scripts/pkgc_local.py --fold K label --arms sonnet gemma31
python scripts/e2_data.py assemble --arm sonnet
python scripts/e2_data.py assemble --arm gemma31
python scripts/e2_data.py specs
python run_experiment.py --runs-dir runs31-pkgc train --index N    # N = 0..19
python run_experiment.py --runs-dir runs31-pkgc eval --run-id RUN_ID
```

**Pooling**, for each runs directory and arm (`runs31`: `baseline`, `qcontrol`; `runs31-rare`: `rareweight`; `runs31-pkgc`: `pkgc_sonnet`, `pkgc_gemma`) and seeds 1 and 2:

```bash
python run_experiment.py --runs-dir RUNS pool --arm ARM --seed S
python run_experiment.py --runs-dir RUNS thresholds --arm ARM --seed S
python run_experiment.py --runs-dir RUNS repool --arm ARM --seed S --variant fixed160
python run_experiment.py --runs-dir RUNS repool --arm ARM --seed S --variant nested     # not for rareweight, as in the paper
```

**Human audit** (three annotators, blind to the assigned labels and the writer):

```bash
python scripts/e2_data.py audit-packet
python audit/annotate.py serve --audit-dir data/synthetic_pkgc/audit/e2_v1 --annotator annotator1   # annotator1..3
python scripts/e2_data.py audit-consensus
python scripts/e2_data.py audit-report
```

**Writer pilot** (supplement Table S3), with `--version pkgc-v2` and again with `pkgc-v1`:

```bash
python scripts/e2_data.py --version pkgc-v2 prepare
PYTHONPATH=src python scripts/pkgc_claude.py --version pkgc-v2 submit       # pkgc-v1: --arms haiku sonnet
PYTHONPATH=src python scripts/pkgc_claude.py --version pkgc-v2 collect
PYTHONPATH=src python scripts/pkgc_local.py --version pkgc-v2 generate --arm ARM   # v2: qwen hemmingway gemma12 gemma31; v1: qwen hemmingway
PYTHONPATH=src python scripts/pkgc_local.py --version pkgc-v2 label
python scripts/e2_data.py --version pkgc-v2 report
```

**Analysis:**

```bash
python scripts/e2_confirm.py
python scripts/freeze.py --rebuilt
```

`freeze.py` rewrites `results/frozen.json`. Without `--rebuilt` it stops on any difference from the paper's values; with it, it lists them. Retrained runs are not bit-identical because GPU kernels are nondeterministic (the paper's retrain of one control run differed by 0.008 on its fold, about the seed spread). Everything before training is deterministic: given the paper's writer outputs and run predictions, this code reproduces exactly the processed data, every training file, the audit sample and consensus, and every number in `results/frozen.json`.
