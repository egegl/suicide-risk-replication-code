# Evaluating LLM-Generated Training Data for Rare Psychosocial Factors in Suicide-Risk Detection

Code, aggregate evaluation outputs, and the frozen numbers for the ICHI 2027 paper of this title, plus the competition system it builds on (IEEE BigData 2026 Cup, explainable suicide-risk detection).

The paper asks whether LLM-generated training posts help a fine-tuned Gemma-4-31B-IT model on six rare psychosocial factors once label quality is verified and the usual confounds are controlled. Inside every split of five-fold user-grouped cross-validation, Claude Sonnet 5.5 and Gemma-4-31B-IT each wrote 6,384 posts. Every post copied its full label set from a real training post and filled a gap in a real user's timeline. The posts passed an out-of-fold model check and a blinded three-annotator audit, yet adding them under a fixed training budget lowered the composite score by about 0.013 for both writers, below real-only training, size-matched repeated real data, and rare-factor loss weighting.

## What is here

| Path | Contents |
|---|---|
| `paper/figures/frozen.json` | Every number the paper and its supplement report (aggregates only) |
| `paper/figures/figures.py`, `fig1_design.tex` | Build every figure and table from `frozen.json` and check the manuscript's numbers |
| `paper/figures/freeze.py` | Recompute `frozen.json` from saved out-of-fold predictions |
| `paper/results/e2_confirm.json` | The registered confirmatory readout (primary contrasts, Holm correction, adoption rule) |
| `src/synth/pkgc.py` | Generated-data pipeline: fold-local cases, writer prompts, automated filters, training-file assembly, audit sampling and 2-of-3 consensus |
| `scripts/e2_data.py` | Command line for the CPU steps of `pkgc.py` |
| `scripts/pkgc_claude.py` | Claude writers on the Message Batches API (pilot and E2) |
| `scripts/pkgc_local.py` | Local writers and the fold labeler on vLLM (GPU) |
| `scripts/e2_controls.py` | Repeated-real and rare-factor-weighting training files, and the control run specs |
| `scripts/e2_confirm.py` | The registered E2 analysis |
| `audit/` | The blinded annotation tool and guidelines used for the human audit |
| `build_dataset.py`, `run_experiment.py`, `src/prep`, `src/train` | Data preparation, LoRA training, evaluation, pooling, nested thresholds |
| `predict.py`, `final_ensemble.py`, `adapters/`, `artifacts/` | The competition system (R9), see below |
| `legacy/generate_synthetic.py` | The competition-era GPT-5.4-mini generator. **Not the paper's method**, see below |

The writer prompt is `SYSTEM_PROMPTS["pkgc-v2"]` and `render_user_prompt` in `src/synth/pkgc.py` (supplement Fig. S1, also stored under `prompts` in `frozen.json`). The automated filters (paper Section IV-C) are `pkgc.screen`: empty output, refusal, over 700 words, an 8-word overlap with any real post, a 6-word overlap or a sequence ratio of at least 0.55 with the shown timeline posts, and a narrated "no plan" in Ideation posts.

**Not released.** Competition posts come from the organizers under their data agreement. Generated posts are not released (paper, Limitations and Ethics), so the writer outputs, assembled training files, audit packet, and annotations are absent. Row-level predictions are absent because they contain verbatim evidence phrases from the posts. LoRA adapters of the E2 runs are not released.

## Check the paper's numbers (no data needed)

Needs Python 3.11+ with matplotlib, plus `pdflatex` (with TikZ) and `pdftoppm` for Fig. 1.

```bash
python -m pip install -r requirements.txt
python paper/figures/figures.py --check
```

This builds every figure and table into `paper/figures/out/` twice and fails unless both builds are byte-identical. With the manuscript's `main.tex` and `supplement.tex` placed in `paper/`, it also checks every number the text cites against `frozen.json`.

## Reproduce the experiment

Every command runs from the repository root. GPU steps need Linux (or WSL2) with CUDA GPUs supported by vLLM and `requirements-gpu.txt`.

**1. Data.** Place the organizer files at `data/raw/train.xlsx` and `data/raw/leaderboard.xlsx`, then:

```bash
python build_dataset.py
python -m pytest -q
export GEMMA_MODEL=/path/to/gemma-4-31B-it
```

**2. Real-data conditions.** Real only (`baseline`), repeated real (`qcontrol`), and rare-factor weighting (`rareweight`); `build` reports whether each file matches the paper's byte for byte.

```bash
python scripts/e2_controls.py build
python scripts/e2_controls.py specs
python run_experiment.py --runs-dir runs31 train --index N         # N = 0..19
python run_experiment.py --runs-dir runs31-rare train --index N    # N = 0..9
python run_experiment.py --runs-dir runs31 eval --run-id RUN_ID    # every run id in runs31/specs.json
python run_experiment.py --runs-dir runs31-rare eval --run-id RUN_ID
```

**3. Generated data.** For each held-out fold `K` in 0..4, the cases and prompts come only from the other four folds. Sonnet costs about $15 on the Batch API (`ANTHROPIC_API_KEY`); Gemma runs locally from `$MODELS_DIR/gemma-4-31B-it`.

```bash
python scripts/e2_data.py prepare --fold K
PYTHONPATH=src python scripts/pkgc_claude.py submit-full     # once, after all five folds are prepared
PYTHONPATH=src python scripts/pkgc_claude.py collect-full    # until the batch has ended
PYTHONPATH=src python scripts/pkgc_local.py --fold K generate --arm gemma31
PYTHONPATH=src python scripts/pkgc_local.py --fold K first-draft --arm gemma31
PYTHONPATH=src python scripts/pkgc_local.py --fold K label --arms sonnet gemma31
```

Labeling uses the real-only run `baseline-fK-s1` from step 2: it supplies the evidence spans of the generated posts and the out-of-fold label check. Then assemble the training files (real rows plus exactly 1,211 to 1,321 generated rows per fold) and train the 20 E2 runs:

```bash
python scripts/e2_data.py assemble --arm sonnet
python scripts/e2_data.py assemble --arm gemma31
python scripts/e2_data.py specs
python run_experiment.py --runs-dir runs31-pkgc train --index N    # N = 0..19
python run_experiment.py --runs-dir runs31-pkgc eval --run-id RUN_ID
```

**4. Pool and threshold.** For each runs directory and arm (`runs31`: `baseline`, `qcontrol`; `runs31-rare`: `rareweight`; `runs31-pkgc`: `pkgc_sonnet`, `pkgc_gemma`) and seed 1 and 2:

```bash
python run_experiment.py --runs-dir RUNS pool --arm ARM --seed S
python run_experiment.py --runs-dir RUNS thresholds --arm ARM --seed S
python run_experiment.py --runs-dir RUNS repool --arm ARM --seed S --variant fixed160   # not for rareweight
python run_experiment.py --runs-dir RUNS repool --arm ARM --seed S --variant nested
```

**5. Human audit.** The registered sample is 4 cases per target factor, the same cases for both writers, drawn with seed 20261015. Annotators label blind in a local web tool; the consensus is computed before the key is opened.

```bash
python scripts/e2_data.py audit-packet
python audit/annotate.py serve --audit-dir data/synthetic_pkgc/audit/e2_v1 --annotator annotator1   # each of 3 annotators
python scripts/e2_data.py audit-consensus
python scripts/e2_data.py audit-report
```

Annotation files go to `data/synthetic_pkgc/audit/e2_v1/annotations/annotator{1,2,3}.json`.

**6. Writer pilot (supplement Table S3).** 300 fold-0 cases read by each candidate writer, with prompt v1 and v2:

```bash
python scripts/e2_data.py --version pkgc-v2 prepare
PYTHONPATH=src python scripts/pkgc_claude.py --version pkgc-v2 submit && PYTHONPATH=src python scripts/pkgc_claude.py --version pkgc-v2 collect
PYTHONPATH=src python scripts/pkgc_local.py --version pkgc-v2 generate --arm ARM      # qwen, hemmingway, gemma12, gemma31
PYTHONPATH=src python scripts/pkgc_local.py --version pkgc-v2 label
python scripts/e2_data.py --version pkgc-v2 report
```

Repeat with `--version pkgc-v1` (arms `haiku`, `sonnet`, `qwen`, `hemmingway`; `submit --arms haiku sonnet`).

**7. Analysis.**

```bash
python scripts/e2_confirm.py
python paper/figures/freeze.py --rebuilt
python paper/figures/figures.py --check
```

`freeze.py` stops on any difference from the paper's registered values unless `--rebuilt` is passed, in which case it lists the differences. Retrained runs are not bit-identical to the paper's: GPU kernels are nondeterministic, and the paper's own retrain of one control run differed by 0.008 on its fold, about the size of the seed spread. Everything before training is deterministic. Given the paper's own writer outputs and run predictions, this code reproduces exactly the processed data, the control and E2 training files, the pilot report, the audit sample and consensus, and every number in `e2_confirm.json` and `frozen.json`.

## Competition system (R9) on unseen examples

The adopted R9 system is a two-seed, real-only Gemma-4-31B-IT LoRA ensemble (0.7644 on the public leaderboard). It is the paper's real-only condition at full scale. Place the two separately supplied LoRA adapters as described in [`adapters/README.md`](adapters/README.md) (`adapters/seed1-step160`, `adapters/seed2-step120`), install `requirements.txt` and `requirements-gpu.txt`, set `GEMMA_MODEL`, and run:

```bash
python predict.py --input organizer_hidden.xlsx --output predictions.csv
```

The input may be `.xlsx` or `.csv` with columns `row_id, anon_user_id, post_id, post`; within each user, `post_id` must be consecutive from zero. The command builds the same limited timeline context used for R9, runs both adapters with the registered retry and fallback behavior, merges their risk and factor probabilities, applies the frozen bagged thresholds in [`artifacts/r9_thresholds.json`](artifacts/r9_thresholds.json), aligns evidence to verbatim substrings, and writes `row_id,risk_level,evidence,factors`. Leaderboard-specific overrides are not applied. Pass `--tensor-parallel-size 1` for a single GPU with enough memory.

To rebuild the adapters from the organizer training files:

```bash
python run_experiment.py specs
python run_experiment.py train --run-id real_only-full-s1
python run_experiment.py train --run-id real_only-full-s2
```

Then copy `runs/real_only-full-s1/adapters/step160` and `runs/real_only-full-s2/adapters/step120` to the two adapter locations. Rebuilt adapters match the reported system in every registered setting but are not bit-identical: the training order is seeded from the run id, which was `baseline-full-s{1,2}` in the original runs. Use the supplied adapters to reproduce the leaderboard figure itself. The competition's grouped cross-validation sweep (`run_experiment.py` arms `real_only`, `synthetic`, `repeated_real`, with outputs compared against `expected_results.json`) is documented in `run_experiment.py --help`.

## Competition-era synthetic pipeline (not the paper's method)

`legacy/generate_synthetic.py` (with `src/synth/` modules `assemble`, `batch`, `bundles`, `emit_synth`, `exemplars`, `noise`, `parse`, `pipeline`, `prompts`, `validate_synth`) generated whole synthetic timelines with GPT-5.4-mini through the OpenAI Batch API during the competition. It built the `synthetic` arm of the competition sweep, which lowered the composite by 0.016. The paper's generated data come from `src/synth/pkgc.py` instead: different writers, prompt, case construction, filters, and evidence labeling. The legacy pipeline costs real money and is kept only to document the competition sweep:

```bash
python legacy/generate_synthetic.py bundles
python legacy/generate_synthetic.py requests --preset mini
python legacy/generate_synthetic.py submit --backend openai --yes
python legacy/generate_synthetic.py poll --watch
python legacy/generate_synthetic.py fetch
python legacy/generate_synthetic.py run --until emit
python legacy/generate_synthetic.py check-realism
python legacy/generate_synthetic.py assemble
```
