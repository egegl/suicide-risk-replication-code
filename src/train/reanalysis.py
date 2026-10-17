"""Checkpoint-rule variants: pool each fold at the final step (fixed160) or at the step chosen on the other folds (nested)."""
import json
import re
from pathlib import Path
import numpy as np
import scorer
from synth import state
from . import constants as C
from . import oof, thresholds

VARIANTS = ('fixed160', 'nested')

FIXED_STEP = C.TOTAL_STEPS

assert FIXED_STEP in C.CHECKPOINT_STEPS

def _run_id(arm: str, fold: int, seed: int) -> str:
    return f'{arm}-f{fold}-s{seed}'

def load_metrics(runs_dir: Path, arm: str, seed: int) -> dict[int, dict[int, dict]]:
    runs_dir = Path(runs_dir)
    pat = re.compile(f'^{re.escape(arm)}-f(\\d+)-s{seed}$')
    folds = sorted((int(m.group(1)) for d in runs_dir.iterdir() if (m := pat.match(d.name))))
    assert folds, f'no {arm}-f*-s{seed} run dirs under {runs_dir}'
    assert folds == list(range(len(folds))), f'{arm}-s{seed} fold dirs are not contiguous 0..n-1: {folds}'
    out: dict[int, dict[int, dict]] = {}
    for k in folds:
        run_id = _run_id(arm, k, seed)
        rows = state.read_jsonl(runs_dir / run_id / 'eval' / 'metrics.jsonl')
        by_step: dict[int, dict] = {}
        for r in rows:
            assert r['run_id'] == run_id, (run_id, r['run_id'])
            by_step[r['step']] = r
        missing = [s for s in C.CHECKPOINT_STEPS if s not in by_step]
        assert not missing, f'{run_id} metrics.jsonl misses steps {missing}'
        out[k] = by_step
    return out

def nested_steps(metrics_by_fold: dict[int, dict[int, dict]], steps=C.CHECKPOINT_STEPS) -> dict[int, int]:
    folds = sorted(metrics_by_fold)
    out: dict[int, int] = {}
    for k in folds:
        others = [j for j in folds if j != k]
        assert others, f'nested_steps needs >=2 folds (got {folds})'
        best_step, best_val = (None, float('-inf'))
        for s in sorted(steps):
            v = float(np.mean([metrics_by_fold[j][s]['composite_ee1'] for j in others]))
            if v > best_val:
                best_step, best_val = (s, v)
        out[k] = best_step
    return out

def variant_steps(metrics_by_fold: dict[int, dict[int, dict]], variant: str) -> dict[int, int]:
    assert variant in VARIANTS, f'unknown variant {variant!r} — one of {VARIANTS}'
    if variant == 'fixed160':
        return {k: FIXED_STEP for k in metrics_by_fold}
    return nested_steps(metrics_by_fold)

def selected_info(runs_dir: Path, run_id: str) -> dict:
    path = Path(runs_dir) / run_id / 'selected.json'
    return json.loads(path.read_text(encoding='utf-8'))

def resolve_step_source(runs_dir: Path, run_id: str, step: int) -> tuple[str, Path]:
    sel = selected_info(runs_dir, run_id)
    run = Path(runs_dir) / run_id
    if step == sel['step']:
        return ('oof', run / 'oof.jsonl')
    if sel.get('skip_step') is not None and step == sel['skip_step']:
        return ('oof_skip', run / 'oof_skip.jsonl')
    return ('preds', run / 'eval' / f'step{step}' / 'preds.jsonl')

def fold_rows_at_step(runs_dir: Path, run_id: str, step: int) -> list[dict]:
    kind, path = resolve_step_source(runs_dir, run_id, step)
    if not path.exists():
        if kind == 'preds':
            raise FileNotFoundError(f'{path} is missing; run this analysis where the evaluation outputs are stored')
        raise FileNotFoundError(f'{path} is missing — committed {kind} artifact expected for {run_id} step {step}')
    rows = state.read_jsonl(path)
    if kind == 'preds':
        rows = [{k: v for k, v in r.items() if k != 'text'} for r in rows]
    for r in rows:
        assert r['step'] == step, (str(path), r['row_id'], r['step'], step)
    return rows

def _score_pooled(rows: list[dict], repo_root: Path) -> dict:
    processed = Path(repo_root) / 'data' / 'processed'
    preds = {r['row_id']: {'risk': r['risk'], 'spans': list(r['spans']), 'factors': list(r['factors'])} for r in rows}
    full = oof.propagate(preds, processed)
    gold = scorer.load_gold(processed / 'gold.jsonl')
    ee1 = scorer.score(full, gold, empty_convention='ee1')
    skip = scorer.score(full, gold, empty_convention='skip')
    return {'composite_ee1': ee1['composite'], 'composite_skip': skip['composite'], 'risk_wf1': ee1['risk_wf1'], 'phrase_f1_ee1': ee1['phrase_f1'], 'phrase_f1_skip': skip['phrase_f1'], 'macro_f1': ee1['macro_f1']}

def repool_variant(repo_root: Path, runs_dir: Path, arm: str, seed: int, variant: str) -> dict:
    repo_root, runs_dir = (Path(repo_root), Path(runs_dir))
    processed = repo_root / 'data' / 'processed'
    steps_by_fold = variant_steps(load_metrics(runs_dir, arm, seed), variant)
    rows: list[dict] = []
    for k in sorted(steps_by_fold):
        fold_rows = fold_rows_at_step(runs_dir, _run_id(arm, k, seed), steps_by_fold[k])
        for r in fold_rows:
            assert r['fold'] == k, (_run_id(arm, k, seed), r['row_id'], r['fold'])
        rows.extend(fold_rows)
    sft_ids = {r['row_id'] for r in state.read_jsonl(repo_root / C.SFT_DIR / 'full_train.jsonl')}
    ids = [r['row_id'] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1}) if len(ids) != len(set(ids)) else []
    assert not dupes, f"rows appear in multiple folds' files, e.g. {dupes[:5]}"
    assert set(ids) == sft_ids, f'pooled {variant} OOF covers {len(set(ids))} rows, expected the {len(sft_ids)} SFT val rows (missing e.g. {sorted(sft_ids - set(ids))[:5]}, extra e.g. {sorted(set(ids) - sft_ids)[:5]})'
    rows.sort(key=lambda r: (r['fold'], r['row_id']))
    base = runs_dir / 'pooled' / f'{arm}-s{seed}.{variant}'
    state.atomic_write_jsonl(Path(f'{base}.oof.jsonl'), rows)
    gold = scorer.load_gold(processed / 'gold.jsonl')
    thr_rows, taus_by_fold = thresholds.nested_thresholded_pool(rows, gold)
    state.atomic_write_jsonl(Path(f'{base}.thresholded.jsonl'), thr_rows)
    state.atomic_write_json(Path(f'{base}.taus.json'), {str(k): v for k, v in taus_by_fold.items()})
    return {'arm': arm, 'seed': seed, 'variant': variant, 'steps_by_fold': {str(k): v for k, v in sorted(steps_by_fold.items())}, 'raw': _score_pooled(rows, repo_root), 'thresholded': _score_pooled(thr_rows, repo_root)}