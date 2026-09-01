#!/usr/bin/env python3
'Score the final two-seed ensemble.'
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import scorer
from synth import state
from train import constants as C
from train import ensemble, oof, stats, thresholds

def _users_by_row(processed: Path) -> dict:
    posts = pd.read_parquet(processed / 'posts.parquet')
    tr = posts[posts['split'] == 'train']
    return dict(zip(tr['row_id'], tr['anon_user_id']))

def _observed(rows: list[dict], gold: dict, processed: Path) -> dict:
    preds = {r['row_id']: {'risk': r['risk'], 'spans': list(r['spans']), 'factors': list(r['factors'])} for r in rows}
    full = oof.propagate(preds, processed)
    ee1 = scorer.score({i: full[i] for i in gold}, gold, empty_convention='ee1')
    return {'composite_ee1': ee1['composite'], 'risk_wf1': ee1['risk_wf1'], 'phrase_f1_ee1': ee1['phrase_f1'], 'macro_f1': ee1['macro_f1'], 's1': (0.4 * ee1['risk_wf1'] + 0.3 * ee1['phrase_f1']) / 0.7, 's2': ee1['macro_f1'], 'per_rare_class_f1': stats.rare_class_f1(full, gold)}

def _delta(reps: dict, a: str, b: str) -> dict:
    d = np.asarray(reps[a], dtype=float) - np.asarray(reps[b], dtype=float)
    return {'mean': float(d.mean()), 'se': float(d.std(ddof=1)), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'frac_positive': float((d > 0).mean())}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs-dir', default=C.RUNS_DIR)
    ap.add_argument('--arm', default='real_only')
    ap.add_argument('--b', type=int, default=C.BOOTSTRAP_B)
    ap.add_argument('--boot-seed', type=int, default=C.BOOTSTRAP_SEED)
    args = ap.parse_args()
    runs = ROOT / args.runs_dir
    processed = ROOT / 'data' / 'processed'
    gold = scorer.load_gold(processed / 'gold.jsonl')
    users = _users_by_row(processed)
    rows_s1 = state.read_jsonl(runs / 'pooled' / f'{args.arm}-s1.oof.jsonl')
    rows_s2 = state.read_jsonl(runs / 'pooled' / f'{args.arm}-s2.oof.jsonl')
    variants = {}
    print('fitting nested variants ...', file=sys.stderr)
    variants['s1_argmax'], _ = thresholds.nested_thresholded_pool(rows_s1, gold)
    variants['s2_argmax'], _ = thresholds.nested_thresholded_pool(rows_s2, gold)
    variants['s1_bagged'], taus_s1_bag = thresholds.nested_thresholded_pool(rows_s1, gold, estimator='bagged')
    variants['s2_bagged'], _ = thresholds.nested_thresholded_pool(rows_s2, gold, estimator='bagged')
    merged = ensemble.merge_rows(rows_s1, rows_s2)
    variants['ensemble'], _ = thresholds.nested_thresholded_pool(merged, gold, estimator='bagged')
    observed = {name: _observed(rows, gold, processed) for name, rows in variants.items()}
    print(f'paired bootstrap (B={args.b}, seed={args.boot_seed}) ...', file=sys.stderr)
    boot = stats.paired_bootstrap({'real_only': variants['s1_argmax'], **{k: v for k, v in variants.items() if k != 's1_argmax'}}, gold, users, processed=processed, b=args.b, seed=args.boot_seed)
    reps = {label: boot['labels'][label]['replicates'] for label in boot['labels']}
    reps['s1_argmax'] = reps.pop('real_only')
    deltas = {'ensemble_vs_seed1': _delta(reps, 'ensemble', 's1_argmax'), 'ensemble_vs_seed2': _delta(reps, 'ensemble', 's2_argmax'), 'bagged_seed1_vs_seed1': _delta(reps, 's1_bagged', 's1_argmax'), 'bagged_seed2_vs_seed2': _delta(reps, 's2_bagged', 's2_argmax'), 'ensemble_vs_bagged_seed1': _delta(reps, 'ensemble', 's1_bagged')}
    primary = deltas['ensemble_vs_seed1']
    adoption = {'floor': C.ENSEMBLE_MIN_GAIN, 'positive_fraction_required': C.ENSEMBLE_MIN_POSITIVE_FRACTION, 'point_ok': primary['mean'] >= C.ENSEMBLE_MIN_GAIN, 'both_seeds_ok': deltas['ensemble_vs_seed1']['mean'] >= 0.0 and deltas['ensemble_vs_seed2']['mean'] >= 0.0, 'positive_fraction_ok': primary['frac_positive'] >= C.ENSEMBLE_MIN_POSITIVE_FRACTION, 'bagging_per_seed': {'seed1': deltas['bagged_seed1_vs_seed1']['mean'], 'seed2': deltas['bagged_seed2_vs_seed2']['mean']}}
    adoption['adopt_ensemble'] = bool(adoption['point_ok'] and adoption['both_seeds_ok'] and adoption['positive_fraction_ok'])
    out = {'method': 'final two-seed ensemble', 'arm': args.arm, 'constants': {'bag_n_boot': C.THRESH_BAG_N_BOOT, 'bag_seed': C.THRESH_BAG_SEED, 'bootstrap_b': args.b, 'bootstrap_seed': args.boot_seed}, 'observed': observed, 'deltas': deltas, 'adoption': adoption, 'example_thresholds': {f: {str(k): round(t[f], 4) for k, t in taus_s1_bag.items()} for f in ('meaning in life', 'cognitive deficits')}}
    out_path = runs / 'reanalysis' / 'final_ensemble.json'
    state.atomic_write_json(out_path, out)
    brief = {'observed_composite': {k: round(v['composite_ee1'], 4) for k, v in observed.items()}, 'factor_macro_f1': {k: round(v['s2'], 4) for k, v in observed.items()}, 'ensemble_vs_seed1': {k: round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] for k, v in primary.items()}, 'adopt_ensemble': adoption['adopt_ensemble'], 'out': str(out_path)}
    print(json.dumps(brief, indent=2))
if __name__ == '__main__':
    main()
