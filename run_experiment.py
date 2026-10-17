#!/usr/bin/env python3
"""Train, evaluate, pool, and threshold the grouped cross-validation runs listed in <runs-dir>/specs.json."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import scorer
from synth import state
from train import constants as C
from train import oof, specs as specs_mod, thresholds

def _pin_cwd(repo_root: str) -> str:
    os.chdir(Path(repo_root).resolve())
    return '.'

def _p(args) -> dict:
    root = Path(args.repo_root)
    return {'root': root, 'runs': root / args.runs_dir, 'processed': root / 'data' / 'processed'}

def _specs(P) -> list[dict]:
    path = P['runs'] / 'specs.json'
    if not path.exists():
        sys.exit('no runs/specs.json; run `python run_experiment.py specs` first')
    return specs_mod.load_specs(path)

def _print(stats_dict) -> None:
    print(json.dumps(stats_dict, indent=2, default=float))

def cmd_train(args, P):
    from train import generate, trainer
    sp = specs_mod.resolve(_specs(P), index=args.index, run_id=args.run_id)
    run_dir = P['runs'] / sp['run_id']
    m = state.Manifest(run_dir)
    config = trainer.train_config(sp, model_override=args.model, steps_override=args.steps, checkpoint_steps_override=args.checkpoint_steps)
    train_file = P['root'] / sp['train_file']
    inputs = [train_file] if train_file.exists() else []
    ckpt = sorted(args.checkpoint_steps or sp['checkpoint_steps'])
    adapters_present = generate.present_steps(run_dir / 'adapters') == ckpt
    if not args.force and adapters_present and m.unchanged('train', config, inputs):
        _print({sp['run_id']: 'train up to date (skipped; --force to rerun)'})
        return
    res = trainer.train_run(P['root'], sp, P['runs'], model_override=args.model, steps_override=args.steps, checkpoint_steps_override=args.checkpoint_steps)
    _print(res)

def cmd_eval(args, P):
    from train import generate
    sp_list = _specs(P)
    if args.run_id:
        todo = [specs_mod.resolve(sp_list, run_id=args.run_id)]
    else:
        todo = [s for s in sp_list if s['arm'] == args.arm and s['fold'] != 'full']
        if not todo:
            sys.exit(f'no sweep specs for arm {args.arm!r}')
    model = args.model or todo[0]['base_model']
    out = {}
    runnable, problems = ([], {})
    for sp in todo:
        run_dir = P['runs'] / sp['run_id']
        problem = generate.adapter_problem(run_dir, sp)
        if problem:
            problems[sp['run_id']] = problem
            continue
        steps = generate.present_steps(run_dir / 'adapters')
        if not args.force and state.Manifest(run_dir).unchanged('eval', generate.eval_config(sp, steps, args.model), generate.eval_inputs(P['root'], sp, run_dir / 'adapters', steps)):
            out[sp['run_id']] = 'eval up to date (skipped; --force to rerun)'
            continue
        runnable.append(sp)
    if problems:
        _print({'refused': problems, 'note': 'refusing before engine build — train these runs first (resubmit the train array; done runs skip)'})
        sys.exit(1)
    engine = None
    for sp in runnable:
        if engine is None:
            engine = generate.Engine(model, enable_guided=C.GUIDED_DECODING)
        out[sp['run_id']] = generate.eval_run(P['root'], sp, P['runs'], engine, model_override=args.model)
    _print(out)

def cmd_pool(args, P):
    res = oof.pool_runs(P['runs'], args.arm, args.seed, P['root'])
    _print(res)

def _load_pooled(P, arm: str, seed: int, thresholded: bool=False) -> list[dict]:
    suffix = 'thresholded' if thresholded else 'oof'
    path = P['runs'] / 'pooled' / f'{arm}-s{seed}.{suffix}.jsonl'
    if not path.exists():
        sys.exit(f"missing {path}; run `python run_experiment.py {('thresholds' if thresholded else 'pool')} --arm {arm} --seed {seed}`")
    return state.read_jsonl(path)

def _score_pooled(rows: list[dict], P) -> dict:
    preds = {r['row_id']: {'risk': r['risk'], 'spans': list(r['spans']), 'factors': list(r['factors'])} for r in rows}
    full = oof.propagate(preds, P['processed'])
    gold = scorer.load_gold(P['processed'] / 'gold.jsonl')
    ee1 = scorer.score(full, gold, empty_convention='ee1')
    skip = scorer.score(full, gold, empty_convention='skip')
    return {'composite_ee1': ee1['composite'], 'composite_skip': skip['composite'], 'risk_wf1': ee1['risk_wf1'], 'phrase_f1_ee1': ee1['phrase_f1'], 'phrase_f1_skip': skip['phrase_f1'], 'macro_f1': ee1['macro_f1']}

def cmd_thresholds(args, P):
    rows = _load_pooled(P, args.arm, args.seed)
    gold = scorer.load_gold(P['processed'] / 'gold.jsonl')
    est = getattr(args, 'tau_estimator', 'argmax')
    out_rows, taus_by_fold = thresholds.nested_thresholded_pool(rows, gold, estimator=est)
    infix = '' if est == 'argmax' else f'.{est}'
    base = P['runs'] / 'pooled' / f'{args.arm}-s{args.seed}{infix}'
    state.atomic_write_jsonl(Path(f'{base}.thresholded.jsonl'), out_rows)
    state.atomic_write_json(Path(f'{base}.taus.json'), {str(k): v for k, v in taus_by_fold.items()})
    _print({'arm': args.arm, 'seed': args.seed, 'tau_estimator': est, 'raw': _score_pooled(rows, P), 'thresholded': _score_pooled(out_rows, P)})

def cmd_repool(args, P):
    from train import reanalysis
    _print(reanalysis.repool_variant(P['root'], P['runs'], args.arm, args.seed, args.variant))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--runs-dir', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    target = train.add_mutually_exclusive_group(required=True)
    target.add_argument('--index', type=int)
    target.add_argument('--run-id')
    train.add_argument('--model')
    train.add_argument('--steps', type=int)
    train.add_argument('--checkpoint-steps', type=int, nargs='+')
    train.add_argument('--force', action='store_true')
    train.set_defaults(fn=cmd_train)
    evaluate = sub.add_parser('eval')
    which = evaluate.add_mutually_exclusive_group(required=True)
    which.add_argument('--arm')
    which.add_argument('--run-id')
    evaluate.add_argument('--model')
    evaluate.add_argument('--force', action='store_true')
    evaluate.set_defaults(fn=cmd_eval)
    pool = sub.add_parser('pool')
    pool.add_argument('--arm', required=True)
    pool.add_argument('--seed', type=int, choices=C.TRAIN_SEEDS, required=True)
    pool.set_defaults(fn=cmd_pool)
    threshold = sub.add_parser('thresholds')
    threshold.add_argument('--arm', required=True)
    threshold.add_argument('--seed', type=int, choices=C.TRAIN_SEEDS, required=True)
    threshold.set_defaults(fn=cmd_thresholds)
    repool = sub.add_parser('repool')
    repool.add_argument('--arm', required=True)
    repool.add_argument('--seed', type=int, choices=C.TRAIN_SEEDS, required=True)
    repool.add_argument('--variant', choices=('fixed160', 'nested'), required=True)
    repool.set_defaults(fn=cmd_repool)
    args = parser.parse_args()
    args.repo_root = _pin_cwd(args.repo_root)
    args.fn(args, _p(args))


if __name__ == '__main__':
    main()
