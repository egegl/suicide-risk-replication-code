#!/usr/bin/env python3
'Run grouped training and evaluation.'
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import scorer
from prep import constants as PC
from synth import state
from train import constants as C
from train import oof, specs as specs_mod, stats, thresholds

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

def _arms_or_launch_default(arms_csv: str, require_real_only: bool=False) -> list[str]:
    arms = [a for a in arms_csv.split(',') if a] if arms_csv else list(C.LAUNCH_ARMS)
    if not arms:
        sys.exit(f'--arms {arms_csv!r} selected no arms — valid: {list(C.ARMS)}')
    unknown = sorted(set(arms) - set(C.ARMS))
    if unknown:
        sys.exit(f'unknown arms {unknown} — valid: {list(C.ARMS)}')
    if require_real_only and 'real_only' not in arms:
        sys.exit(f"--arms {arms} must include 'real_only' — every delta and the ship decision are computed against it")
    return arms

def cmd_specs(args, P):
    built = specs_mod.write_specs(P['root'], P['runs'] / 'specs.json')
    pending = sorted({s['arm'] for s in built if s['train_file_sha'] == 'pending'})
    _print({'specs': len(built), 'sweep_runs': sum((1 for s in built if s['fold'] != 'full')), 'pending_arms': pending, 'note': 'pending arms need `generate_synthetic.py assemble` (realism-gated) before their runs can launch' if pending else ''})

def cmd_preflight(args, P):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit('preflight needs `transformers` (tokenizer-only, no torch/GPU) — run it on the login node in .venv-train')
    from train import data as tdata, probe
    arms = _arms_or_launch_default(args.arms)
    sp_list = [spec for spec in _specs(P) if spec['arm'] in arms]
    model = args.model or sp_list[0]['base_model']
    report: dict = {'model': model}
    failures: list[str] = []
    mp = Path(model)
    if mp.is_dir():
        for req in ('config.json', 'tokenizer_config.json'):
            if not (mp / req).is_file():
                failures.append(f'snapshot: missing {req}')
        idx = mp / 'model.safetensors.index.json'
        if idx.is_file():
            shards = sorted(set(json.loads(idx.read_text())['weight_map'].values()))
            missing = [s for s in shards if not (mp / s).is_file() or (mp / s).stat().st_size == 0]
            report['snapshot'] = {'shards_expected': len(shards), 'shards_missing': missing}
            if missing:
                failures.append(f'snapshot: {len(missing)} missing/empty shards (e.g. {missing[:3]})')
        elif (mp / 'model.safetensors').is_file():
            report['snapshot'] = {'shards_expected': 1, 'shards_missing': []}
        else:
            failures.append('snapshot: no model.safetensors(.index.json) — an HF cache ROOT is not loadable; point at the snapshots/<hash>/ dir instead')
    else:
        report['snapshot'] = 'skipped (model is not a local directory)'
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as err:
        failures.append(f'tokenizer load failed: {type(err).__name__}: {err}')
    template_ok = False
    if tok is not None:
        for name, fn in (('suffix_ids', lambda: tdata.suffix_ids(tok)), ('assert_template_prefix', lambda: tdata.assert_template_prefix(tok)), ('probe_self_test', lambda: probe.probe_self_test(tok))):
            try:
                fn()
                report[name] = 'pass'
            except Exception as err:
                first = str(err).splitlines()[0] if str(err) else type(err).__name__
                report[name] = f'FAIL: {first}'
                failures.append(f'{name}: {first}')
                report.setdefault('template_check_details', {})[name] = str(err)
        template_ok = report.get('suffix_ids') == 'pass' and report.get('assert_template_prefix') == 'pass'
    if template_ok:
        train_report: dict = {}
        for sp in sp_list:
            rel, sha = (sp['train_file'], sp['train_file_sha'])
            if rel in train_report:
                continue
            path = P['root'] / rel
            if sha == 'pending' or not path.exists():
                failures.append(f'{rel}: pending/missing — run `generate_synthetic.py assemble`, rebuild specs')
                train_report[rel] = 'pending/missing'
                continue
            if state.sha256_file(path) != sha:
                failures.append(f'{rel}: sha mismatch vs specs.json — rebuild specs')
                train_report[rel] = 'sha mismatch'
                continue
            over, mx, n_rows = ([], 0, 0)
            for rec in state.read_jsonl(path):
                n_rows += 1
                try:
                    n = len(tdata.encode_example(tok, rec)['input_ids'])
                except AssertionError:
                    n = len(tdata.build_prompt_ids(tok, rec['input'])) + len(tok.encode(rec['target'], add_special_tokens=False)) + len(tdata.suffix_ids(tok))
                    over.append(rec['row_id'])
                mx = max(mx, n)
            train_report[rel] = {'rows': n_rows, 'max_tokens': mx, 'limit': C.MAX_SEQ_LEN, 'over': over}
            if over:
                failures.append(f'{rel}: {len(over)} rows over MAX_SEQ_LEN={C.MAX_SEQ_LEN} (e.g. {over[:5]})')
        report['train_files'] = train_report
        budget = C.VLLM_MAX_MODEL_LEN - C.MAX_NEW_TOKENS
        prompt_report: dict = {}
        val_files = sorted({sp['val_file'] for sp in sp_list if sp['val_file']})
        for rel in [*val_files, f'{C.SFT_DIR}/leaderboard_infer.jsonl']:
            rows = state.read_jsonl(P['root'] / rel)
            lens = {r['row_id']: len(tdata.build_prompt_ids(tok, r['input'])) for r in rows}
            over = sorted((rid for rid, n in lens.items() if n > budget))
            prompt_report[rel] = {'rows': len(rows), 'max_prompt_tokens': max(lens.values()), 'budget': budget, 'over': over}
            if over:
                failures.append(f'{rel}: {len(over)} prompts over VLLM_MAX_MODEL_LEN-MAX_NEW_TOKENS={budget} (e.g. {over[:5]})')
        report['prompt_files'] = prompt_report
    elif tok is not None:
        report['train_files'] = report['prompt_files'] = 'skipped (template checks failed)'
    report['failures'] = failures
    report['pass'] = not failures
    _print(report)
    if failures:
        sys.exit(1)

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

def _users_by_row(P) -> dict:
    import pandas as pd
    posts = pd.read_parquet(P['processed'] / 'posts.parquet')
    tr = posts[posts['split'] == 'train']
    return dict(zip(tr['row_id'], tr['anon_user_id']))

def _arm_level(boot: dict, arm_seeds: dict[str, list[str]]) -> dict:
    import numpy as np
    reps = {arm: np.mean([np.asarray(boot['labels'][l]['replicates'], dtype=float) for l in labels], axis=0) for arm, labels in arm_seeds.items()}
    labels_out = {arm: {'observed': float(np.mean([boot['labels'][l]['observed'] for l in arm_seeds[arm]])), 'mean': float(r.mean()), 'se': float(r.std(ddof=1)), 'replicates': [float(x) for x in r]} for arm, r in reps.items()}
    deltas = {}
    for arm, r in reps.items():
        if arm == 'real_only':
            continue
        d = r - reps['real_only']
        deltas[arm] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1)), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return {**{k: boot[k] for k in ('unit', 'n_users', 'n_rows', 'b', 'seed', 'convention', 'metric') if k in boot}, 'labels': labels_out, 'deltas': deltas}

def cmd_compare(args, P):
    arms = _arms_or_launch_default(args.arms, require_real_only=True)
    gold = scorer.load_gold(P['processed'] / 'gold.jsonl')
    users = _users_by_row(P)
    per_label_rows, arm_seeds, arm_summaries = ({}, {}, {})
    for arm in arms:
        arm_seeds[arm] = []
        seeds_summary = {}
        for seed in C.TRAIN_SEEDS:
            rows = _load_pooled(P, arm, seed, thresholded=True)
            label = f'{arm}-s{seed}'
            per_label_rows[label] = rows
            arm_seeds[arm].append(label)
            seeds_summary[seed] = {'raw': _score_pooled(_load_pooled(P, arm, seed), P), 'thresholded': _score_pooled(rows, P)}
        arm_summaries[arm] = {'per_seed': seeds_summary}
    boot_labels = dict(per_label_rows)
    boot_labels['real_only'] = per_label_rows[f'real_only-s{C.TRAIN_SEEDS[0]}']
    boot = stats.paired_bootstrap(boot_labels, gold, users, processed=P['processed'], b=args.b, seed=C.BOOTSTRAP_SEED)
    arm_boot = _arm_level(boot, arm_seeds)
    decision = stats.one_se_ship(arm_boot)
    table = stats.sweep_table(arm_summaries, arm_boot, decision)
    state.atomic_write_json(P['runs'] / 'sweep_table.json', table)
    _write_sweep_md(P['runs'] / 'sweep_table.md', table)
    _print({'decision': decision, 'table': str(P['runs'] / 'sweep_table.json')})

def cmd_utility(args, P):
    arms = _arms_or_launch_default(args.arms, require_real_only=True)
    gold = scorer.load_gold(P['processed'] / 'gold.jsonl')
    users = _users_by_row(P)
    per_label_rows, arm_seeds, per_rare = ({}, {}, {})
    for arm in arms:
        arm_seeds[arm] = []
        rare_seeds = {}
        for seed in C.TRAIN_SEEDS:
            rows = _load_pooled(P, arm, seed, thresholded=True)
            label = f'{arm}-s{seed}'
            per_label_rows[label] = rows
            arm_seeds[arm].append(label)
            preds = {r['row_id']: {'risk': r['risk'], 'spans': list(r['spans']), 'factors': list(r['factors'])} for r in rows}
            full = oof.propagate(preds, P['processed'])
            rare_seeds[str(seed)] = stats.rare_class_f1(full, gold)
        per_rare[arm] = rare_seeds
    boot_labels = dict(per_label_rows)
    boot_labels['real_only'] = per_label_rows[f'real_only-s{C.TRAIN_SEEDS[0]}']
    comp = stats.paired_bootstrap(boot_labels, gold, users, processed=P['processed'], b=args.b, seed=C.BOOTSTRAP_SEED)
    rare = stats.paired_bootstrap(boot_labels, gold, users, processed=P['processed'], b=args.b, seed=C.BOOTSTRAP_SEED, metric='rare_macro')
    decision = stats.one_se_ship(_arm_level(comp, arm_seeds))
    gate = stats.utility_gate(_arm_level(rare, arm_seeds), decision)
    realism_note = None
    rep_path = Path(args.synth_dir) / 'realism_report.json'
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding='utf-8'))
        realism_note = {'auc_mean': rep.get('auc_mean'), 'threshold': rep.get('threshold'), 'role': 'descriptive'}
    report = {'gate': gate, 'per_rare_class_f1': per_rare, 'bootstrap': {k: comp[k] for k in ('unit', 'n_users', 'n_rows', 'b', 'seed', 'convention')}, 'realism': realism_note}
    state.atomic_write_json(P['runs'] / 'utility_report.json', report)
    _print({'utility_pass': gate['pass'], 'candidate': gate['candidate'], 'rare_delta_vs_repeated_real': gate['rare_delta_vs_repeated_real'], 'report': str(P['runs'] / 'utility_report.json')})

def cmd_repool(args, P):
    from train import reanalysis
    _print(reanalysis.repool_variant(P['root'], P['runs'], args.arm, args.seed, args.variant))

def cmd_compare_variant(args, P):
    from train import reanalysis
    arms = _arms_or_launch_default(args.arms, require_real_only=True)
    _print(reanalysis.compare_variant(P['root'], P['runs'], args.variant, arms=arms, b=args.b))

def cmd_seed_report(args, P):
    from train import reanalysis
    arms = _arms_or_launch_default(args.arms, require_real_only=True)
    _print(reanalysis.seed_report(P['root'], P['runs'], arms, list(C.TRAIN_SEEDS)))

def cmd_boot_sens(args, P):
    from train import reanalysis
    arms = _arms_or_launch_default(args.arms, require_real_only=True)
    _print(reanalysis.bootstrap_sensitivity(P['root'], P['runs'], arms, boot_seeds=tuple(args.boot_seeds), bs=tuple(args.bs)))

def _write_sweep_md(path: Path, table: dict) -> None:
    lines = ['# Grouped evaluation', '', '| arm | composite | alternate empty-span score | delta vs real_only (95% CI) | SE |', '|---|---|---|---|---|']
    for arm, e in table['arms'].items():
        seeds = e.get('per_seed', {})
        ee1 = '/'.join((f"{s['thresholded']['composite_ee1']:.4f}" for s in seeds.values()))
        skp = '/'.join((f"{s['thresholded']['composite_skip']:.4f}" for s in seeds.values()))
        d = e.get('delta_vs_real_only')
        dtxt = f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else '—'
        se = f"{e['bootstrap']['se']:.4f}" if 'bootstrap' in e else '—'
        lines.append(f'| {arm} | {ee1} | {skp} | {dtxt} | {se} |')
    dec = table['decision']
    lines += ['', f"Decision: candidate=`{dec.get('candidate')}` ship=`{dec.get('ship')}` (bar {table['ship_bar']}, convention {table['primary_convention']})", '']
    state.atomic_write_text(path, '\n'.join(lines))

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

def cmd_indices(args, paths):
    arms = _arms_or_launch_default(args.arms)
    specs = _specs(paths)
    if args.stage == 'train':
        for spec in specs:
            if spec['arm'] in arms and spec['fold'] != 'full':
                print(spec['index'])
    else:
        for arm in arms:
            print(arm)

def main():
    parser = argparse.ArgumentParser(description='Run the grouped training and evaluation.')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--runs-dir', default=C.RUNS_DIR)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('specs').set_defaults(fn=cmd_specs)
    preflight = sub.add_parser('preflight')
    preflight.add_argument('--model')
    preflight.add_argument('--arms', default='',
                           help='comma-separated arms to validate')
    preflight.set_defaults(fn=cmd_preflight)
    indices = sub.add_parser('indices')
    indices.add_argument('--stage', choices=('train', 'eval'), required=True)
    indices.add_argument('--arms', default='')
    indices.set_defaults(fn=cmd_indices)
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
    evaluate.add_argument('--arm', default='real_only')
    evaluate.add_argument('--run-id')
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
    threshold.add_argument('--tau-estimator', choices=('argmax', 'bagged'), default='argmax')
    threshold.set_defaults(fn=cmd_thresholds)
    compare = sub.add_parser('compare')
    compare.add_argument('--arms', default='')
    compare.add_argument('--b', type=int, default=C.BOOTSTRAP_B)
    compare.set_defaults(fn=cmd_compare)
    utility = sub.add_parser('utility')
    utility.add_argument('--arms', default='')
    utility.add_argument('--b', type=int, default=C.BOOTSTRAP_B)
    utility.add_argument('--synth-dir', default='data/synthetic')
    utility.set_defaults(fn=cmd_utility)
    repool = sub.add_parser('repool')
    repool.add_argument('--arm', required=True)
    repool.add_argument('--seed', type=int, choices=C.TRAIN_SEEDS, required=True)
    repool.add_argument('--variant', choices=('fixed160', 'nested'), required=True)
    repool.set_defaults(fn=cmd_repool)
    variant = sub.add_parser('compare-variant')
    variant.add_argument('--variant', choices=('fixed160', 'nested'), required=True)
    variant.add_argument('--arms', default='')
    variant.add_argument('--b', type=int, default=C.BOOTSTRAP_B)
    variant.set_defaults(fn=cmd_compare_variant)
    seed_report = sub.add_parser('seed-report')
    seed_report.add_argument('--arms', default='')
    seed_report.set_defaults(fn=cmd_seed_report)
    sensitivity = sub.add_parser('bootstrap-sensitivity')
    sensitivity.add_argument('--arms', default='')
    sensitivity.add_argument('--boot-seeds', type=int, nargs='+', default=[C.BOOTSTRAP_SEED, 1, 2, 3])
    sensitivity.add_argument('--bs', type=int, nargs='+', default=[2000, 10000])
    sensitivity.set_defaults(fn=cmd_boot_sens)
    args = parser.parse_args()
    args.repo_root = _pin_cwd(args.repo_root)
    args.fn(args, _p(args))
if __name__ == '__main__':
    main()
