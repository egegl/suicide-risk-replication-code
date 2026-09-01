import json
import re
from pathlib import Path
import numpy as np
import scorer
from synth import state
from . import constants as C
from . import oof, stats, thresholds
VARIANTS = ('fixed160', 'nested')
FIXED_STEP = C.TOTAL_STEPS
assert FIXED_STEP in C.CHECKPOINT_STEPS
BOOT_SEEDS = (C.BOOTSTRAP_SEED, 1, 2, 3)
BOOT_BS = (2000, 10000)
BOOT_METRICS = ('composite', 'rare_macro')
REANALYSIS_SUBDIR = 'reanalysis'

def _run_id(arm: str, fold: int, seed: int) -> str:
    return f'{arm}-f{fold}-s{seed}'

def _reanalysis_dir(runs_dir: Path) -> Path:
    return Path(runs_dir) / REANALYSIS_SUBDIR

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

def select_audit(repo_root: Path, runs_dir: Path, arms=('real_only', 'synthetic', 'repeated_real'), seeds=C.TRAIN_SEEDS) -> dict:
    runs_dir = Path(runs_dir)
    out: dict = {'analysis': 'checkpoint selection', 'checkpoint_steps': list(C.CHECKPOINT_STEPS), 'fixed_step': FIXED_STEP, 'arms': {}}
    for arm in arms:
        out['arms'][arm] = {}
        for seed in seeds:
            mbf = load_metrics(runs_dir, arm, seed)
            folds = sorted(mbf)
            sel = {k: selected_info(runs_dir, _run_id(arm, k, seed)) for k in folds}
            nst = nested_steps(mbf)
            per_fold = {}
            for k in folds:
                s_sel, s_nst = (sel[k]['step'], nst[k])
                per_fold[str(k)] = {'selected_step': s_sel, 'nested_step': s_nst, 'skip_step': sel[k]['skip_step'], 'val_at_selected': mbf[k][s_sel]['composite_ee1'], 'val_at_nested': mbf[k][s_nst]['composite_ee1'], 'val_at_160': mbf[k][FIXED_STEP]['composite_ee1']}
            d160 = [per_fold[str(k)]['val_at_selected'] - per_fold[str(k)]['val_at_160'] for k in folds]
            dnst = [per_fold[str(k)]['val_at_selected'] - per_fold[str(k)]['val_at_nested'] for k in folds]
            sources = {}
            for variant in VARIANTS:
                steps = variant_steps(mbf, variant)
                per_run = {}
                for k in folds:
                    kind, path = resolve_step_source(runs_dir, _run_id(arm, k, seed), steps[k])
                    if kind == 'preds' and (not path.exists()):
                        kind = 'preds-missing-locally'
                    per_run[str(k)] = {'step': steps[k], 'source': kind}
                sources[variant] = per_run
            out['arms'][arm][str(seed)] = {'selected_steps': {str(k): sel[k]['step'] for k in folds}, 'nested_steps': {str(k): nst[k] for k in folds}, 'per_fold_val': per_fold, 'mean_delta_selected_vs_160': float(np.mean(d160)), 'mean_delta_selected_vs_nested': float(np.mean(dnst)), 'sources': sources}
    rdir = _reanalysis_dir(runs_dir)
    state.atomic_write_json(rdir / 'select_audit.json', out)
    state.atomic_write_text(rdir / 'select_audit.md', _select_audit_md(out))
    return out

def _select_audit_md(audit: dict) -> str:
    lines = ['# Checkpoint selection audit', '', f"Checkpoint set {audit['checkpoint_steps']}; fixed step {audit['fixed_step']}.", '']
    for arm, per_seed in audit['arms'].items():
        for seed, a in per_seed.items():
            lines += [f'## {arm} seed {seed}', '', '| fold | selected | nested | val@selected | val@nested | val@160 | src fixed160 | src nested |', '|---|---|---|---|---|---|---|---|']
            for k, pf in a['per_fold_val'].items():
                sf = a['sources']['fixed160'][k]['source']
                sn = a['sources']['nested'][k]['source']
                lines.append(f"| {k} | {pf['selected_step']} | {pf['nested_step']} | {pf['val_at_selected']:.4f} | {pf['val_at_nested']:.4f} | {pf['val_at_160']:.4f} | {sf} | {sn} |")
            lines += ['', f"mean val delta selected-vs-160: {a['mean_delta_selected_vs_160']:+.4f}; selected-vs-nested: {a['mean_delta_selected_vs_nested']:+.4f}", '']
    return '\n'.join(lines)

def _users_by_row(processed: Path) -> dict:
    import pandas as pd
    posts = pd.read_parquet(Path(processed) / 'posts.parquet')
    tr = posts[posts['split'] == 'train']
    return dict(zip(tr['row_id'], tr['anon_user_id']))

def _arm_level(boot: dict, arm_seeds: dict[str, list[str]]) -> dict:
    reps = {arm: np.mean([np.asarray(boot['labels'][l]['replicates'], dtype=float) for l in labels], axis=0) for arm, labels in arm_seeds.items()}
    labels_out = {arm: {'observed': float(np.mean([boot['labels'][l]['observed'] for l in arm_seeds[arm]])), 'mean': float(r.mean()), 'se': float(r.std(ddof=1)), 'replicates': [float(x) for x in r]} for arm, r in reps.items()}
    deltas = {}
    for arm, r in reps.items():
        if arm == 'real_only':
            continue
        d = r - reps['real_only']
        deltas[arm] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1)), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return {**{k: boot[k] for k in ('unit', 'n_users', 'n_rows', 'b', 'seed', 'convention', 'metric') if k in boot}, 'labels': labels_out, 'deltas': deltas}

def _load_labels(runs_dir: Path, arms, seeds, suffix: str) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    per_label_rows: dict[str, list[dict]] = {}
    arm_seeds: dict[str, list[str]] = {}
    for arm in arms:
        arm_seeds[arm] = []
        for seed in seeds:
            path = Path(runs_dir) / 'pooled' / f'{arm}-s{seed}.{suffix}.jsonl'
            assert path.exists(), f'missing pooled file {path}'
            label = f'{arm}-s{seed}'
            per_label_rows[label] = state.read_jsonl(path)
            arm_seeds[arm].append(label)
    return (per_label_rows, arm_seeds)

def _with_real_only_alias(per_label_rows: dict, seeds) -> dict:
    boot_labels = dict(per_label_rows)
    boot_labels['real_only'] = per_label_rows[f'real_only-s{seeds[0]}']
    return boot_labels

def compare_variant(repo_root: Path, runs_dir: Path, variant: str, arms=('real_only', 'synthetic', 'repeated_real'), seeds=C.TRAIN_SEEDS, b: int=C.BOOTSTRAP_B) -> dict:
    assert variant in VARIANTS, f'unknown variant {variant!r} — one of {VARIANTS}'
    assert 'real_only' in arms, 'compare_variant needs the real_only arm'
    repo_root, runs_dir = (Path(repo_root), Path(runs_dir))
    processed = repo_root / 'data' / 'processed'
    gold = scorer.load_gold(processed / 'gold.jsonl')
    users = _users_by_row(processed)
    per_label_rows, arm_seeds = _load_labels(runs_dir, arms, seeds, f'{variant}.thresholded')
    raw_rows, _ = _load_labels(runs_dir, arms, seeds, f'{variant}.oof')
    arm_summaries = {arm: {'per_seed': {str(seed): {'raw': _score_pooled(raw_rows[f'{arm}-s{seed}'], repo_root), 'thresholded': _score_pooled(per_label_rows[f'{arm}-s{seed}'], repo_root)} for seed in seeds}} for arm in arms}
    boot_labels = _with_real_only_alias(per_label_rows, seeds)
    comp = stats.paired_bootstrap(boot_labels, gold, users, processed=processed, b=b, seed=C.BOOTSTRAP_SEED)
    rare = stats.paired_bootstrap(boot_labels, gold, users, processed=processed, b=b, seed=C.BOOTSTRAP_SEED, metric='rare_macro')
    arm_comp = _arm_level(comp, arm_seeds)
    decision = stats.one_se_ship(arm_comp)
    gate = stats.utility_gate(_arm_level(rare, arm_seeds), decision)
    table = stats.sweep_table(arm_summaries, arm_comp, decision)
    table['variant'] = variant
    table['utility_gate'] = gate
    rdir = _reanalysis_dir(runs_dir)
    state.atomic_write_json(rdir / f'{variant}_sweep_table.json', table)
    state.atomic_write_text(rdir / f'{variant}_sweep_table.md', _variant_sweep_md(table))
    return table

def _variant_sweep_md(table: dict) -> str:
    lines = [f"# Sweep table under `{table['variant']}` checkpoint selection", '', '| arm | thresholded ee1 | thresholded skip | delta vs real_only (95% CI) | SE |', '|---|---|---|---|---|']
    for arm, e in table['arms'].items():
        seeds = e.get('per_seed', {})
        ee1 = '/'.join((f"{s['thresholded']['composite_ee1']:.4f}" for s in seeds.values()))
        skp = '/'.join((f"{s['thresholded']['composite_skip']:.4f}" for s in seeds.values()))
        d = e.get('delta_vs_real_only')
        dtxt = f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else '—'
        se = f"{e['bootstrap']['se']:.4f}" if 'bootstrap' in e else '—'
        lines.append(f'| {arm} | {ee1} | {skp} | {dtxt} | {se} |')
    dec, gate = (table['decision'], table['utility_gate'])
    rq = gate['rare_delta_vs_repeated_real']
    rq_txt = '—' if rq['mean'] is None else f"{rq['mean']:+.4f} [{rq['ci95'][0]:+.4f}, {rq['ci95'][1]:+.4f}]"
    lines += ['', f"Decision: candidate=`{dec.get('candidate')}` ship=`{dec.get('ship')}` (bar {table['ship_bar']}, convention {table['primary_convention']})", f"Utility gate: pass=`{gate['pass']}` rare-macro Δ vs repeated_real {rq_txt}", '']
    return '\n'.join(lines)

def seed_report(repo_root: Path, runs_dir: Path, arms=('real_only', 'synthetic', 'repeated_real'), seeds=C.TRAIN_SEEDS) -> dict:
    repo_root, runs_dir = (Path(repo_root), Path(runs_dir))
    out: dict = {'analysis': 'training seed sensitivity', 'seeds': [int(s) for s in seeds], 'arms': {}}
    pooled_by_arm_seed: dict[str, dict[str, dict]] = {}
    for arm in arms:
        val_matrix: dict[str, dict[str, dict]] = {}
        pooled: dict[str, dict] = {}
        for seed in seeds:
            mbf = load_metrics(runs_dir, arm, seed)
            for k in sorted(mbf):
                sel = selected_info(runs_dir, _run_id(arm, k, seed))
                val_matrix.setdefault(str(k), {})[str(seed)] = {'selected_step': sel['step'], 'val_at_selected': mbf[k][sel['step']]['composite_ee1'], 'val_at_160': mbf[k][FIXED_STEP]['composite_ee1']}
            rows = state.read_jsonl(runs_dir / 'pooled' / f'{arm}-s{seed}.thresholded.jsonl')
            pooled[str(seed)] = _score_pooled(rows, repo_root)
        comps = [pooled[str(s)] for s in seeds]
        out['arms'][arm] = {'val_matrix': val_matrix, 'pooled_thresholded': pooled, 'seed_spread': {conv: float(max((c[conv] for c in comps)) - min((c[conv] for c in comps))) for conv in ('composite_ee1', 'composite_skip')}}
        pooled_by_arm_seed[arm] = pooled
    for arm in arms:
        if arm == 'real_only':
            continue
        out['arms'][arm]['delta_vs_real_only_per_seed'] = {str(seed): {conv: pooled_by_arm_seed[arm][str(seed)][conv] - pooled_by_arm_seed['real_only'][str(seed)][conv] for conv in ('composite_ee1', 'composite_skip')} for seed in seeds}
    rdir = _reanalysis_dir(runs_dir)
    state.atomic_write_json(rdir / 'seed_report.json', out)
    state.atomic_write_text(rdir / 'seed_report.md', _seed_report_md(out))
    return out

def _seed_report_md(rep: dict) -> str:
    seeds = [str(s) for s in rep['seeds']]
    lines = ['# Training seed report', '']
    for arm, a in rep['arms'].items():
        lines += [f'## {arm}', '', '| fold | ' + ' | '.join((f's{s} sel step | s{s} val@sel | s{s} val@160' for s in seeds)) + ' |', '|---|' + '---|' * (3 * len(seeds))]
        for k, by_seed in a['val_matrix'].items():
            cells = []
            for s in seeds:
                v = by_seed[s]
                cells += [str(v['selected_step']), f"{v['val_at_selected']:.4f}", f"{v['val_at_160']:.4f}"]
            lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
        pooled = a['pooled_thresholded']
        lines += ['', '| seed | pooled thr ee1 | pooled thr skip |', '|---|---|---|']
        for s in seeds:
            lines.append(f"| {s} | {pooled[s]['composite_ee1']:.4f} | {pooled[s]['composite_skip']:.4f} |")
        sp = a['seed_spread']
        lines += ['', f"across-seed spread (max-min): ee1 {sp['composite_ee1']:.4f}, skip {sp['composite_skip']:.4f}"]
        if 'delta_vs_real_only_per_seed' in a:
            for s in seeds:
                d = a['delta_vs_real_only_per_seed'][s]
                lines.append(f"delta vs real_only (seed {s}, same-seed pairing): ee1 {d['composite_ee1']:+.4f}, skip {d['composite_skip']:+.4f}")
        lines.append('')
    return '\n'.join(lines)

def bootstrap_sensitivity(repo_root: Path, runs_dir: Path, arms=('real_only', 'synthetic', 'repeated_real'), seeds=C.TRAIN_SEEDS, boot_seeds=BOOT_SEEDS, bs=BOOT_BS, metrics=BOOT_METRICS, out_stem: str='bootstrap_sensitivity') -> dict:
    repo_root, runs_dir = (Path(repo_root), Path(runs_dir))
    processed = repo_root / 'data' / 'processed'
    gold = scorer.load_gold(processed / 'gold.jsonl')
    users = _users_by_row(processed)
    per_label_rows, arm_seeds = _load_labels(runs_dir, arms, seeds, 'thresholded')
    boot_labels = _with_real_only_alias(per_label_rows, seeds)
    cells = []
    for metric in metrics:
        for b in bs:
            for boot_seed in boot_seeds:
                boot = stats.paired_bootstrap(boot_labels, gold, users, processed=processed, b=int(b), seed=int(boot_seed), metric=metric)
                arm_boot = _arm_level(boot, arm_seeds)
                cell = {'metric': metric, 'b': int(b), 'boot_seed': int(boot_seed), 'deltas_vs_real_only': {a: {'mean': d['mean'], 'ci95': d['ci95']} for a, d in arm_boot['deltas'].items()}}
                if metric == 'rare_macro' and 'repeated_real' in arm_seeds:
                    reps = {a: np.asarray(arm_boot['labels'][a]['replicates'], dtype=float) for a in arm_seeds}
                    vs_q = {}
                    for a in arm_seeds:
                        if a in ('real_only', 'repeated_real'):
                            continue
                        d = reps[a] - reps['repeated_real']
                        vs_q[a] = {'mean': float(d.mean()), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
                    cell['rare_delta_vs_repeated_real'] = vs_q
                cells.append(cell)
    out = {'analysis': 'bootstrap sensitivity', 'arms': list(arms), 'seeds': [int(s) for s in seeds], 'boot_seeds': [int(s) for s in boot_seeds], 'bs': [int(b) for b in bs], 'metrics': list(metrics), 'cells': cells}
    rdir = _reanalysis_dir(runs_dir)
    state.atomic_write_json(rdir / f'{out_stem}.json', out)
    state.atomic_write_text(rdir / f'{out_stem}.md', _bootstrap_sensitivity_md(out))
    return out

def _bootstrap_sensitivity_md(rep: dict) -> str:
    lines = ['# Bootstrap sensitivity', '', f"Arms {rep['arms']}, seeds {rep['seeds']}; bootstrap seeds {rep['boot_seeds']}; sample counts {rep['bs']}.", '', '| metric | B | boot seed | comparison | delta mean | 95% CI |', '|---|---|---|---|---|---|']
    for c in rep['cells']:
        for a, d in c['deltas_vs_real_only'].items():
            lines.append(f"| {c['metric']} | {c['b']} | {c['boot_seed']} | {a} - real_only | {d['mean']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
        for a, d in c.get('rare_delta_vs_repeated_real', {}).items():
            lines.append(f"| {c['metric']} | {c['b']} | {c['boot_seed']} | {a} - repeated_real | {d['mean']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines.append('')
    return '\n'.join(lines)
