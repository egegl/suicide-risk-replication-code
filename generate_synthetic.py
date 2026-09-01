#!/usr/bin/env python3
'Generate and assemble synthetic training data.'
import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def _load_dotenv(path: Path=Path(__file__).parent / '.env') -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.removeprefix('export ').partition('=')
        key, value = (key.strip(), value.strip().strip('\'"'))
        if key:
            os.environ.setdefault(key, value)
from synth import constants as SC
from synth import pipeline, state
from synth import batch as batchmod

def make_paths(args) -> pipeline.Paths:
    root = Path(args.out)
    return pipeline.Paths(Path(args.processed), root, Path(args.sft))

def _manifest(P: pipeline.Paths) -> state.Manifest:
    return state.Manifest(P.root)

def _processed_inputs(P: pipeline.Paths) -> list[Path]:
    return [P.processed / n for n in ('gold.parquet', 'posts.parquet', 'spans.parquet', 'folds.parquet')]

def _refuse_post_submission(args, P, stage):
    if P(SC.P_BATCH_STATE).exists() and (not getattr(args, 'force', False)):
        sys.exit(f'{stage} refused: a batch was already submitted from this dir (batch_state.json exists); re-rendering would desync provenance. Use --force only to start a new generation round.')

def cmd_bundles(args, P):
    _refuse_post_submission(args, P, 'bundles')
    if getattr(args, 'check_determinism', False):
        import tempfile
        h = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                bp, ep = (Path(d) / 'b.jsonl', Path(d) / 'e.json')
                b, pools = _build_bundles(P)
                from synth import bundles as B
                B.write(b, pools, bp, ep)
                h.append((state.sha256_file(bp), state.sha256_file(ep)))
        assert h[0] == h[1], f'bundles non-deterministic: {h}'
        print(json.dumps({'check_determinism': 'ok', 'hash': h[0]}, indent=2))
    stats = pipeline.stage_bundles(P)
    _manifest(P).record('bundles', {'prompt_version': SC.PROMPT_VERSION, 'seed': SC.SYNTH_SEED}, _processed_inputs(P), [P(SC.P_BUNDLES), P(SC.P_EXEMPLARS)])
    print(json.dumps(stats, indent=2))

def _build_bundles(P):
    from synth import bundles as B
    return B.build(P.processed)

def cmd_requests(args, P):
    _refuse_post_submission(args, P, 'requests')
    m = _manifest(P)
    m.check_upstream('requests', 'bundles')
    stats = pipeline.stage_requests(P, args.preset)
    outs = sorted(P(SC.P_REQUESTS).glob('batch_input_pool*.jsonl'))
    m.record('requests', {'prompt_version': SC.PROMPT_VERSION, 'preset': args.preset, 'preset_def': dict(SC.MODEL_PRESETS[args.preset])}, [P(SC.P_BUNDLES), P(SC.P_EXEMPLARS)], outs)
    print(json.dumps({**stats, 'cost_estimate': pipeline.cost_estimate(P)}, indent=2))

def _backend(args):
    return batchmod.OpenAIBatchBackend()

def cmd_submit(args, P):
    reqs = {p.name[len('batch_input_pool'):-len('.jsonl')]: p for p in sorted(P(SC.P_REQUESTS).glob('batch_input_pool*.jsonl'))}
    est = pipeline.cost_estimate(P)
    if args.backend == 'sync':
        est = {**est, 'est_cost_usd_no_cache': round(est['est_cost_usd_no_cache'] * 2, 2), 'est_cost_usd_cached': round(est['est_cost_usd_cached'] * 2, 2), 'pricing': 'standard (sync bypasses the 50% Batch discount)'}
    print(json.dumps({'about_to_submit': len(reqs), 'cost_estimate': est}, indent=2))
    if not args.yes:
        print('refusing to submit without --yes (this spends money).', file=sys.stderr)
        sys.exit(2)
    if args.backend == 'sync':
        from openai import OpenAI
        client = OpenAI()
        summary = batchmod.collect_sync(client, reqs, P(SC.P_BATCH_STATE), P(SC.P_RAW), workers=SC.SYNC_WORKERS, max_retries=SC.SYNC_MAX_RETRIES)
        print(json.dumps({args.backend: summary, 'note': 'outputs already in raw_responses/; poll/fetch not needed — continue with `run --until emit`'}, indent=2))
        return
    st = batchmod.submit(_backend(args), reqs, P(SC.P_BATCH_STATE), endpoint=SC.BATCH_URL, window=SC.BATCH_COMPLETION_WINDOW, prompt_version=SC.PROMPT_VERSION, retry_failed=args.retry_failed)
    print(json.dumps({'shards': {k: v['status'] for k, v in st['shards'].items()}}, indent=2))

def cmd_poll(args, P):
    backend = _backend(args)
    while True:
        st = batchmod.poll(backend, P(SC.P_BATCH_STATE))
        done = batchmod.all_terminal(st)
        print(json.dumps({'shards': {k: v['status'] for k, v in st['shards'].items()}, 'all_terminal': done}, indent=2), flush=True)
        if done or not getattr(args, 'watch', False):
            break
        time.sleep(SC.POLL_INTERVAL_S)

def cmd_fetch(args, P):
    fetched = batchmod.fetch(_backend(args), P(SC.P_BATCH_STATE), P(SC.P_RAW))
    print(json.dumps({'fetched_pools': fetched}, indent=2))

def cmd_cancel(args, P):
    st = batchmod.cancel(_backend(args), P(SC.P_BATCH_STATE))
    print(json.dumps({'shards': {k: v['status'] for k, v in st['shards'].items()}, 'note': 'resubmit with: submit --retry-failed --yes'}, indent=2))

def _ledgered(args, P, stage, inputs, run_fn, outputs_fn, extra_config=None):
    m = _manifest(P)
    config = {'seed': SC.SYNTH_SEED, 'prompt_version': SC.PROMPT_VERSION, 'preset': pipeline.preset_name(P), **(extra_config or {})}
    inputs = [p for p in inputs if p.exists()]
    if not getattr(args, 'force', False) and m.unchanged(stage, config, inputs):
        print(json.dumps({stage: 'up to date (skipped; --force to rerun)'}, indent=2))
        return None
    stats = run_fn()
    m.record(stage, config, inputs, [p for p in outputs_fn() if p.exists()])
    return stats

def cmd_parse(args, P):
    raw = sorted(P(SC.P_RAW).glob('batch_output_pool*.jsonl.gz'))
    stats = _ledgered(args, P, 'parse', [P(SC.P_BUNDLES), *raw], lambda: pipeline.stage_parse(P), lambda: [P(SC.P_PARSED) / 'candidates.jsonl', P(SC.P_PARSED) / 'rejects.jsonl'])
    if stats is not None:
        print(json.dumps(stats, indent=2))

def cmd_noise(args, P):
    stats = _ledgered(args, P, 'noise', [P(SC.P_PARSED) / 'candidates.jsonl', P(SC.P_BUNDLES)], lambda: pipeline.stage_noise(P), lambda: [P(SC.P_NOISED) / 'candidates.jsonl', P(SC.P_NOISED) / 'rejects.jsonl'], extra_config={'noise_version': SC.NOISE_VERSION})
    if stats is None:
        return
    warnings = []
    if abs(stats.get('curly_rate', 0) - SC.CURLY_TARGET_MARGINAL) > SC.CURLY_RATE_TOL:
        warnings.append(f"curly rate {stats['curly_rate']:.3f} off target {SC.CURLY_TARGET_MARGINAL}")
    for key, target in SC.REAL_STYLE_TARGETS.items():
        rate = stats.get(f"{key.removesuffix('_posts')}_post_rate", 0.0)
        if abs(rate - target) > SC.STYLE_RATE_TOL:
            warnings.append(f'{key} rate {rate:.3f} off target {target}')
    if warnings:
        stats['warnings'] = warnings
    print(json.dumps(stats, indent=2))

def cmd_filter(args, P):
    stats = _ledgered(args, P, 'filter', [P(SC.P_NOISED) / 'candidates.jsonl', P.processed / 'posts.parquet'], lambda: pipeline.stage_filter(P), lambda: [P(SC.P_FILTERED) / 'candidates.jsonl', P(SC.P_FILTERED) / 'rejects.jsonl'])
    if stats is not None:
        print(json.dumps(stats, indent=2))

def cmd_emit(args, P):
    stats = _ledgered(args, P, 'emit', [P(SC.P_FILTERED) / 'candidates.jsonl', P(SC.P_BUNDLES), *_processed_inputs(P)], lambda: pipeline.stage_emit(P), lambda: [*sorted(P(SC.P_POOLS).glob('pool_fold*.jsonl')), P(SC.P_PROVENANCE), P(SC.P_VALIDATION)])
    if stats is not None:
        print(json.dumps(stats, indent=2))

def cmd_realism(args, P):
    pools = sorted(P(SC.P_POOLS).glob('pool_fold*.jsonl'))
    stats = _ledgered(args, P, 'realism', [*pools, P(SC.P_PROVENANCE), P.processed / 'posts.parquet', P.processed / 'gold.parquet'], lambda: pipeline.stage_realism(P), lambda: [P(SC.P_REALISM)], extra_config={'max_auc': SC.REALISM_MAX_AUC})
    if stats is not None:
        print(json.dumps(stats, indent=2))
    rep_path = P(SC.P_REALISM)
    rep = json.loads(rep_path.read_text(encoding='utf-8')) if rep_path.exists() else {}
    if not rep.get('pass', False):
        print(json.dumps({'realism': {k: rep.get(k) for k in ('auc_mean', 'threshold', 'pass')}, 'note': 'descriptive only — synthetic data ships on the utility gate, not this AUC'}, indent=2))

def cmd_assemble(args, P):
    arms = args.arms.split(',') if args.arms else list(SC.ARMS)
    synth_arms = [a for a in arms if a != 'repeated_real']
    m = _manifest(P)
    realism_auc = None
    if synth_arms:
        rep_path = P(SC.P_REALISM)
        if 'realism' not in m.data or not rep_path.exists():
            sys.exit('assemble refused: check-realism has not run on these pools (its descriptive AUC must be recorded against the exact pools being assembled). Run check-realism first.')
        m.check_upstream('assemble', 'realism')
        if not m.inputs_unchanged('realism'):
            sys.exit('assemble refused: the realism report is stale — the pools changed after check-realism ran. Rerun check-realism.')
        rep = json.loads(rep_path.read_text(encoding='utf-8'))
        realism_auc = rep.get('auc_mean')
        if not rep.get('pass', False):
            print(json.dumps({'realism_note': f"descriptive AUC {realism_auc} > {rep.get('threshold')} — recorded; ship/no-ship is the utility gate's call"}, indent=2))
    pools = sorted(P(SC.P_POOLS).glob('pool_fold*.jsonl'))
    sft_files = [p for p in [*sorted(P.sft.glob('train_fold*.jsonl')), P.sft / 'full_train.jsonl'] if p.exists()]
    stats = _ledgered(args, P, 'assemble', [*pools, P(SC.P_REALISM), P.processed / 'folds.parquet', *sft_files], lambda: pipeline.stage_assemble(P, arms), lambda: [*sorted(P(SC.P_ARMS).glob('*/train_*.jsonl')), *sorted(P(SC.P_ARMS).glob('*/manifest.json'))], extra_config={'arms': sorted(arms), 'realism_auc': realism_auc})
    if stats is not None:
        print(json.dumps(stats, indent=2))

def cmd_status(args, P):
    m = _manifest(P)
    st = batchmod.load_state(P(SC.P_BATCH_STATE))
    print(json.dumps({'stages': sorted(m.data), 'batch_shards': {k: v['status'] for k, v in st.get('shards', {}).items()}}, indent=2))
LOCAL_ORDER = ['bundles', 'requests', 'parse', 'noise', 'filter', 'emit']

def cmd_run(args, P):
    until = args.until
    assert until in LOCAL_ORDER, f'--until must be one of {LOCAL_ORDER}'
    submitted = P(SC.P_BATCH_STATE).exists()
    for stage in LOCAL_ORDER[:LOCAL_ORDER.index(until) + 1]:
        if stage in ('bundles', 'requests') and submitted:
            print(f'[run] skipping {stage}: batch already submitted', file=sys.stderr)
            continue
        if stage in ('parse',) and (not any(P(SC.P_RAW).glob('*.jsonl.gz'))):
            print(f'[run] stopping before {stage}: no fetched responses (submit/poll/fetch are manual).', file=sys.stderr)
            break
        print(f'[run] {stage}', file=sys.stderr)
        globals()[f'cmd_{stage}'](args, P)

def main(argv=None, *, prog=None, description=None, out_default=SC.SYNTH_DIR, preset_default=SC.DEFAULT_PRESET, backend_default='openai'):
    _load_dotenv()
    ap = argparse.ArgumentParser(prog=prog, description=description or __doc__)
    ap.add_argument('--out', default=out_default)
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--sft', default='data/processed/sft')
    ap.add_argument('--force', action='store_true', help='rerun a stage even if the manifest says it is up to date')
    ap.add_argument('--preset', default=preset_default, choices=sorted(SC.MODEL_PRESETS), help='generator model preset; takes effect at the `requests` stage and is recorded in the manifest for later stages')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def _also_after_subcommand(pp, *names):
        for name in names:
            if name == '--force':
                pp.add_argument('--force', action='store_true', default=argparse.SUPPRESS)
            elif name == '--preset':
                pp.add_argument('--preset', choices=sorted(SC.MODEL_PRESETS), default=argparse.SUPPRESS)
    b = sub.add_parser('bundles')
    b.add_argument('--check-determinism', action='store_true')
    b.set_defaults(fn=cmd_bundles)
    _also_after_subcommand(b, '--force')
    r = sub.add_parser('requests')
    r.set_defaults(fn=cmd_requests)
    _also_after_subcommand(r, '--force', '--preset')
    s = sub.add_parser('submit')
    s.add_argument('--yes', action='store_true')
    s.add_argument('--retry-failed', action='store_true')
    s.add_argument('--backend', default=backend_default, choices=['openai', 'sync'])
    s.set_defaults(fn=cmd_submit)
    for name, fn in [('poll', cmd_poll), ('fetch', cmd_fetch), ('cancel', cmd_cancel)]:
        pp = sub.add_parser(name)
        pp.add_argument('--backend', default='openai', choices=['openai'])
        pp.add_argument('--watch', action='store_true')
        pp.set_defaults(fn=fn)
    for name, fn in [('parse', cmd_parse), ('noise', cmd_noise), ('filter', cmd_filter), ('emit', cmd_emit), ('check-realism', cmd_realism), ('status', cmd_status)]:
        pp = sub.add_parser(name)
        pp.set_defaults(fn=fn)
        _also_after_subcommand(pp, '--force')
    a = sub.add_parser('assemble')
    a.add_argument('--arms', default='')
    _also_after_subcommand(a, '--force')
    a.set_defaults(fn=cmd_assemble)
    rn = sub.add_parser('run')
    rn.add_argument('--until', default='emit')
    rn.set_defaults(fn=cmd_run)
    _also_after_subcommand(rn, '--force')
    args = ap.parse_args(argv)
    P = make_paths(args)
    args.fn(args, P)
if __name__ == '__main__':
    main()
