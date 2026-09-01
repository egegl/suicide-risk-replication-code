import json
from pathlib import Path
import pandas as pd
from prep import constants as PC
from . import bundles, constants as SC, emit_synth, exemplars, filters, noise, parse, prompts, realism, state, validate_synth
from . import batch as batchmod

class Paths:

    def __init__(self, processed: Path, root: Path, sft_dir: Path):
        self.processed = Path(processed)
        self.root = Path(root)
        self.sft = Path(sft_dir)

    def __call__(self, rel: str) -> Path:
        return self.root / rel

def preset_name(P: Paths) -> str:
    rec = state.Manifest(P.root).data.get('requests')
    if rec and 'preset' in rec.get('config', {}):
        return rec['config']['preset']
    return SC.DEFAULT_PRESET

def preset(P: Paths) -> dict:
    return SC.MODEL_PRESETS[preset_name(P)]

def _fold_by_row(processed: Path) -> dict:
    f = pd.read_parquet(processed / 'folds.parquet')
    return dict(zip(f['row_id'], (int(x) for x in f['fold'])))

def _factor_carriers(processed: Path) -> dict:
    g = pd.read_parquet(processed / 'gold.parquet')
    carriers = {f: set() for f in SC.RARE_FACTORS}
    for r in g.itertuples():
        for fac in r.factors_set:
            if fac in carriers:
                carriers[fac].add(r.row_id)
    return carriers

def stage_bundles(P: Paths) -> dict:
    b, pools = bundles.build(P.processed)
    bundles.write(b, pools, P(SC.P_BUNDLES), P(SC.P_EXEMPLARS))
    return {'n_bundles': len(b), 'n_posts': sum((x['n_posts'] for x in b))}

def stage_requests(P: Paths, preset_key: str=SC.DEFAULT_PRESET) -> dict:
    prompts.assert_prompt_pinned()
    pre = SC.MODEL_PRESETS[preset_key]
    b = state.read_jsonl(P(SC.P_BUNDLES))
    stale = sorted({x['bundle_id'] for x in b if not x['bundle_id'].startswith(f'{SC.PROMPT_VERSION}-')})
    if stale:
        raise SystemExit(f'requests refused: {len(stale)} bundle_ids are not {SC.PROMPT_VERSION}-vintage (e.g. {stale[:3]}) — bundles.jsonl was sampled under an older PROMPT_VERSION. Rerun bundles first.')
    pools = json.loads(P(SC.P_EXEMPLARS).read_text(encoding='utf-8'))
    ex_by_id = {p['row_id']: p for pool in pools.values() for exs in pool.values() for p in exs}
    by_pool: dict = {}
    for bd in b:
        rendered = [exemplars.render(ex_by_id[rid]) for rid in bd['exemplar_row_ids']]
        by_pool.setdefault(bd['pool'], []).append(prompts.build_request(bd, rendered, pre))
    counts = {}
    for pool, reqs in by_pool.items():
        path = P(SC.P_REQUESTS) / f'batch_input_pool{pool}.jsonl'
        state.atomic_write_jsonl(path, reqs)
        counts[pool] = len(reqs)
    return {'requests_per_pool': counts}

def stage_parse(P: Paths) -> dict:
    b = state.read_jsonl(P(SC.P_BUNDLES))
    st = batchmod.load_state(P(SC.P_BATCH_STATE))
    batch_id_by_pool = {pool: rec.get('batch_id', '') for pool, rec in st['shards'].items()}
    raw_by_pool = {}
    for gz in sorted(P(SC.P_RAW).glob('batch_output_pool*.jsonl.gz')):
        pool = gz.name[len('batch_output_pool'):-len('.jsonl.gz')]
        recorded = st['shards'].get(pool, {}).get('fetched_sha')
        actual = state.sha256_file(gz)
        if recorded != actual:
            raise SystemExit(f'parse refused: {gz.name} was not fetched by the current submission (recorded fetched_sha {str(recorded)[:12]}, file {actual[:12]}). Run poll/fetch for this round first.')
        raw_by_pool[pool] = state.read_jsonl_gz(gz)
    cands, rejects = parse.parse_all(raw_by_pool, b, batch_id_by_pool, model=preset(P)['snapshot'])
    if rejects and (not cands):
        raise SystemExit(f'parse refused: 0 candidates with {len(rejects)} rejects — responses do not match the on-disk bundles (wrong round?) or every timeline failed. Investigate parsed/rejects.jsonl before overwriting downstream stages.')
    state.atomic_write_jsonl(P(SC.P_PARSED) / 'candidates.jsonl', cands)
    state.atomic_write_jsonl(P(SC.P_PARSED) / 'rejects.jsonl', rejects)
    return {'candidates': len(cands), 'rejects': len(rejects)}

def stage_noise(P: Paths) -> dict:
    cands = state.read_jsonl(P(SC.P_PARSED) / 'candidates.jsonl')
    bundles_ = state.read_jsonl(P(SC.P_BUNDLES))
    drops_by_bundle = {b['bundle_id']: bool(b['persona_style']['drops_apostrophes']) for b in bundles_}
    lowercase_by_bundle = {b['bundle_id']: bool(b['persona_style']['lowercase_i']) for b in bundles_}
    missing = sorted({c['bundle_id'] for c in cands} - drops_by_bundle.keys())
    if missing:
        raise SystemExit(f'noise refused: {len(missing)} candidate bundle_ids absent from bundles.jsonl (e.g. {missing[:3]}) — parsed/candidates.jsonl is from a different round than bundles.jsonl. Rerun parse for this round before noise.')
    out, rejects, stats = noise.noise_all(cands, drops_by_bundle, lowercase_by_bundle)
    state.atomic_write_jsonl(P(SC.P_NOISED) / 'candidates.jsonl', out)
    state.atomic_write_jsonl(P(SC.P_NOISED) / 'rejects.jsonl', rejects)
    return {'noised': len(out), 'dropped': len(rejects), **stats}

def stage_filter(P: Paths) -> dict:
    cands = state.read_jsonl(P(SC.P_NOISED) / 'candidates.jsonl')
    index = filters.RealIndex(filters.load_real_posts(P.processed))
    accepted, rejects = filters.filter_all(cands, index)
    state.atomic_write_jsonl(P(SC.P_FILTERED) / 'candidates.jsonl', accepted)
    state.atomic_write_jsonl(P(SC.P_FILTERED) / 'rejects.jsonl', rejects)
    return {'accepted': len(accepted), 'rejected': len(rejects)}

def stage_emit(P: Paths, target_schema: str=PC.TARGET_SCHEMA) -> dict:
    cands = state.read_jsonl(P(SC.P_FILTERED) / 'candidates.jsonl')
    b = state.read_jsonl(P(SC.P_BUNDLES))
    pool_records, provenance = emit_synth.emit(cands, target_schema=target_schema)
    index = filters.RealIndex(filters.load_real_posts(P.processed))
    fold_by_row = _fold_by_row(P.processed)
    carriers = _factor_carriers(P.processed)
    validate_synth.run_all(pool_records, provenance, b, index, fold_by_row, carriers, P(SC.P_VALIDATION), expected_model=preset(P)['snapshot'], target_schema=target_schema)
    for pool in SC.POOLS:
        state.atomic_write_jsonl(P(SC.P_POOLS) / f'pool_fold{pool}.jsonl', pool_records[pool])
    state.atomic_write_jsonl(P(SC.P_PROVENANCE), provenance)
    return {'records': sum((len(v) for v in pool_records.values())), 'per_pool': {p: len(pool_records[p]) for p in SC.POOLS}}

def load_emitted(P: Paths, stage: str='check-realism') -> tuple[dict, dict]:
    pool_files = {p: P(SC.P_POOLS) / f'pool_fold{p}.jsonl' for p in SC.POOLS}
    missing = [f.name for f in [*pool_files.values(), P(SC.P_PROVENANCE)] if not f.exists()]
    if missing:
        raise SystemExit(f"{stage} refused: emit artifacts missing ({', '.join(missing)}) — run emit first. A vacuous pass on zero rows must never gate assemble.")
    pool_records = {p: state.read_jsonl(f) for p, f in pool_files.items()}
    if not any(pool_records.values()):
        raise SystemExit(f'{stage} refused: every pool file is empty — a vacuous pass on zero rows must never gate assemble.')
    prov = {r['row_id']: r for r in state.read_jsonl(P(SC.P_PROVENANCE))}
    return (pool_records, prov)

def stage_realism(P: Paths) -> dict:
    pool_records, prov = load_emitted(P, 'check-realism')
    return realism.check(pool_records, prov, P.processed, P(SC.P_REALISM))

def stage_assemble(P: Paths, arms: list[str]) -> dict:
    from . import assemble
    fold_by_row = _fold_by_row(P.processed)
    ms = assemble.assemble(arms, P(SC.P_POOLS), P.sft, P(SC.P_ARMS), fold_by_row)
    return {a: {'folds': {k: v['n_real'] + v['n_synth'] for k, v in m['folds'].items()}} for a, m in ms.items()}

def cost_estimate(P: Paths) -> dict:
    sys_chars = user_chars = 0
    for req in P(SC.P_REQUESTS).glob('batch_input_pool*.jsonl'):
        for line in state.read_jsonl(req):
            for msg in line['body'].get('input') or line['body'].get('messages') or []:
                if msg['role'] == 'system':
                    sys_chars += len(msg['content'])
                else:
                    user_chars += len(msg['content'])
    pre = preset(P)
    sys_tok = sys_chars / SC.EST_CHARS_PER_TOKEN
    user_tok = user_chars / SC.EST_CHARS_PER_TOKEN
    in_tok = sys_tok + user_tok
    out_tok = in_tok * SC.EST_OUTPUT_FRAC
    out_usd = out_tok / 1000000.0 * pre['price_out']
    return {'preset': preset_name(P), 'model': pre['snapshot'], 'est_input_tokens': int(in_tok), 'est_output_tokens': int(out_tok), 'est_cost_usd_no_cache': round(in_tok / 1000000.0 * pre['price_in'] + out_usd, 2), 'est_cost_usd_cached': round(user_tok / 1000000.0 * pre['price_in'] + sys_tok / 1000000.0 * pre['price_cached_in'] + out_usd, 2)}
