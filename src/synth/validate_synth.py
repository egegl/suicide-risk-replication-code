import json
import re
from pathlib import Path
from prep import constants as PC
from prep import context, textnorm
from prep import target as prep_target
from prep.validate import Report
from . import constants as SC
_RID = re.compile('S\\d{5}')
_SU = re.compile('SU\\d{4}')

def _reserialize(target: str, schema: str) -> str:
    t = json.loads(target)
    facs = sorted(set(prep_target.true_factors(t)), key=PC.FACTOR_INDEX.__getitem__)
    return prep_target.render_target(t['risk'], list(t['evidence']), facs, schema)

def run_all(pool_records: dict, provenance: list[dict], bundles: list[dict], real_index, fold_by_row: dict, factor_carriers: dict, out_path: Path, expected_model: str=SC.MODEL_PRESETS[SC.DEFAULT_PRESET]['snapshot'], target_schema: str=PC.TARGET_SCHEMA) -> Report:
    rep = Report()
    records = [r for p in SC.POOLS for r in pool_records.get(p, [])]
    prov_by_id = {p['row_id']: p for p in provenance}
    spec_by_key = {(b['bundle_id'], pp['pos']): pp for b in bundles for pp in b['posts']}
    bundle_size = {b['bundle_id']: b['n_posts'] for b in bundles}
    row_ids = [r['row_id'] for r in records]
    rep.check('row_id_format', all((_RID.fullmatch(r) for r in row_ids)), 'S##### ids')
    rep.check('row_id_unique', len(set(row_ids)) == len(row_ids), len(row_ids))
    all_keys: dict[str, str] = {}
    pos_counts: dict[tuple, int] = {}
    for r in records:
        rid = r['row_id']
        s = r['synth']
        prov = prov_by_id.get(rid)
        rep.check('provenance_present', prov is not None, rid)
        rep.check('id_shape', bool(_SU.fullmatch(s['su_id'])) and r['fold'] == -1 and (r['weight'] == 1.0) and (r['is_truncated'] is False), rid)
        rep.check('target_reserializes', _reserialize(r['target'], target_schema) == r['target'], rid)
        t = json.loads(r['target'])
        rep.check('risk_valid', t['risk'] in PC.RISK_CLASSES, rid)
        if target_schema == 'v1':
            rep.check('factors_valid', all((f in PC.FACTOR_INDEX for f in t['factors'])) and t['factors'] == sorted(set(t['factors']), key=PC.FACTOR_INDEX.__getitem__), rid)
        else:
            rep.check('factors_valid', list(t['factors']) == list(PC.FACTORS_24) and all((isinstance(v, bool) for v in t['factors'].values())), rid)
        if t['risk'] == 'Indicator':
            rep.check('indicator_no_evidence', t['evidence'] == [], rid)
        else:
            rep.check('nonindicator_has_evidence', len(t['evidence']) >= 1, rid)
        focal_text = prov['timeline_texts'][prov['focal_pos']]
        for ev in t['evidence']:
            ok = ';' not in ev and '\n' not in ev and (len(textnorm.scorer_norm(ev).split()) <= SC.SPAN_TOKENS_MAX) and (ev in focal_text) and (ev in r['input'])
            rep.check('evidence_wellformed', ok, (rid, ev))
        rep.check('no_label_leak', r['target'] not in r['input'], rid)
        rebuilt, _, is_trunc = context.build_input(prov['timeline_texts'], prov['focal_pos'], cover=[tuple(c) for c in prov['cover']] or None, pos_label=tuple(prov['pos_label']), instruction=prep_target.instruction_for(target_schema))
        rep.check('input_reconstructs', rebuilt == r['input'] and (not is_trunc), rid)
        for p in prov['timeline_texts']:
            hit = real_index.hits(p)
            rep.check('no_real_overlap', hit is None, (rid, hit))
            key = _near_dup_key(p)
            if key in all_keys and all_keys[key] != s['su_id']:
                rep.check('synth_unique', False, (rid, key))
            all_keys[key] = s['su_id']
        if s['pool_fold'] is not None:
            for ex in s['exemplar_row_ids']:
                rep.check('exemplar_fold_safe', fold_by_row.get(ex) != s['pool_fold'], (rid, ex))
        for ex in s['exemplar_row_ids']:
            rep.check('exemplar_carries_factor', ex in factor_carriers.get(s['target_factor'], set()), (rid, ex))
        spec = spec_by_key.get((s['bundle_id'], s['post_pos']))
        rep.check('bundle_risk_match', spec is not None and t['risk'] == spec['risk'], rid)
        rep.check('bundle_factors_match', sorted(set(spec['factors']), key=PC.FACTOR_INDEX.__getitem__) == prep_target.true_factors(t), rid)
        if t['risk'] != 'Indicator':
            rep.check('span_count_bounded', 1 <= len(t['evidence']) <= SC.SPAN_COUNT_MAX and abs(len(t['evidence']) - spec['n_spans']) <= 1, rid)
        rep.check('timeline_size', 3 <= bundle_size[s['bundle_id']] <= 7, rid)
        rep.check('provenance_consts', s['prompt_version'] == s['bundle_id'].split('-k', 1)[0] and s['model'] == expected_model and (s.get('target_schema', PC.TARGET_SCHEMA) == target_schema), rid)
        pool = 'full' if s['pool_fold'] is None else str(s['pool_fold'])
        for f in prep_target.true_factors(t):
            if f in SC.RARE_FACTORS:
                pos_counts[pool, f] = pos_counts.get((pool, f), 0) + 1
    rep.check('provenance_bijection', set(prov_by_id) == set(row_ids), (len(prov_by_id), len(row_ids)))
    for pool in SC.POOLS:
        for f in SC.RARE_FACTORS:
            n = pos_counts.get((pool, f), 0)
            rep.check(f'quota_{pool}_{SC.FACTOR_SLUGS[f]}', n >= SC.SYNTHETIC_POSITIVES_PER_FACTOR, f'pool {pool} {f!r}: n={n} (min {SC.SYNTHETIC_POSITIVES_PER_FACTOR})')
    if out_path is not None:
        out_path.write_text(json.dumps({'checks': rep.checks, 'warnings': rep.warnings}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return rep

def _near_dup_key(text: str) -> str:
    from prep import dedup
    return dedup.near_dup_key(text)
