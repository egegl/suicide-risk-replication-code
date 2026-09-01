import json
import random
import re
from pathlib import Path
from . import constants as C
from . import target as target_schema
from . import textnorm
from .clean import parse_evidence
_WS = re.compile('\\s+')

def _scorer_norm(s: str) -> str:
    return _WS.sub(' ', s.lower().strip())

class Report:

    def __init__(self):
        self.checks: dict[str, dict] = {}
        self.warnings: list[str] = []

    def check(self, name: str, ok: bool, detail=''):
        self.checks[name] = {'pass': bool(ok), 'detail': str(detail)}
        assert ok, f'invariant failed: {name} ({detail})'

    def warn(self, msg: str):
        self.warnings.append(msg)

def run_all(train, lb, spans, out_dir: Path) -> Report:
    rep = Report()
    E = C.EXPECTED
    n_spans = int(train['gold_eval_spans'].map(len).sum())
    rep.check('spans_total', n_spans == E['gold_spans'], n_spans)
    n_ev = int(train['has_evidence'].sum())
    rep.check('evidence_posts', n_ev == E['evidence_posts'], n_ev)
    no_ev = train[~train['has_evidence']]
    rep.check('no_evidence_posts', len(no_ev) == E['no_evidence_posts'], len(no_ev))
    rep.check('no_evidence_all_indicator', (no_ev['risk_norm'] == 'Indicator').all())
    non_ind = train[train['risk_norm'] != 'Indicator']
    rep.check('non_indicator_all_have_evidence', non_ind['has_evidence'].all())
    ind_ev = int(((train['risk_norm'] == 'Indicator') & train['has_evidence']).sum())
    rep.check('indicator_with_evidence', ind_ev == E['indicator_with_evidence'], ind_ev)
    for r in train.itertuples():
        for s in r.gold_eval_spans:
            assert ';' not in s and '\n' not in s, (r.row_id, s)
            assert len(_scorer_norm(s).split()) <= E['max_span_tokens'], (r.row_id, s)
        joined = '; '.join(r.gold_eval_spans)
        if r.gold_eval_spans:
            assert parse_evidence(joined) == list(r.gold_eval_spans), r.row_id
    rep.check('span_constraints_and_roundtrip', True)
    diff = train[train['risk_train'] != train['risk_norm']]
    rep.check('risk_train_diffs_bounded', len(diff) <= 4, len(diff))
    rep.check('risk_train_diffs_flagged', diff['is_label_conflict'].all() if len(diff) else True)
    one_rep = train.groupby('near_dup_cluster_id')['is_representative'].sum()
    rep.check('one_representative_per_cluster', (one_rep == 1).all())
    posts_by_id = dict(zip(train['row_id'], train['post_raw']))
    rung_counts = spans[(spans['parent_span_idx'] == -1) & spans['matched']]['rung'].value_counts().to_dict()
    for rung, expected in E['ladder'].items():
        rep.check(f'ladder_{rung}', rung_counts.get(rung, 0) == expected, f'{rung_counts.get(rung, 0)} vs {expected}')
    unmatched = int((~spans[spans['parent_span_idx'] == -1]['matched']).sum())
    rep.check('unmatched_bounded', unmatched <= E['max_unmatched_spans'], unmatched)
    official_norm_misses = 0
    for s in spans[spans['matched']].itertuples():
        raw = posts_by_id[s.row_id]
        assert raw[s.raw_start:s.raw_end] == s.raw_slice, s.row_id
        np_, ng_ = (textnorm.normalize(s.raw_slice), textnorm.normalize(s.text_gold))
        assert np_ in ng_ or ng_ in np_, (s.row_id, s.raw_slice, s.text_gold)
        ns, ng = (_scorer_norm(s.raw_slice), _scorer_norm(s.text_gold))
        if not (ns in ng or ng in ns):
            official_norm_misses += 1
    rep.check('aligned_byte_identity_and_pipeline_containment', True, f'official-norm misses (annotator-straightened quotes): {official_norm_misses}')
    for s in spans[spans['use_for_supervision']].itertuples():
        raw = posts_by_id[s.row_id]
        assert raw[s.scorer_start:s.scorer_end] == s.scorer_slice, s.row_id
        ns, ng = (_scorer_norm(s.scorer_slice), _scorer_norm(s.text_gold))
        assert ns and (ns in ng or ng in ns), (s.row_id, s.scorer_slice, s.text_gold)
        assert len(ns.split()) <= 3 * len(ng.split()), (s.row_id, s.scorer_slice)
    rep.check('supervision_slices_score_officially', True, f"{int(spans['use_for_supervision'].sum())} slices")
    multi_occ = int((spans[(spans['parent_span_idx'] == -1) & spans['matched']]['n_occurrences'] > 1).sum())
    rep.check('multi_occurrence_spans', multi_occ == C.EXPECTED['multi_occurrence_spans'], multi_occ)
    p00732 = spans[spans['row_id'] == 'P00732']
    rep.check('p00732_not_gap_split', (p00732['parent_span_idx'] == -1).all() and p00732['matched'].all())
    for p in list(train['post_raw']) + list(lb['post_raw']):
        assert len(p.lower()) == len(p)
    rep.check('lowercase_length_stable_all_posts', True)
    rng = random.Random(0)
    sample = train.sample(n=200, random_state=0)
    for r in sample.itertuples():
        norm, starts, ends = textnorm.normalize_with_map(r.post_raw)
        if len(norm) < 4:
            continue
        for _ in range(3):
            a = rng.randrange(0, len(norm) - 2)
            b = rng.randrange(a + 1, len(norm))
            while a > 0 and starts[a] == starts[a - 1]:
                a -= 1
            while b > a + 1 and b < len(norm) and (starts[b] == starts[b - 1]):
                b -= 1
            t = norm[a:b].strip()
            if not t:
                continue
            s, e = textnorm.raw_slice(r.post_raw, starts, ends, a, b)
            assert textnorm.normalize(r.post_raw[s:e]) == t, (r.row_id, a, b)
    rep.check('offset_map_property_test', True)
    grp = train.groupby('dup_group_id').size()
    multi = grp[grp > 1]
    rep.check('exact_dup_groups', len(multi) == E['exact_dup_groups'], len(multi))
    rep.check('exact_dup_rows', int(multi.sum()) == E['exact_dup_rows'], int(multi.sum()))
    comp = dict(zip(train['row_id'], train['cv_component']))
    rep.check('cross_user_dup_same_component', comp['P00082'] == comp['P01743'])
    rep.check('no_component_spans_folds', (train.groupby('cv_component')['fold'].nunique() == 1).all())
    att = train[train['risk_norm'] == 'Attempt'].groupby('fold').size()
    rep.check('attempt_per_fold', len(att) == C.N_FOLDS and (att >= C.MIN_ATTEMPT_PER_FOLD).all(), att.to_dict())
    for j, f in enumerate(C.FACTORS_24):
        per_fold = train[train[f'f_{j:02d}'] == 1].groupby('fold').size()
        if len(per_fold) < C.N_FOLDS:
            rep.warn(f'factor {f!r} (support {C.FACTOR_SUPPORTS[f]}) absent from folds {sorted(set(range(C.N_FOLDS)) - set(per_fold.index))} — pool rare-class stats across folds')
    lb_posts = dict(zip(lb['row_id'], lb['post_raw']))
    for a, b in C.LB_DUP_PAIRS:
        assert lb_posts[a] == lb_posts[b]
    rep.check('lb_dup_pairs_byte_identical', True)
    rep.check('p01130_free_label', lb_posts['P01130'] == posts_by_id['P00082'] == posts_by_id['P01743'])
    train_chars = set(''.join(train['post_raw']))
    novel = {ch for ch in set(''.join(lb['post_raw'])) - train_chars if ord(ch) > 127}
    if novel:
        rep.warn(f'leaderboard has non-ASCII chars unseen in train: {sorted(novel)!r} — check CHAR_MAP coverage')

    def _check_sft_dir(sft_dir: Path, schema: str) -> list[dict]:
        sft = [json.loads(line) for line in (sft_dir / 'full_train.jsonl').read_text().splitlines()]
        for recd in sft:
            t = json.loads(recd['target'])
            assert recd['input'].startswith(C.SFT_INSTRUCTIONS[schema]), recd['row_id']
            assert t['risk'] in C.RISK_CLASSES
            if schema == 'v1':
                assert list(t) == ['risk', 'evidence', 'factors'], recd['row_id']
                assert t['factors'] == sorted(set(t['factors']), key=C.FACTOR_INDEX.__getitem__)
            else:
                assert list(t) == ['evidence', 'factors', 'risk'], recd['row_id']
                assert list(t['factors']) == list(C.FACTORS_24), recd['row_id']
                assert all((isinstance(v, bool) for v in t['factors'].values())), recd['row_id']
            raw = posts_by_id[recd['row_id']]
            if t['risk'] == 'Indicator':
                assert t['evidence'] == [], recd['row_id']
            else:
                assert t['evidence'], recd['row_id']
            for ev in t['evidence']:
                assert ev in raw and ev in recd['input'], (recd['row_id'], ev)
            assert recd['target'] not in recd['input']
        lb_inf = [json.loads(line) for line in (sft_dir / 'leaderboard_infer.jsonl').read_text().splitlines()]
        assert len(lb_inf) == len(lb), (schema, len(lb_inf))
        return sft
    sft_v2 = _check_sft_dir(out_dir / 'sft', 'v2')
    sft_v1 = _check_sft_dir(out_dir / 'sft_v1', 'v1')
    rep.check('sft_targets_valid', True, f'{len(sft_v2)} rows x {len(C.TARGET_SCHEMAS)} schemas')
    rep.check('sft_lb_rows', True, len(lb))
    assert len(sft_v1) == len(sft_v2)
    for record1, record2 in zip(sft_v1, sft_v2):
        assert record1['row_id'] == record2['row_id']
        assert record1['fold'] == record2['fold'] and record1['weight'] == record2['weight']
        t1, t2 = (json.loads(record1['target']), json.loads(record2['target']))
        assert t1['risk'] == t2['risk'] and t1['evidence'] == t2['evidence']
        assert t1['factors'] == target_schema.true_factors(t2), record1['row_id']
        assert record1['input'].split('\n\n', 1)[1] == record2['input'].split('\n\n', 1)[1], record1['row_id']
    rep.check('sft_schemas_equivalent', True, f'{len(sft_v1)} rows')
    return rep

def check_inference_frame(df, train_users: set, train_char_census: set) -> list[str]:
    problems = []
    if not df['row_id'].str.fullmatch('P\\d{5}').all() or not df['row_id'].is_unique:
        problems.append('row_id format/uniqueness')
    if set(df['anon_user_id']) & train_users:
        problems.append('user overlap with train')
    novel = {ch for ch in set(''.join(df['post'])) - train_char_census if ord(ch) > 127}
    if novel:
        problems.append(f'novel non-ASCII chars {sorted(novel)!r} — extend CHAR_MAP?')
    for _, g in df.groupby('anon_user_id'):
        if sorted(g['post_id']) != list(range(len(g))):
            problems.append(f"post_id not consecutive for {g['anon_user_id'].iloc[0]}")
            break
    return problems
