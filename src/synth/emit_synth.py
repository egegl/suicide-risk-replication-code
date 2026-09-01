from prep import constants as PC
from prep import context
from prep import target as prep_target
from . import constants as SC

def _ranges(post_text: str, evidence: list[str]) -> list[tuple[int, int]]:
    claimed: list[tuple[int, int]] = []
    for ev in evidence:
        start = 0
        while True:
            i = post_text.find(ev, start)
            assert i != -1, ('evidence not in post', ev)
            j = i + len(ev)
            if not any((i < e and s < j for s, e in claimed)):
                claimed.append((i, j))
                break
            start = i + 1
    return sorted(claimed)

def emit(candidates: list[dict], target_schema: str=PC.TARGET_SCHEMA) -> tuple[dict, list[dict]]:
    pool_records: dict = {p: [] for p in SC.POOLS}
    provenance: list = []
    serial = 0

    def pool_key(c):
        p = c['pool']
        return (SC.POOLS.index(p), c['bundle_id'])
    for c in sorted(candidates, key=pool_key):
        texts = [p['post_text'] for p in sorted(c['posts'], key=lambda x: x['pos'])]
        for p in sorted(c['posts'], key=lambda x: x['pos']):
            if not p['emit_record']:
                continue
            focal_pos = p['pos']
            ranges = _ranges(texts[focal_pos], p['evidence']) if p['evidence'] else []
            cover = ranges or None
            pos_label = (focal_pos + 1, c['declared_n'])
            input_text, kept, is_trunc = context.build_input(texts, focal_pos, cover=cover, pos_label=pos_label, instruction=prep_target.instruction_for(target_schema))
            assert not is_trunc, (c['bundle_id'], focal_pos)
            ordered_ev = [] if p['risk'] == 'Indicator' else [texts[focal_pos][s:e] for s, e in ranges]
            serial += 1
            assert serial <= 99999, 'S id space exhausted'
            row_id = f'S{serial:05d}'
            rec = {'row_id': row_id, 'input': input_text, 'is_truncated': is_trunc, 'fold': -1, 'weight': 1.0, 'target': prep_target.render_target(p['risk'], ordered_ev, sorted(set(p['factors']), key=PC.FACTOR_INDEX.__getitem__), target_schema), 'synth': {'pool_fold': None if c['pool'] == 'full' else int(c['pool']), 'su_id': c['su_id'], 'post_pos': focal_pos, 'timeline_len': c['n_posts'], 'declared_n': c['declared_n'], 'target_factor': c['target_factor'], 'is_target_post': p['is_target'], 'exemplar_row_ids': list(c['exemplar_row_ids']), 'bundle_id': c['bundle_id'], 'prompt_version': c['prompt_version'], 'model': c['model'], 'batch_id': c['batch_id'], 'target_schema': target_schema, 'noise_seed': f"{SC.SYNTH_SEED}|noise|{c['su_id']}|{focal_pos}"}}
            pool_records[c['pool']].append(rec)
            provenance.append({'row_id': row_id, 'bundle_id': c['bundle_id'], 'pool': c['pool'], 'su_id': c['su_id'], 'focal_pos': focal_pos, 'pos_label': list(pos_label), 'cover': [list(r) for r in ranges], 'timeline_texts': texts, 'pre_noise_text': p.get('pre_noise_text', ''), 'persona_note': c.get('persona_note', ''), 'target_factor': c['target_factor'], 'prompt_version': c['prompt_version'], 'model': c['model']})
    return (pool_records, provenance)
