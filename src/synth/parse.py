import json
from prep import align, textnorm
from . import constants as SC
_REFUSAL_HINTS = ("i can't help", 'i cannot help', "i can't assist", 'i cannot assist', "i'm not able to help", 'i am not able to help', "i won't", 'cannot provide')

def _extract_body(line: dict) -> tuple[dict | None, str]:
    if line.get('error'):
        return (None, f"api_error:{line['error']}")
    resp = line.get('response')
    if not resp or resp.get('status_code') != 200:
        return (None, f"status:{(resp.get('status_code') if resp else 'none')}")
    body = resp.get('body') or {}
    if 'choices' in body:
        text, why = _chat_text(body)
        if why:
            return (None, why)
    else:
        if body.get('status') == 'incomplete':
            reason = (body.get('incomplete_details') or {}).get('reason', 'incomplete')
            return (None, f'incomplete:{reason}')
        text, refused = _output_text(body)
        if refused:
            return (None, 'refusal')
        if text is None:
            return (None, 'no_output_text')
    low = text.lower()
    if any((h in low for h in _REFUSAL_HINTS)) and '"posts"' not in text:
        return (None, 'refusal_phrase')
    try:
        return (json.loads(text), '')
    except json.JSONDecodeError as e:
        return (None, f'json_decode:{e}')

def _output_text(body: dict) -> tuple[str | None, bool]:
    if isinstance(body.get('output_text'), str):
        return (body['output_text'], False)
    parts = []
    for item in body.get('output', []):
        if item.get('type') == 'reasoning':
            continue
        for c in item.get('content', []):
            ctype = c.get('type')
            if ctype == 'refusal':
                return (None, True)
            if ctype in ('output_text', 'text') and 'text' in c:
                parts.append(c['text'])
    return (''.join(parts) if parts else None, False)

def _chat_text(body: dict) -> tuple[str | None, str]:
    choice = (body.get('choices') or [{}])[0]
    msg = choice.get('message') or {}
    if msg.get('refusal') or choice.get('finish_reason') == 'content_filter':
        return (None, 'refusal')
    if choice.get('finish_reason') == 'length':
        return (None, 'incomplete:length')
    text = msg.get('content')
    if not isinstance(text, str) or not text:
        return (None, 'no_output_text')
    return (text, '')

def _heal_evidence(phrase: str, post_text: str) -> str | None:
    nc = align.NormCache(post_text)
    hit = align.align_generated(phrase, nc)
    if hit is None:
        return None
    s, e = hit
    sliced = post_text[s:e]
    if ';' in sliced or '\n' in sliced:
        return None
    if len(textnorm.scorer_norm(sliced).split()) > SC.SPAN_TOKENS_MAX:
        return None
    return sliced

def parse_response(line: dict, bundle: dict, batch_id: str, model: str) -> tuple[dict | None, dict | None]:

    def reject(reason):
        return (None, {'bundle_id': bundle['bundle_id'], 'pool': bundle['pool'], 'target_factor': bundle['target_factor'], 'reason': reason})
    body = (line.get('response') or {}).get('body') or {}
    body_model = body.get('model') or ''
    if body_model and body_model != model:
        return reject(f'model_mismatch:{body_model}')
    if not body_model and 'choices' in body:
        return reject('model_mismatch:missing')
    timeline, why = _extract_body(line)
    if why:
        return reject(why)
    posts_out = timeline.get('posts')
    if not isinstance(posts_out, list) or len(posts_out) != bundle['n_posts']:
        return reject(f"post_count:{(len(posts_out) if isinstance(posts_out, list) else 'na')}")
    by_pos = {p.get('pos'): p for p in posts_out if isinstance(p, dict)}
    healed_posts = []
    align_fail = 0
    for spec in bundle['posts']:
        gen = by_pos.get(spec['pos'])
        if gen is None or not isinstance(gen.get('post'), str) or (not gen['post'].strip()):
            return reject(f"missing_post_pos:{spec['pos']}")
        post_text = gen['post']
        if len(post_text.split()) > SC.POST_TOKENS_REJECT:
            return reject(f"post_too_long_pos:{spec['pos']}")
        if spec['risk'] == 'Indicator':
            evidence = []
        else:
            raw_ev = [e for e in gen.get('evidence', []) if isinstance(e, str) and e.strip()]
            healed = []
            for e in raw_ev:
                h = _heal_evidence(e, post_text)
                if h is None:
                    align_fail += 1
                    continue
                healed.append(h)
            lo = max(1, spec['n_spans'] - 1)
            hi = min(spec['n_spans'] + 1, SC.SPAN_COUNT_MAX)
            if len(healed) < lo:
                return reject(f"too_few_alignable_spans_pos:{spec['pos']}:{len(healed)}<{lo}")
            evidence = healed[:hi]
        healed_posts.append({'pos': spec['pos'], 'is_target': spec['is_target'], 'emit_record': spec['emit_record'], 'risk': spec['risk'], 'factors': list(spec['factors']), 'post_text': post_text, 'evidence': evidence})
    candidate = {'bundle_id': bundle['bundle_id'], 'pool': bundle['pool'], 'target_factor': bundle['target_factor'], 'su_id': bundle['su_id'], 'declared_n': bundle['declared_n'], 'n_posts': bundle['n_posts'], 'posts': healed_posts, 'exemplar_row_ids': list(bundle['exemplar_row_ids']), 'prompt_version': bundle['bundle_id'].split('-k', 1)[0], 'model': body_model or model, 'batch_id': batch_id, 'persona_note': timeline.get('persona_note', ''), 'qc': {'align_fail': align_fail, 'finish': 'completed'}}
    return (candidate, None)

def parse_all(raw_by_pool: dict[str, list[dict]], bundles: list[dict], batch_id_by_pool: dict[str, str], model: str) -> tuple[list[dict], list[dict]]:
    by_id = {b['bundle_id']: b for b in bundles}
    candidates, rejects = ([], [])
    for pool, lines in sorted(raw_by_pool.items()):
        batch_id = batch_id_by_pool.get(pool, '')
        seen = set()
        for line in lines:
            cid = line.get('custom_id')
            seen.add(cid)
            bundle = by_id.get(cid)
            if bundle is None:
                rejects.append({'bundle_id': cid, 'reason': 'unknown_custom_id'})
                continue
            cand, rej = parse_response(line, bundle, batch_id, model)
            (candidates if cand else rejects).append(cand or rej)
        for b in bundles:
            if b['pool'] == pool and b['bundle_id'] not in seen:
                rejects.append({'bundle_id': b['bundle_id'], 'pool': pool, 'target_factor': b['target_factor'], 'reason': 'status:missing_response'})
    candidates.sort(key=lambda c: c['bundle_id'])
    rejects.sort(key=lambda r: r.get('bundle_id') or '')
    return (candidates, rejects)
