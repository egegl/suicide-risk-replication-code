import math
from prep import constants as PC

def risk_class_probs(row: dict) -> dict[str, float]:
    rl = row.get('risk_logprobs') or []
    if rl and rl[0]:
        mass = {c: 0.0 for c in PC.RISK_CLASSES}
        for tok, lp in rl[0]:
            t = tok.lstrip()
            if t.startswith('"'):
                t = t[1:]
            if not t:
                continue
            hits = [c for c in PC.RISK_CLASSES if c.startswith(t) or t.startswith(c)]
            if len(hits) == 1:
                mass[hits[0]] += math.exp(lp)
        denom = sum(mass.values())
        if denom > 0.0:
            return {c: m / denom for c, m in mass.items()}
    return {c: 1.0 if row['risk'] == c else 0.0 for c in PC.RISK_CLASSES}

def _merge_p_true(a: dict, b: dict) -> dict[str, float]:
    pa, pb = (a.get('p_true') or {}, b.get('p_true') or {})
    if not pa:
        return dict(pb)
    if not pb:
        return dict(pa)
    return {f: (pa[f] + pb[f]) / 2.0 for f in PC.FACTORS_24}

def merge_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    by_id_b = {r['row_id']: r for r in rows_b}
    assert len(by_id_b) == len(rows_b), 'duplicate row_id in rows_b'
    missing = [r['row_id'] for r in rows_a if r['row_id'] not in by_id_b]
    assert not missing, f'rows_b misses row_ids from rows_a, e.g. {missing[:5]}'
    assert len(rows_a) == len(rows_b), (len(rows_a), len(rows_b))
    out = []
    for a in rows_a:
        b = by_id_b[a['row_id']]
        assert a.get('fold') == b.get('fold'), (a['row_id'], a.get('fold'), b.get('fold'))
        a_fb, b_fb = (not a.get('p_true'), not b.get('p_true'))
        if a_fb or b_fb:
            src = b if a_fb and (not b_fb) else a
            new = dict(src)
            new['spans'] = list(src['spans'])
            out.append(new)
            continue
        pa, pb = (risk_class_probs(a), risk_class_probs(b))
        avg = {c: (pa[c] + pb[c]) / 2.0 for c in PC.RISK_CLASSES}
        best = max(avg.values())
        risk = next((c for c in PC.RISK_CLASSES if avg[c] == best))
        spans = list(a['spans'])
        if risk != 'Indicator' and (not spans):
            spans = list(b['spans'])
            if not spans:
                risk = a['risk']
                spans = list(a['spans'])
        if risk == 'Indicator':
            spans = []
        new = dict(a)
        new.update(risk=risk, spans=spans, p_true=_merge_p_true(a, b), risk_probs=avg)
        out.append(new)
    return out
