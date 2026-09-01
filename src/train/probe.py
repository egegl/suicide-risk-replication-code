import math
import re
from prep import constants as PC
from prep import target as PT
_PROBE_CONTEXTS = ('', ' ', '": ', 's": ')
_RISK_MARKER_RE = re.compile('"risk":\\s*"')

def _encode(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))

def _char_to_step(token_ids: list[int], tokenizer) -> tuple[str, list[int]]:
    lengths = [len(tokenizer.decode(token_ids[:i])) for i in range(len(token_ids) + 1)]
    full = tokenizer.decode(token_ids) if token_ids else ''
    steps: list[int] = []
    for i in range(len(token_ids)):
        steps.extend([i] * max(lengths[i + 1] - lengths[i], 0))
    if len(steps) < len(full) and token_ids:
        steps.extend([len(token_ids) - 1] * (len(full) - len(steps)))
    return (full, steps)

def build_probe(tokenizer) -> dict:
    sets: dict[str, set[int]] = {}
    for name, literal in (('TRUE_FIRST', 'true'), ('FALSE_FIRST', 'false')):
        ids_set: set[int] = set()
        for prefix in _PROBE_CONTEXTS:
            ids = _encode(tokenizer, prefix + literal)
            full, steps = _char_to_step(ids, tokenizer)
            p = full.find(literal)
            if p < 0 or p >= len(steps):
                continue
            ids_set.add(ids[steps[p]])
        if not ids_set:
            raise ValueError(f'probe: no context yields a first token for {literal!r}')
        sets[name] = ids_set
    overlap = sets['TRUE_FIRST'] & sets['FALSE_FIRST']
    if overlap:
        raise ValueError(f'probe first-token sets overlap on ids {sorted(overlap)} — first-token mass cannot separate true/false; redesign the probe')
    return sets

def factor_probs(token_ids: list[int], logprobs: list[dict], tokenizer, probe: dict) -> tuple[dict, dict]:
    full, steps = _char_to_step(token_ids, tokenizer)
    true_first, false_first = (probe['TRUE_FIRST'], probe['FALSE_FIRST'])
    flags = {'probe_low_conf': 0, 'probe_key_missing': 0}
    probs: dict[str, float] = {}
    anchor = full.find('"factors"')
    cursor = anchor if anchor >= 0 else 0
    for key in PC.FACTORS_24:
        pat = f'"{key}":'
        i = full.find(pat, cursor)
        if i < 0:
            probs[key] = 0.0
            flags['probe_key_missing'] += 1
            continue
        p = i + len(pat)
        while p < len(full) and full[p] in ' \t\r\n':
            p += 1
        if p >= len(steps):
            probs[key] = 0.0
            flags['probe_key_missing'] += 1
            continue
        cursor = p
        step = steps[p]
        t_sum = f_sum = 0.0
        if step < len(logprobs):
            for tid, lp in logprobs[step].items():
                if tid in true_first:
                    t_sum += math.exp(lp)
                elif tid in false_first:
                    f_sum += math.exp(lp)
        if t_sum + f_sum > 0.0:
            probs[key] = t_sum / (t_sum + f_sum)
        else:
            probs[key] = 1.0 if full.startswith('true', p) else 0.0
            flags['probe_low_conf'] += 1
    return (probs, {k: v for k, v in flags.items() if v})

def risk_logprobs(token_ids: list[int], logprobs: list[dict], tokenizer) -> list:
    full, steps = _char_to_step(token_ids, tokenizer)
    last = None
    for last in _RISK_MARKER_RE.finditer(full):
        pass
    if last is None:
        return []
    p = last.end()
    if p >= len(steps):
        return []
    s0 = steps[p]
    out = []
    for s in range(s0, min(s0 + 3, len(logprobs))):
        entries = sorted(logprobs[s].items(), key=lambda kv: (-kv[1], kv[0]))
        out.append([[tokenizer.decode([tid]), float(lp)] for tid, lp in entries])
    return out

def probe_self_test(tokenizer) -> dict:
    true_set = [PC.FACTORS_24[0], PC.FACTORS_24[5], PC.FACTORS_24[23]]
    text = PT.render_target('Ideation', ['want to die'], true_set, 'v2')
    ids = _encode(tokenizer, text)
    probe = build_probe(tokenizer)
    union = probe['TRUE_FIRST'] | probe['FALSE_FIRST']
    fabricated = []
    for tid in ids:
        if tid in probe['TRUE_FIRST']:
            alt = min(probe['FALSE_FIRST'])
        elif tid in probe['FALSE_FIRST']:
            alt = min(probe['TRUE_FIRST'])
        else:
            alt = tid + 1
            while alt in union or alt == tid:
                alt += 1
        fabricated.append({tid: -0.01, alt: -4.0})
    probs, flags = factor_probs(ids, fabricated, tokenizer, probe)
    assert not flags.get('probe_key_missing'), f'keys not located: {flags}'
    assert not flags.get('probe_low_conf'), f'unexpected low-confidence steps: {flags}'
    assert set(probs) == set(PC.FACTORS_24)
    for f in PC.FACTORS_24:
        want = f in true_set
        assert (probs[f] > 0.5) == want, (f, want, probs[f])
    rl = risk_logprobs(ids, fabricated, tokenizer)
    assert rl, 'risk value position not found'
    return probs
