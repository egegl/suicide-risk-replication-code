import json
import re
from prep import align
from prep import constants as PC
from prep.target import true_factors
_FENCE_OPEN_RE = re.compile('^```[A-Za-z0-9_-]*[ \\t]*\\r?\\n?')
_FENCE_CLOSE_RE = re.compile('\\r?\\n?```$')

def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith('```'):
        t = _FENCE_OPEN_RE.sub('', t, count=1)
    if t.endswith('```'):
        t = _FENCE_CLOSE_RE.sub('', t, count=1)
    return t.strip()

def parse_generation(text: str, raw_post: str) -> dict | None:
    try:
        obj = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(obj, dict):
        return None
    risk = obj.get('risk')
    if risk not in PC.RISK_CLASSES:
        return None
    evidence = obj.get('evidence')
    if not isinstance(evidence, list):
        return None
    if not isinstance(obj.get('factors'), (dict, list)):
        return None
    factors = true_factors(obj)
    if not all((isinstance(f, str) and f in PC.FACTOR_INDEX for f in factors)):
        return None
    nc = align.NormCache(raw_post)
    spans: list[str] = []
    dropped = 0
    for phrase in evidence:
        if not isinstance(phrase, str) or not phrase.strip():
            dropped += 1
            continue
        hit = align.align_generated(phrase, nc)
        if hit is None:
            dropped += 1
        else:
            s, e = hit
            spans.append(nc.raw[s:e])
    return {'risk': risk, 'spans': spans, 'factors': list(factors), 'align_dropped': dropped}

def repair_and_flag(pred: dict) -> tuple[dict, dict]:
    out = dict(pred)
    flags: dict = {}
    if out['risk'] == 'Indicator':
        if out['spans']:
            out['spans'] = []
            flags['coupling_repaired'] = True
    elif not out['spans']:
        flags['coupling_violation'] = True
    return (out, flags)
