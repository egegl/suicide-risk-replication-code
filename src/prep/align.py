from dataclasses import dataclass, field
from . import constants as C
from . import textnorm

@dataclass
class AlignedSpan:
    text_gold: str
    matched: bool
    rung: str
    raw_start: int = -1
    raw_end: int = -1
    raw_slice: str = ''
    n_occurrences: int = 0
    all_starts: list = field(default_factory=list)
    parent_idx: int = -1

def _find_all(hay: str, needle: str) -> list[int]:
    out, start = ([], 0)
    while True:
        i = hay.find(needle, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1

class NormCache:

    def __init__(self, raw: str):
        self.raw = raw
        self.lower = raw.lower()
        assert len(self.lower) == len(raw)
        self.norm, self.starts, self.ends = textnorm.normalize_with_map(raw)
        self.snorm, self.sstarts, self.sends = textnorm.normalize_with_map(raw, charmap=False)

def _find_span(span: str, nc: NormCache) -> tuple[str, int, int, list[int]] | None:
    hits = _find_all(nc.raw, span)
    if hits:
        return ('exact', hits[0], hits[0] + len(span), hits)
    low = span.lower()
    hits = _find_all(nc.lower, low)
    if hits:
        return ('casefold', hits[0], hits[0] + len(low), hits)
    norm_span = textnorm.normalize(span)
    if norm_span:
        hits = _find_all(nc.norm, norm_span)
        if hits:
            rs = [textnorm.raw_slice(nc.raw, nc.starts, nc.ends, h, h + len(norm_span)) for h in hits]
            return ('normalized', rs[0][0], rs[0][1], [s for s, _ in rs])
        stripped = C.EDGE_PUNCT_RE.sub('', norm_span).strip()
        if stripped and stripped != norm_span:
            hits = _find_all(nc.norm, stripped)
            if hits:
                rs = [textnorm.raw_slice(nc.raw, nc.starts, nc.ends, h, h + len(stripped)) for h in hits]
                return ('trimmed', rs[0][0], rs[0][1], [s for s, _ in rs])
    return None

def _n_occurrences(span: str, nc: NormCache) -> int:
    return len(_find_all(nc.lower, span.lower()))

def align_span(span: str, nc: NormCache) -> list[AlignedSpan]:
    hit = _find_span(span, nc)
    if hit:
        rung, s, e, all_starts = hit
        return [AlignedSpan(text_gold=span, matched=True, rung=rung, raw_start=s, raw_end=e, raw_slice=nc.raw[s:e], n_occurrences=max(_n_occurrences(span, nc), len(all_starts)), all_starts=all_starts)]
    if C.GAP_RE.search(span):
        parent = AlignedSpan(text_gold=span, matched=False, rung='split')
        children = []
        for frag in (f.strip() for f in C.GAP_RE.split(span)):
            if not frag:
                continue
            frag_hit = _find_span(frag, nc)
            if frag_hit:
                rung, s, e, all_starts = frag_hit
                children.append(AlignedSpan(text_gold=frag, matched=True, rung=rung, raw_start=s, raw_end=e, raw_slice=nc.raw[s:e], n_occurrences=max(_n_occurrences(frag, nc), len(all_starts)), all_starts=all_starts, parent_idx=0))
        return [parent, *children]
    return [AlignedSpan(text_gold=span, matched=False, rung='unmatched')]

def official_match(pred_text: str, gold_text: str) -> bool:
    p, g = (textnorm.scorer_norm(pred_text), textnorm.scorer_norm(gold_text))
    return bool(p) and (p in g or g in p) and (len(p.split()) <= 3 * len(g.split()))

def _lcs(a: str, b: str) -> tuple[int, int]:
    best_start, best_len = (0, 0)
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best_len:
                    best_start, best_len = (i - v, v)
        prev = cur
    return (best_start, best_len)

def scorer_matchable_slice(gold_text: str, nc: NormCache) -> tuple[int, int] | None:
    g = textnorm.scorer_norm(gold_text)
    if not g:
        return None
    A = nc.snorm
    ia, L = _lcs(A, g)
    if L == 0:
        return None
    start, end = (ia, ia + L)
    if start > 0 and A[start - 1] != ' ':
        nxt = A.find(' ', start, end)
        if nxt == -1:
            return None
        start = nxt + 1
    if end < len(A) and A[end] != ' ' and (end == 0 or A[end - 1] != ' '):
        sp = A.rfind(' ', start, end)
        if sp == -1:
            return None
        end = sp
    while start < end and A[start] == ' ':
        start += 1
    while end > start and A[end - 1] == ' ':
        end -= 1
    if start >= end:
        return None
    sub = A[start:end]
    if len(sub.split()) < 2 and sub != g:
        return None
    s, e = textnorm.raw_slice(nc.raw, nc.sstarts, nc.sends, start, end)
    if not official_match(nc.raw[s:e], gold_text):
        return None
    return (s, e)

def align_generated(text: str, nc: NormCache) -> tuple[int, int] | None:
    hit = _try_r0_to_r3(text, nc)
    if hit is None:
        return None
    _, s, e, _ = hit
    return (s, e)
