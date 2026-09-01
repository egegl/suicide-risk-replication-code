import re
from . import constants as C
_TOKEN_RE = re.compile('\\S+')

def head_tail(raw: str, budget_tokens: int=C.FOCAL_BUDGET_TOKENS, cover: list[tuple[int, int]] | None=None) -> tuple[str, list[tuple[int, int]], bool]:
    tokens = list(_TOKEN_RE.finditer(raw))
    if len(tokens) <= budget_tokens:
        return (raw, [(0, len(raw))], False)
    head_n = int(budget_tokens * C.HEAD_FRAC)
    tail_n = budget_tokens - head_n
    ranges = [(0, tokens[head_n - 1].end()), (tokens[len(tokens) - tail_n].start(), len(raw))]
    if cover and (not any((span_inside(ranges, s, e) for s, e in cover))):
        c_start = min((s for s, _ in cover))
        c_end = max((e for _, e in cover))
        ti = next((i for i, t in enumerate(tokens) if t.end() > c_start))
        tj = max((i for i, t in enumerate(tokens) if t.start() < c_end))
        pad = 4
        mid_budget = min(tj - ti + 1 + 2 * pad, budget_tokens // 2)
        mi = max(ti - pad, 0)
        mj = min(mi + mid_budget - 1, len(tokens) - 1)
        rest = budget_tokens - (mj - mi + 1)
        head_n = max(rest // 2, 1)
        tail_n = max(rest - head_n, 1)
        ranges = sorted([(0, tokens[head_n - 1].end()), (tokens[mi].start(), tokens[mj].end()), (tokens[len(tokens) - tail_n].start(), len(raw))])
        merged = [list(ranges[0])]
        for s, e in ranges[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        ranges = [(s, e) for s, e in merged]
    kept = C.TRUNCATION_MARKER.join((raw[s:e] for s, e in ranges))
    return (kept, ranges, True)

def span_inside(kept_ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    return any((start >= s and end <= e for s, e in kept_ranges))
