import re
from . import constants as C

def normalize_with_map(raw: str, charmap: bool=True) -> tuple[str, list[int], list[int]]:
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = (0, len(raw))
    while i < n:
        ch = raw[i]
        mapped = C.CHAR_MAP.get(ch, ch) if charmap else ch
        if mapped.isspace() or ch.isspace():
            j = i
            while j < n and (C.CHAR_MAP.get(raw[j], raw[j]).isspace() or raw[j].isspace()):
                j += 1
            if out:
                out.append(' ')
                starts.append(i)
                ends.append(j)
            i = j
            continue
        lowered = mapped.lower()
        if len(lowered) != len(mapped):
            lowered = mapped
        for c in lowered:
            out.append(c)
            starts.append(i)
            ends.append(i + 1)
        i += 1
    while out and out[-1] == ' ':
        out.pop()
        starts.pop()
        ends.pop()
    return (''.join(out), starts, ends)

def normalize(raw: str) -> str:
    return normalize_with_map(raw)[0]

def scorer_norm(s: str) -> str:
    return re.sub('\\s+', ' ', s.lower().strip())

def raw_slice(raw: str, starts: list[int], ends: list[int], a: int, b: int) -> tuple[int, int]:
    assert 0 <= a < b <= len(starts)
    while a < b - 1 and raw[starts[a]].isspace():
        a += 1
    while b - 1 > a and raw[starts[b - 1]].isspace():
        b -= 1
    return (starts[a], ends[b - 1])
