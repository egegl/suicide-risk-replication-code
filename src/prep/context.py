from . import constants as C
from . import truncate

def _neighbor(text: str) -> str:
    text = ' '.join(text.split())
    if len(text) > C.CTX_NEIGHBOR_CHARS:
        text = text[:C.CTX_NEIGHBOR_CHARS].rstrip() + ' [...]'
    return text

def build_input(user_posts: list[str], focal_pos: int, cover: list[tuple[int, int]] | None=None, pos_label: tuple[int, int] | None=None, instruction: str | None=None) -> tuple[str, list[tuple[int, int]], bool]:
    n = len(user_posts)
    pos_i, pos_n = pos_label if pos_label is not None else (focal_pos + 1, n)
    focal_kept, kept_ranges, is_trunc = truncate.head_tail(user_posts[focal_pos], cover=cover)
    if instruction is None:
        instruction = C.SFT_INSTRUCTION
    lines = [instruction, '', f"Post {pos_i} of {pos_n} in this user's timeline."]
    for off in range(-C.CTX_N_PREV, 0):
        i = focal_pos + off
        if i >= 0:
            lines.append(f'[Earlier post {off}] {_neighbor(user_posts[i])}')
    lines.append('[FOCAL POST]')
    lines.append(focal_kept)
    for off in range(1, C.CTX_N_NEXT + 1):
        i = focal_pos + off
        if i < n:
            lines.append(f'[Later post +{off}] {_neighbor(user_posts[i])}')
    return ('\n'.join(lines), kept_ranges, is_trunc)
