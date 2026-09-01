import pandas as pd
from prep import dedup, textnorm
from . import constants as SC

class RealIndex:

    def __init__(self, real_posts: list[str]):
        self.keys = {dedup.near_dup_key(p) for p in real_posts}
        self.ngrams: set = set()
        for p in real_posts:
            toks = textnorm.normalize(p).split()
            for i in range(len(toks) - SC.OVERLAP_NGRAM + 1):
                self.ngrams.add(tuple(toks[i:i + SC.OVERLAP_NGRAM]))

    def hits(self, post_text: str) -> str | None:
        if dedup.near_dup_key(post_text) in self.keys:
            return 'real_near_dup'
        toks = textnorm.normalize(post_text).split()
        for i in range(len(toks) - SC.OVERLAP_NGRAM + 1):
            if tuple(toks[i:i + SC.OVERLAP_NGRAM]) in self.ngrams:
                return 'real_8gram_overlap'
        return None

def load_real_posts(processed) -> list[str]:
    posts = pd.read_parquet(processed / 'posts.parquet')
    return posts['post_raw'].tolist()

def filter_all(candidates: list[dict], index: RealIndex) -> tuple[list[dict], list[dict]]:
    accepted, rejects = ([], [])
    seen_synth: set = set()
    for c in sorted(candidates, key=lambda c: c['bundle_id']):
        reason, bad_pos = (None, None)
        for p in c['posts']:
            hit = index.hits(p['post_text'])
            if hit:
                reason, bad_pos = (hit, p['pos'])
                break
            key = dedup.near_dup_key(p['post_text'])
            if key in seen_synth:
                reason, bad_pos = ('synth_near_dup', p['pos'])
                break
        if reason:
            rejects.append({'bundle_id': c['bundle_id'], 'pool': c['pool'], 'reason': reason, 'pos': bad_pos})
            continue
        for p in c['posts']:
            seen_synth.add(dedup.near_dup_key(p['post_text']))
        accepted.append(c)
    return (accepted, rejects)
