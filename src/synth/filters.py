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
