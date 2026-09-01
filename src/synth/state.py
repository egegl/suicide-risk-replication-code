import gzip
import hashlib
import json
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def _atomic(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    write_fn(tmp)
    tmp.replace(path)

def atomic_write_jsonl(path: Path, records: list[dict]) -> None:

    def _w(tmp: Path):
        with open(tmp, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    _atomic(path, _w)

def atomic_write_text(path: Path, text: str) -> None:
    _atomic(path, lambda tmp: tmp.write_text(text, encoding='utf-8'))

def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def atomic_write_jsonl_gz(path: Path, lines: list[str]) -> None:

    def _w(tmp: Path):
        with open(tmp, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as g:
            for line in lines:
                g.write((line.rstrip('\n') + '\n').encode('utf-8'))
    _atomic(path, _w)

class Manifest:

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / 'manifest.json'
        self.data: dict = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))

    def _rel(self, path: Path) -> str:
        p = Path(path)
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            import os
            return os.path.relpath(p) if p.is_absolute() else str(p)

    def _resolve(self, rel: str) -> Path:
        p = self.root / rel
        if p.exists():
            return p
        q = Path(rel)
        return q if q.exists() else p

    def record(self, stage: str, config: dict, inputs: list[Path], outputs: list[Path]) -> None:
        self.data[stage] = {'config': config, 'inputs': {self._rel(p): sha256_file(Path(p)) for p in inputs}, 'outputs': {self._rel(p): sha256_file(Path(p)) for p in outputs}}
        atomic_write_json(self.path, self.data)

    def unchanged(self, stage: str, config: dict, inputs: list[Path]) -> bool:
        rec = self.data.get(stage)
        if rec is None or rec['config'] != config:
            return False
        cur_inputs = {self._rel(p): sha256_file(Path(p)) for p in inputs if Path(p).exists()}
        if rec['inputs'] != cur_inputs:
            return False
        for rel, sha in rec['outputs'].items():
            p = self.root / rel
            if not p.exists() or sha256_file(p) != sha:
                return False
        return True

    def inputs_unchanged(self, stage: str) -> bool:
        rec = self.data.get(stage)
        if rec is None:
            return False
        for rel, sha in rec['inputs'].items():
            p = self._resolve(rel)
            if not p.exists() or sha256_file(p) != sha:
                return False
        return True

    def check_upstream(self, stage: str, upstream: str) -> None:
        rec = self.data.get(upstream)
        assert rec is not None, f'{stage}: upstream stage {upstream!r} has not run'
        for rel, sha in rec['outputs'].items():
            p = self.root / rel
            assert p.exists(), f'{stage}: upstream output missing: {rel}'
            actual = sha256_file(p)
            assert actual == sha, f'{stage}: upstream output {rel} changed since {upstream!r} ran (recorded {sha[:12]}, found {actual[:12]}) — rerun {upstream!r} or --force'
