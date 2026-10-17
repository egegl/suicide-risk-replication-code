import json

from pathlib import Path

from synth import state

def _mean_weight(path: Path) -> float:
    ws = [float(r['weight']) for r in state.read_jsonl(path)]
    return sum(ws) / len(ws)

def load_specs(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def resolve(specs: list[dict], index: int | None=None, run_id: str | None=None) -> dict:
    if (index is None) == (run_id is None):
        raise SystemExit('resolve: pass exactly one of index / run_id')
    if index is not None:
        if not 0 <= index < len(specs):
            raise SystemExit(f'resolve: index {index} out of range 0..{len(specs) - 1}')
        spec = specs[index]
        assert spec['index'] == index, (spec['index'], index)
        return spec
    for spec in specs:
        if spec['run_id'] == run_id:
            return spec
    raise SystemExit(f'resolve: unknown run_id {run_id!r}')
