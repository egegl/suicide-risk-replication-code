import json
from . import constants as C

def render_target(risk: str, evidence: list[str], factors: list[str], schema: str) -> str:
    assert risk in C.RISK_CLASSES, risk
    unknown = set(factors) - set(C.FACTORS_24)
    assert not unknown, f'non-taxonomy factors: {sorted(unknown)}'
    if schema == 'v1':
        return json.dumps({'risk': risk, 'evidence': list(evidence), 'factors': list(factors)}, ensure_ascii=False)
    if schema == 'v2':
        fset = set(factors)
        return json.dumps({'evidence': list(evidence), 'factors': {f: f in fset for f in C.FACTORS_24}, 'risk': risk}, ensure_ascii=False)
    raise ValueError(f'unknown target schema {schema!r}')

def true_factors(target: dict) -> list[str]:
    f = target['factors']
    if isinstance(f, dict):
        return [k for k in C.FACTORS_24 if f.get(k)]
    return list(f)

def instruction_for(schema: str) -> str:
    return C.SFT_INSTRUCTIONS[schema]
