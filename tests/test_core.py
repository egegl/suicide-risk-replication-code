import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import scorer
from prep import constants as PC
from train import ensemble, specs, thresholds


def test_composite_formula():
    gold = {'x': {'risk': 'Indicator', 'spans': [], 'factors': []}}
    pred = {'x': {'risk': 'Indicator', 'spans': [], 'factors': []}}
    result = scorer.score(pred, gold)
    expected = 0.4 * result['risk_wf1'] + 0.3 * result['phrase_f1'] + 0.3 * result['macro_f1']
    assert result['composite'] == expected


def test_ensemble_averages_factor_probabilities():
    probs1 = {factor: 0.2 for factor in PC.FACTORS_24}
    probs2 = {factor: 0.8 for factor in PC.FACTORS_24}
    row1 = {'row_id': 'x', 'fold': 0, 'risk': 'Ideation', 'spans': ['want to die'], 'factors': [], 'p_true': probs1}
    row2 = {'row_id': 'x', 'fold': 0, 'risk': 'Ideation', 'spans': ['want to die'], 'factors': [], 'p_true': probs2}
    merged = ensemble.merge_rows([row1], [row2])[0]
    assert merged['risk'] == 'Ideation'
    assert set(merged['p_true'].values()) == {0.5}


def test_thresholds_do_not_use_held_out_fold():
    factor = PC.FACTORS_24[0]
    rows = []
    gold = {}
    for fold, probability, present in [(0, 0.99, False), (1, 0.8, True), (2, 0.2, False)]:
        row_id = str(fold)
        probs = {name: 0.0 for name in PC.FACTORS_24}
        probs[factor] = probability
        rows.append({'row_id': row_id, 'fold': fold, 'p_true': probs})
        gold[row_id] = {'factors': [factor] if present else []}
    taus = thresholds.fit_thresholds(rows, gold, exclude_fold=0)
    assert taus[factor] < 0.99


def test_spec_count(tmp_path):
    sft = tmp_path / 'data' / 'processed' / 'sft'
    sft.mkdir(parents=True)
    record = '{"weight": 1.0}\n'
    for fold in range(PC.N_FOLDS):
        (sft / f'train_fold{fold}.jsonl').write_text(record)
    (sft / 'full_train.jsonl').write_text(record)
    built = specs.build_specs(tmp_path)
    assert len(built) == 36
    assert {spec['arm'] for spec in built} == {'real_only', 'synthetic', 'repeated_real'}
