import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import scorer
from prep import constants as PC
from train import ensemble, specs, thresholds
import submission

sys.path.insert(0, str(ROOT))
import predict


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


def test_unseen_input_builds_r9_timeline_context(tmp_path):
    source = tmp_path / 'hidden.csv'
    source.write_text(
        'row_id,anon_user_id,post_id,post\n'
        'H2,U1,1,second post\n'
        'H1,U1,0,first post\n'
        'H3,U2,0,only post\n', encoding='utf-8')
    examples = predict.load_examples(source)
    rows = predict.build_inference_rows(examples)
    by_id = {row['row_id']: row for row in rows}
    assert set(by_id) == {'H1', 'H2', 'H3'}
    assert "Post 1 of 2 in this user's timeline." in by_id['H1']['input']
    assert '[Later post +1] second post' in by_id['H1']['input']
    assert '[Earlier post -1] first post' in by_id['H2']['input']
    assert '[FOCAL POST]\nonly post' in by_id['H3']['input']


def test_frozen_r9_threshold_artifact_is_complete():
    payload, fitted = predict.load_thresholds(
        ROOT / 'artifacts' / 'r9_thresholds.json')
    assert set(fitted) == set(PC.FACTORS_24)
    assert payload['members'] == [
        {'name': 'seed1', 'run_id': 'baseline-full-s1', 'step': 160},
        {'name': 'seed2', 'run_id': 'baseline-full-s2', 'step': 120},
    ]
    assert payload['threshold_estimator']['bootstrap_draws'] == 200
    assert payload['threshold_estimator']['seed'] == 20260815


def test_unseen_submission_round_trip(tmp_path):
    examples = predict.load_examples(_write_hidden_fixture(tmp_path))
    preds = {
        'H1': {'risk': 'Indicator', 'spans': ['discard me'], 'factors': []},
        'H2': {'risk': 'Ideation', 'spans': ['want to die'],
               'factors': ['hopelessness']},
    }
    out = submission.write_submission(preds, examples, tmp_path / 'predictions.csv')
    written = out.read_text(encoding='utf-8')
    assert 'H1,Indicator,none,[]' in written
    assert 'H2,Ideation,want to die' in written


def _write_hidden_fixture(tmp_path):
    path = tmp_path / 'hidden.csv'
    path.write_text(
        'row_id,anon_user_id,post_id,post\n'
        'H1,U1,0,ordinary post\n'
        'H2,U2,0,I want to die tonight\n', encoding='utf-8')
    return path


def test_adapter_preflight_requires_both_files(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    try:
        predict.validate_adapter(adapter, 'test')
    except FileNotFoundError as error:
        assert 'adapter_model.safetensors' in str(error)
    else:
        raise AssertionError('incomplete adapter was accepted')


def test_r9_run_orchestrates_ensemble_thresholds_and_writer(tmp_path, monkeypatch):
    examples_path = _write_hidden_fixture(tmp_path)
    adapters = []
    for name in ('s1', 's2'):
        adapter = tmp_path / name
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
        (adapter / 'adapter_model.safetensors').write_bytes(b'test')
        adapters.append(adapter)

    class FakeEngine:
        max_model_len = 2560

        def __init__(self, *args, **kwargs):
            pass

        def get_tokenizer(self):
            return object()

    monkeypatch.setattr(predict.generate, 'Engine', FakeEngine)
    monkeypatch.setattr(predict.data_mod, 'assert_template_prefix', lambda tokenizer: None)
    monkeypatch.setattr(predict.probe_mod, 'build_probe', lambda tokenizer: {})
    monkeypatch.setattr(predict.probe_mod, 'probe_self_test', lambda tokenizer: None)

    def fake_member(engine, adapter, run_id, rows, raw_posts, tokenizer, factor_probe):
        factor_probs = {factor: 0.0 for factor in PC.FACTORS_24}
        factor_probs['hopelessness'] = 0.9
        return [
            {'row_id': 'H1', 'fold': -1, 'risk': 'Indicator', 'spans': [],
             'factors': [], 'p_true': factor_probs, 'risk_logprobs': []},
            {'row_id': 'H2', 'fold': -1, 'risk': 'Ideation',
             'spans': ['want to die'], 'factors': [],
             'p_true': factor_probs, 'risk_logprobs': []},
        ], {'retries': 0, 'fallbacks': 0}

    monkeypatch.setattr(predict, '_member_rows', fake_member)
    output = tmp_path / 'r9.csv'
    result = predict.run(SimpleNamespace(
        model='local-gemma', input=str(examples_path), output=str(output),
        thresholds=str(ROOT / 'artifacts' / 'r9_thresholds.json'),
        adapter_seed1=str(adapters[0]), adapter_seed2=str(adapters[1]),
        tensor_parallel_size=2, gpu_memory_utilization=0.92))
    assert result['system'] == 'R9'
    written = output.read_text(encoding='utf-8')
    assert "H2,Ideation,want to die,['hopelessness']" in written
