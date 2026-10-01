import pytest
from math_verify.errors import TimeoutException
from verifier import MathVerifier


@pytest.mark.parametrize('text, gold, correct', [
    ('Reasoning 37\nAnswer: 42', '42', True),
    ('Answer: $1,024$.', '1024', True),
    ('Answer: -3', '-3', True),
    (r'Thus \boxed{\frac{84}{2}}', '42', True),
    (r'Answer: $\sqrt{1764}$', '42', True),
    ('Answer: 3/2', '1.5', True),
    ('Answer: 42.0', '42', True),
    ('Answer: 2\nAnswer: 7', '7', True),
    ('Answer: 2\nAnswer: 7', '2', False),
    ('I tried 42 but do not know', '42', False),
    ('No solution.', '42', False),
    (r'Answer: \boxed{\frac{', '42', False),
    ('Answer: 3.5', '3', False),
    (r'Answer: $\sqrt{1765}$', '42', False),
])
def test_symbolic_grading(text, gold, correct):
    assert MathVerifier().grade(text, gold)['correct'] is correct


def test_truncated_response_gets_no_reward():
    result = MathVerifier().grade(r'Answer: \boxed{42}', '42', completed=False)
    assert result['parsed']
    assert not result['correct']


@pytest.mark.parametrize('stage', ['parse', 'verify'])
@pytest.mark.parametrize('error_type', [TimeoutError, TimeoutException])
def test_prediction_timeouts_fail_closed(monkeypatch, stage, error_type):
    import verifier as module
    grader = MathVerifier()
    grader.parse_gold('42')
    def timeout(*args, **kwargs):
        raise error_type('test timeout')
    monkeypatch.setattr(module, stage, timeout)
    result = grader.grade('Answer: 42', '42')
    assert not result['correct']
    assert result['error'] == f'{error_type.__name__}: test timeout'


def test_invalid_reference_is_a_data_error():
    with pytest.raises(ValueError, match='reference'):
        MathVerifier().grade('Answer: 42', '')


def test_versions_and_extraction_are_recorded():
    config = MathVerifier().metadata()
    assert config['package'] == 'math-verify'
    assert config['version']
    assert config['fallback_mode'] == 'no_fallback'
    assert config['evaluation_verifier'] == 'math-verify'


def test_gsm8k_evaluation_uses_symbolic_verifier(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import main
    from fbt_experiments import evaluate_checkpoint
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    monkeypatch.setattr(evaluate_checkpoint, 'load_gsm8k_rows', lambda *a: [{'context': 'q', 'answer': '42'}])
    monkeypatch.setattr(evaluate_checkpoint, 'build_gsm8k_chat_prompt_ids', lambda *a: ([1], 'q'))
    monkeypatch.setattr(main, 'collect', lambda *a, **kw: ([[2, 9]], [True]))
    model = SimpleNamespace(eval=lambda: None, get_device=lambda: 'cpu')
    tokenizer = SimpleNamespace(decode=lambda ids: r'Answer: $\frac{84}{2}$')
    output = tmp_path / 'eval.jsonl'
    metrics = main.evaluate(model, None, tokenizer, 1, output)
    assert metrics['eval/gsm8k_correct'] == 1
    record = json.loads(output.read_text())
    assert record['verifier'] == 'math-verify'
    assert record['correct']


def test_real_math_verify_signal_timeout_is_caught(monkeypatch):
    import time
    import verifier as module
    from math_verify.utils import timeout
    grader = MathVerifier()
    grader.parse_gold('42')
    @timeout(timeout_seconds=1)
    def slow_verify(*args, **kwargs):
        time.sleep(3)
        return True
    monkeypatch.setattr(module, 'verify', slow_verify)
    result = grader.grade('Answer: 42', '42')
    assert not result['correct']
    assert result['error'].startswith('TimeoutException:')


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_process_interrupts_are_not_swallowed(monkeypatch, error_type):
    import verifier as module
    grader = MathVerifier()
    grader.parse_gold('42')
    def interrupt(*args, **kwargs):
        raise error_type()
    monkeypatch.setattr(module, 'verify', interrupt)
    with pytest.raises(error_type):
        grader.grade('Answer: 42', '42')


@pytest.mark.parametrize('verdict', [True, False, 'error'])
def test_unprintable_prediction_does_not_crash_or_change_reward(monkeypatch, verdict):
    import json
    import verifier as module
    grader = MathVerifier()
    grader.parse_gold('42')

    class UnprintablePrediction:
        def __str__(self):
            raise TypeError('Invalid NaN comparison')

    monkeypatch.setattr(module, 'parse', lambda *a, **kw: [UnprintablePrediction(), 42])

    def verify(*args, **kwargs):
        if verdict == 'error':
            raise ValueError('symbolic verification failed')
        return verdict

    monkeypatch.setattr(module, 'verify', verify)
    result = grader.grade('Answer: malformed symbolic set', '42')
    assert result['correct'] is (verdict is True)
    assert result['parsed']
    assert result['prediction'] == ['<unprintable UnprintablePrediction>', '42']
    assert 'prediction serialization: TypeError: Invalid NaN comparison' in result['error']
    if verdict == 'error':
        assert result['error'].startswith('ValueError: symbolic verification failed; ')
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_prediction_formatting_preserves_process_interrupts(monkeypatch, error_type):
    import verifier as module
    grader = MathVerifier()
    grader.parse_gold('42')

    class InterruptedPrediction:
        def __str__(self):
            raise error_type()

    monkeypatch.setattr(module, 'parse', lambda *a, **kw: [InterruptedPrediction()])
    monkeypatch.setattr(module, 'verify', lambda *a, **kw: False)
    with pytest.raises(error_type):
        grader.grade('Answer: interrupted', '42')
