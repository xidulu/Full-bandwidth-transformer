"""Math-Verify reward grading shared by online training and GSM8K checks."""
from functools import lru_cache
from importlib.metadata import version

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException


class MathVerifier:
    def __init__(self, parse_timeout=5, verify_timeout=5):
        if min(parse_timeout, verify_timeout) <= 0:
            raise ValueError('Verifier timeouts must be positive')
        self.parse_timeout = parse_timeout
        self.verify_timeout = verify_timeout
        # Let Math-Verify extract final-answer expressions/boxes, without treating
        # an arbitrary intermediate number in the reasoning as a final answer.
        self.prediction_config = [
            LatexExtractionConfig(try_extract_without_anchor=False),
            ExprExtractionConfig(try_extract_without_anchor=False),
        ]

    def metadata(self):
        return {
            'package': 'math-verify', 'version': version('math-verify'),
            'latex2sympy2_extended_version': version('latex2sympy2-extended'),
            'antlr4_runtime_version': version('antlr4-python3-runtime'),
            'sympy_version': version('sympy'),
            'parse_timeout_seconds': self.parse_timeout,
            'verify_timeout_seconds': self.verify_timeout,
            'gold_extraction': 'LatexExtractionConfig on $reference$',
            'prediction_extraction': 'LatexExtractionConfig + ExprExtractionConfig; anchors required',
            'fallback_mode': 'no_fallback', 'extraction_mode': 'first_match',
            'strict': True, 'truncated_training_reward': 0.0,
            'evaluation_verifier': 'math-verify',
        }

    @lru_cache(maxsize=4096)
    def parse_gold(self, reference):
        gold = parse(
            f'${reference}$', extraction_config=[LatexExtractionConfig()],
            fallback_mode='no_fallback', extraction_mode='first_match',
            parsing_timeout=self.parse_timeout, raise_on_error=True,
        )
        if not gold:
            # A broken reference/dependency is a data/setup error, not a wrong
            # prediction: do not silently turn every response into zero reward.
            raise ValueError(f'Math-Verify could not parse reference {reference!r}')
        return gold

    def grade(self, text, reference, completed=True):
        gold = self.parse_gold(str(reference))
        prediction = []
        error = None
        correct = False
        try:
            prediction = parse(
                text, extraction_config=self.prediction_config,
                fallback_mode='no_fallback', extraction_mode='first_match',
                parsing_timeout=self.parse_timeout, raise_on_error=True,
            )
            if completed and prediction:
                correct = bool(verify(gold, prediction, strict=True,
                                      timeout_seconds=self.verify_timeout, raise_on_error=True))
        except (Exception, TimeoutException) as exc:
            # Math-Verify timeouts inherit BaseException. Catch them explicitly
            # while letting KeyboardInterrupt/SystemExit terminate the process.
            error = f'{type(exc).__name__}: {exc}'
        # Formatting is diagnostic: a SymPy printer failure must not crash a
        # worker or change a reward that Math-Verify already computed.
        prediction_text = []
        for item in prediction:
            try:
                prediction_text.append(str(item))
            except (Exception, TimeoutException) as exc:
                prediction_text.append(f'<unprintable {type(item).__name__}>')
                formatting_error = f'prediction serialization: {type(exc).__name__}: {exc}'
                error = f'{error}; {formatting_error}' if error else formatting_error
        return {'correct': correct, 'parsed': bool(prediction),
                'prediction': prediction_text, 'error': error}
