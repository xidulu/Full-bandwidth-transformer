from prepare_orz_data import ANSWER_INSTRUCTION, convert_rows, validate_reference


def row(question, answer):
    return [{'from': 'human', 'value': question},
            {'from': 'assistant', 'ground_truth': {'value': answer}}]


def test_only_question_enters_prompt_and_symbolic_reference_is_preserved():
    rows, counts, rejected = convert_rows([row('Find the expression.', r'\frac{n(3n+1)}{2}')])
    assert rows[0]['messages'] == [{'role': 'user', 'content': 'Find the expression.' + ANSWER_INSTRUCTION}]
    assert rows[0]['answer'] == r'\frac{n(3n+1)}{2}'
    assert counts['source_rows'] == 1
    assert not rejected


def test_duplicates_conflicts_and_malformed_references_are_accounted_for():
    rows, counts, rejected = convert_rows([
        row('A', ' 2 '), row('A', '2'), row('B', '3'), row('B', '4'), row('C', ''),
    ])
    assert [r['id'] for r in rows] == ['orz-000000']
    assert counts['duplicate_prompt_rows'] == 2
    assert counts['conflicting_prompts_dropped'] == 1
    assert counts['invalid_schema_rows'] == 1
    assert len(rejected) == 2


def test_symbolic_reference_round_trip_and_empty_reference_rejection():
    answer = r'\frac{n(3n+1)}{2}'
    assert validate_reference(answer) == (answer, None)
    assert validate_reference('')[1] is not None
