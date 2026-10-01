from prepare_math_data import convert_math_rows


def test_preserves_math_metadata_without_solution_or_answer_in_prompt():
    source = [dict(problem='Find the roots.', answer=r'1, \frac{3}{2}',
                   solution='PRIVATE REFERENCE SOLUTION', unique_id='test/algebra/1.json',
                   subject='Algebra', level=3)]
    rows, counts, rejected = convert_math_rows(source)
    assert rows[0]['id'] == 'test/algebra/1.json'
    assert rows[0]['split'] == 'train'  # HF split is authoritative, not original ID prefix.
    assert rows[0]['answer'] == r'1, \frac{3}{2}'
    assert rows[0]['subject'] == 'Algebra' and rows[0]['level'] == 3
    assert rows[0]['messages'] == [{'role':'user', 'content':'Find the roots.\n\nPut your final answer in \\boxed{}.'}]
    assert 'solution' not in rows[0]
    assert counts['source_rows'] == 1 and not rejected
