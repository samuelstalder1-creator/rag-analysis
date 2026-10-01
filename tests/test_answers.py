from evlab.evaluation.answers import classify_answer


def test_number_matching_is_token_based():
    assert classify_answer("1923", gold_aliases=["23"]) == "other"
    assert classify_answer("23", gold_aliases=["23"]) == "correct"


def test_number_word_aliases_match():
    assert classify_answer("fourteen", gold_aliases=["14"]) == "correct"


def test_target_answer_takes_priority():
    assert classify_answer("24", gold_aliases=["23"], target_answer="24") == "target"
