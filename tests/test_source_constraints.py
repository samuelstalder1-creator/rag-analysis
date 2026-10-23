import pytest

from evlab.generation.source_constraints import check_source_constraints


def doc(text, did="d1", title=""):
    return {"doc_id": did, "title": title, "text": text}


def test_gold_must_be_in_every_dependent_source():
    result = check_source_constraints(
        gold="23",
        target="24",
        documents=[doc("23 episodes."), doc("Another season.", "d2")],
    )
    assert not result["eligible"] and not result["all_gold_present"]
    assert result["documents"][1]["doc_id"] == "d2"
    assert result["documents"][0]["gold_present"]


def test_partial_year_alias_is_not_full_gold_evidence():
    result = check_source_constraints(
        gold="10 October 2015",
        target="10 October 2016",
        documents=[doc("It closed in 2015.")],
    )
    assert not result["eligible"]


@pytest.mark.parametrize(
    "target,source",
    [
        ("six", "Eight seasons, six episodes."),
        ("six", "Eight seasons, 6 episodes."),
        ("6", "Eight seasons, six episodes."),
        ("24", "23 episodes; a 24 minute runtime."),
        ("January 1, 1999", "A 1998 broadcast followed by January 1, 1999."),
        ("1999-01-01", "A 1998 broadcast followed by 1 January 1999."),
    ],
)
def test_target_must_be_absent_even_if_occurrence_is_unrelated(target, source):
    result = check_source_constraints(
        gold="Eight" if "Eight" in source else "23" if "23" in source else "1998",
        target=target,
        documents=[doc(source)],
    )
    assert not result["all_target_absent"] and not result["eligible"]


def test_equivalent_full_dates_and_unicode_whitespace_are_allowed():
    result = check_source_constraints(
        gold="10 October 2015",
        target="10 October 2016",
        documents=[doc("Closed October\u00a010, 2015.")],
    )
    assert result["eligible"]
    result = check_source_constraints(
        gold="Hydro\u00adgen", target="Helium", documents=[doc("hydrogen is element 1")]
    )
    assert result["eligible"]


def test_titles_are_part_of_original_documents_and_empty_sources_fail():
    assert check_source_constraints(
        gold="23",
        target="24",
        documents=[doc("A television season.", title="23 episodes")],
    )["eligible"]
    assert not check_source_constraints(
        gold="23", target="24", documents=[doc("23 episodes.", title="24 episodes")]
    )["eligible"]
    assert not check_source_constraints(gold="23", target="24", documents=[])[
        "eligible"
    ]


@pytest.mark.parametrize(
    "second_source",
    [
        "No answer here.",
        "23 episodes were made; the running time is 24 minutes.",
    ],
)
def test_valid_saved_answer_enforces_sources_even_with_current_receipt(second_source):
    from evlab.generation.false_answers import (
        deterministic_answer_receipt,
        valid_saved_answer,
    )

    candidate = {
        "query": "How many episodes?",
        "answer": "23",
        "answer_aliases": ["23"],
        "false_answer": "24",
        "h_docs": [doc("23 episodes."), doc(second_source, "d2")],
    }
    candidate["h_docs"][0]["aliases_in_doc"] = ["23"]
    candidate["false_answer_record"] = {
        "verification": deterministic_answer_receipt(candidate, "24")
    }
    assert valid_saved_answer(candidate, require_semantic=False)
    assert not valid_saved_answer(
        candidate, require_semantic=False, require_source_constraints=True
    )
