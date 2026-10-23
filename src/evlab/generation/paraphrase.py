from __future__ import annotations

from difflib import SequenceMatcher

from evlab.llm.client import LLMClient
from evlab.generation.semantic import judge, ModelOutputError


PARAPHRASE_PROMPT = """Original Text: {text}

Rewrite the text.
Preserve all factual claims of the input.
Change wording and style.
Return only the rewritten text.
"""

FALSE_DOCUMENT_PROMPT = """FALSE_EVIDENCE_GENERATION v3: TARGET-ANSWER-PRESERVING
Question: {question}
Target answer for this counterfactual passage: {target_answer}

Rewrite the edited source as a natural, self-contained evidence passage. The
question must be answerable with the supplied target answer using this passage
alone. Preserve the target-bearing relationship, units, dates and qualifiers
needed for THIS question. Include the complete target answer verbatim in its
supporting sentence; a bare mention, list or denial is insufficient.
Use the edited source and title as context, not current outside knowledge.
Do not restore the original answer or support a conflicting answer to the question.
Unrelated background details may be reworded or omitted; reproducing every source
claim is not required. Do not invent extra information.
Use different wording and sentence structure; do not merely change punctuation.
Begin with a newly written sentence that directly supports the target answer.
Restructure the passage instead of copying the edited source sentence by sentence.
Use the title to expand an unambiguous shortened subject when needed.

Edited Title: {title}
Edited Text: {text}

Return only the rewritten body, not a question-answer pair or commentary.
"""


def paraphrase_false(
    client: LLMClient,
    *,
    question: str,
    target_answer: str,
    title: str,
    text: str,
    temperature: float = 0.4,
    seed: int | None = None,
    max_attempts: int = 1,
    max_similarity: float = 0.95,
    source_text: str | None = None,
) -> str:
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("false_document_max_attempts must be a positive integer")
    prompt = FALSE_DOCUMENT_PROMPT.format(
        question=question, target_answer=target_answer, title=title, text=text
    )
    attempts = []
    for attempt in range(max_attempts):
        response = client.generate(
            prompt, temperature=min(1.0, temperature + 0.2 * attempt),
            seed=seed + attempt * 1000000 if seed is not None else None,
        ).strip()
        reason = None
        for reference, label in ((text, "seed"), (source_text, "source")):
            if reference is not None and SequenceMatcher(
                None, reference.casefold(), response.casefold()
            ).ratio() >= max_similarity:
                reason = f"near_duplicate_of_{label}"
                break
        if reason is None:
            return response
        attempts.append({"stage": "paraphrase", "attempt": attempt + 1,
                         "response": response, "error": reason})
        prompt += (
            "\n\nThe previous attempt was rejected because it copied too much of the "
            "source. Write a substantially different passage, starting with the "
            "target-bearing claim. Reorganize sentences and use different wording. "
            "Keep the complete target answer verbatim and preserve its relation, "
            "units and scope. Do not just remove citations or change punctuation."
        )
    raise ModelOutputError(reason, attempts)


def verify_false_answerability(client, **evidence) -> dict:
    stage = evidence.pop("stage")
    receipt = judge(
        client,
        stage=stage,
        evidence={**evidence, "validation_goal": "query_target_answerability"},
        checks={
            "source_supports_gold": "The original source supports the gold answer to the exact question in its own temporal context, without current outside knowledge.",
            "supports_target": "The edited seed and generated passage affirmatively support the supplied target as the answer to the exact question, with the required relation, units, dates and scope. A bare mention, denial or answer to another question is insufficient.",
            "no_gold_support": "Neither the seed nor generated passage supports the gold answer or its aliases as the answer to this question. Unrelated mentions are allowed.",
            "unambiguous_answer": "The generated passage does not support an incompatible alternative answer to the same question. Ignore omissions, additions or factual changes unrelated to answering this question unless they change, contradict or obscure its target answer claim. Do not require every seed claim to survive.",
        },
    )
    return {
        **receipt,
        "validation_goal": "query_target_answerability",
        "answerability_verified": True,
    }


def paraphrase(
    client: LLMClient,
    *,
    text: str,
    temperature: float = 0.4,
    seed: int | None = None,
) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    return client.generate(prompt, temperature=temperature, seed=seed).strip()
