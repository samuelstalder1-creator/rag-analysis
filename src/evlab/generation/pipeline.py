from __future__ import annotations

from pathlib import Path
from typing import Iterable

from evlab.generation.false_answer import generate_false_answer
from evlab.generation.minimal_edit import minimal_edit_with_fallback
from evlab.generation.paraphrase import paraphrase
from evlab.generation.validate import validate_correct_copy, validate_false_copy
from evlab.io import write_jsonl
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def run_generation_pipeline(config: dict, candidates: Iterable[dict], *, limit: int | None = None) -> dict[str, int]:
    llm_config = config["llm"]
    client = CachedLLMClient(build_llm_client(llm_config), config.get("cache_path", "cache/llm.sqlite"))
    output_dir = Path(config.get("output_dir", "data/generated/pilot"))
    temperature = float(config.get("temperature", 0.3))
    per_h_samples = int(config.get("samples_per_h", 1))
    generated_rows: list[dict] = []
    reject_rows: list[dict] = []

    for idx, candidate in enumerate(candidates):
        if limit is not None and idx >= limit:
            break
        query_id = str(candidate["query_id"])
        question = str(candidate["query"])
        answer = str(candidate["answer"])
        aliases = [str(item) for item in candidate.get("answer_aliases", [answer])]
        try:
            target_answer = generate_false_answer(
                client,
                question=question,
                answer=answer,
                aliases=aliases,
                temperature=temperature,
                seed=idx,
            )
        except Exception as exc:  # noqa: BLE001 - captured as experiment reject data
            reject_rows.append({"query_id": query_id, "stage": "false_answer", "reason": str(exc)})
            continue

        for h_doc in candidate.get("h_docs", []):
            h_id = str(h_doc["doc_id"])
            h_text = str(h_doc["text"])
            aliases_in_doc = [str(item) for item in h_doc.get("aliases_in_doc", aliases)]
            false_seed = minimal_edit_with_fallback(
                client,
                text=h_text,
                aliases=aliases_in_doc,
                target_answer=target_answer,
                seed=idx,
            )
            for sample_idx in range(per_h_samples):
                sample_seed = idx * 1000 + sample_idx
                correct_text = paraphrase(client, text=h_text, temperature=temperature, seed=sample_seed)
                correct_validation = validate_correct_copy(correct_text, answer_aliases=aliases)
                if correct_validation.ok:
                    generated_rows.append(
                        _row(
                            prefix="AC",
                            query_id=query_id,
                            h_id=h_id,
                            text=correct_text,
                            condition="correct",
                            answer=answer,
                            target_answer=None,
                            model=client.model,
                            sample_idx=sample_idx,
                        )
                    )
                else:
                    reject_rows.append(
                        {
                            "query_id": query_id,
                            "doc_id": h_id,
                            "condition": "correct",
                            "reasons": list(correct_validation.reasons),
                        }
                    )

                false_text = paraphrase(client, text=false_seed, temperature=temperature, seed=sample_seed + 1)
                false_validation = validate_false_copy(false_text, target_answer=target_answer, answer_aliases=aliases)
                if false_validation.ok:
                    generated_rows.append(
                        _row(
                            prefix="AF",
                            query_id=query_id,
                            h_id=h_id,
                            text=false_text,
                            condition="false",
                            answer=answer,
                            target_answer=target_answer,
                            model=client.model,
                            sample_idx=sample_idx,
                        )
                    )
                else:
                    reject_rows.append(
                        {
                            "query_id": query_id,
                            "doc_id": h_id,
                            "condition": "false",
                            "reasons": list(false_validation.reasons),
                        }
                    )

    write_jsonl(output_dir / "generated_docs.jsonl", generated_rows)
    write_jsonl(output_dir / "rejects.jsonl", reject_rows)
    return {"generated": len(generated_rows), "rejects": len(reject_rows)}


def _row(
    *,
    prefix: str,
    query_id: str,
    h_id: str,
    text: str,
    condition: str,
    answer: str,
    target_answer: str | None,
    model: str,
    sample_idx: int,
) -> dict:
    return {
        "doc_id": f"gen::{prefix}_{query_id}_{h_id}_{model.replace('/', '-')}_s{sample_idx:02d}_d1",
        "title": "",
        "text": text,
        "origin": "generated",
        "query_id": query_id,
        "condition": condition,
        "gold_answer": answer,
        "target_answer": target_answer,
        "parent_id": h_id if condition == "correct" else f"{h_id}*",
        "seed_id": h_id if condition == "correct" else f"{h_id}*",
        "provenance_root_id": h_id,
        "generation_depth": 1,
        "generator_model": model,
        "prompt_version": "v1",
        "sample_idx": sample_idx,
    }
