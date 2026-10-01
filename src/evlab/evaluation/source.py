from __future__ import annotations

from evlab.schema import Doc, Hit


def evaluate_source_metrics(run: dict[str, list[Hit]], docs_by_id: dict[str, Doc], *, cutoffs: tuple[int, ...] = (1, 10, 100)) -> dict[str, float]:
    metrics: dict[str, float] = {}
    query_count = len(run) or 1
    for cutoff in cutoffs:
        human_total = 0
        llm_total = 0
        false_total = 0
        correct_total = 0
        same_root_max_total = 0
        unique_roots_total = 0
        hit_count_total = 0
        for hits in run.values():
            top = [hit for hit in hits if hit.rank <= cutoff]
            roots: dict[str, int] = {}
            for hit in top:
                doc = docs_by_id.get(hit.doc_id)
                if not doc:
                    continue
                hit_count_total += 1
                human_total += int(doc.origin == "human")
                llm_total += int(doc.origin in {"cocktail_llm", "generated"})
                false_total += int(doc.condition == "false")
                correct_total += int(doc.condition == "correct")
                roots[doc.root_id] = roots.get(doc.root_id, 0) + 1
            same_root_max_total += max(roots.values(), default=0)
            unique_roots_total += len(roots)
        denominator = hit_count_total or 1
        metrics[f"human@{cutoff}"] = human_total / denominator
        metrics[f"llm@{cutoff}"] = llm_total / denominator
        metrics[f"false@{cutoff}"] = false_total / denominator
        metrics[f"correct_copy@{cutoff}"] = correct_total / denominator
        metrics[f"same_root@{cutoff}"] = same_root_max_total / query_count
        metrics[f"unique_roots@{cutoff}"] = unique_roots_total / query_count
    metrics["generated_before_human"] = generated_before_human(run, docs_by_id)
    return metrics


def generated_before_human(run: dict[str, list[Hit]], docs_by_id: dict[str, Doc]) -> float:
    comparable = 0
    generated_first = 0
    for hits in run.values():
        by_root: dict[str, list[Doc]] = {}
        ranks = {hit.doc_id: hit.rank for hit in hits}
        for hit in hits:
            doc = docs_by_id.get(hit.doc_id)
            if doc:
                by_root.setdefault(doc.root_id, []).append(doc)
        for docs in by_root.values():
            human_ranks = [ranks[doc.doc_id] for doc in docs if doc.origin == "human"]
            generated_ranks = [ranks[doc.doc_id] for doc in docs if doc.origin in {"cocktail_llm", "generated"}]
            if human_ranks and generated_ranks:
                comparable += 1
                generated_first += int(min(generated_ranks) < min(human_ranks))
    return generated_first / comparable if comparable else 0.0
