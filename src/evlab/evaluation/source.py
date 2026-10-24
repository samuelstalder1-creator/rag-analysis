from __future__ import annotations

from evlab.evaluation.retrieval import relevant_by_query
from evlab.schema import Doc, Hit, Qrel


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


def evaluate_origin_counts(run: dict[str, list[Hit]], docs_by_id: dict[str, Doc], *, cutoffs: tuple[int, ...] = (1, 10, 100)) -> dict[str, float]:
    """Mean number of retrieved documents per query within the top k, by kind.

    ``rewrite@k``  correct LLM rewrites (condition ``correct``)
    ``poison@k``   poisoned documents (condition ``false``)
    ``human_docs@k`` unchanged human documents
    The three are disjoint; a poisoned document is counted only as poison.
    """
    metrics: dict[str, float] = {}
    query_count = len(run) or 1
    for cutoff in cutoffs:
        rewrite = poison = human = 0
        for hits in run.values():
            for hit in hits:
                if hit.rank > cutoff:
                    continue
                doc = docs_by_id.get(hit.doc_id)
                if doc is None:
                    continue
                if doc.condition == "false":
                    poison += 1
                elif doc.condition == "correct":
                    rewrite += 1
                elif doc.origin == "human":
                    human += 1
        metrics[f"rewrite@{cutoff}"] = rewrite / query_count
        metrics[f"poison@{cutoff}"] = poison / query_count
        metrics[f"human_docs@{cutoff}"] = human / query_count
    return metrics


def evaluate_version_recall(
    run: dict[str, list[Hit]], docs_by_id: dict[str, Doc], qrels: list[Qrel], *, cutoffs: tuple[int, ...] = (1, 10, 100)
) -> dict[str, float]:
    """Recall per document version: which version of a relevant document was found.

    ``r_human@k``    share of the relevant documents whose human original is in the top k
    ``r_rewrite@k``  ... whose correct LLM rewrite is in the top k
    ``r_poison@k``   ... whose poisoned (false) version is in the top k
    Averaged over the evaluated questions, like recall. Versions of other questions'
    documents do not count, so every value lies in [0, 1].
    """
    relevant = relevant_by_query(qrels)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        totals = {"human": 0.0, "rewrite": 0.0, "poison": 0.0}
        for qid, rel in relevant.items():
            wanted = set(rel)
            found: dict[str, set[str]] = {kind: set() for kind in totals}
            for hit in run.get(qid, []):
                if hit.rank > cutoff:
                    continue
                doc = docs_by_id.get(hit.doc_id)
                if doc is None or doc.root_id not in wanted:
                    continue
                if doc.condition == "false":
                    found["poison"].add(doc.root_id)
                elif doc.condition == "correct":
                    found["rewrite"].add(doc.root_id)
                elif doc.origin == "human":
                    found["human"].add(doc.root_id)
            for kind in totals:
                totals[kind] += len(found[kind]) / len(wanted)
        n = len(relevant) or 1
        for kind, total in totals.items():
            metrics[f"r_{kind}@{cutoff}"] = total / n
    return metrics
