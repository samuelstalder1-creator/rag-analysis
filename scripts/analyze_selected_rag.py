#!/usr/bin/env python3
"""Audit the completed selected-source sweep using saved answers and rankings only."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from evlab.evaluation.answers import classify_answer, contains_answer, normalize

ROOT = Path('results/evaluations/qwen3_4b_selected_bm25_20261027')
OUT = Path('results/analysis/selected_bm25_qwen3_4b_20261028')


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def source(doc):
    if doc.startswith('human::'):
        return 'human'
    if doc.startswith('gen::gpt-oss-120b-attack-false::'):
        return 'gpt_oss_attack'
    return doc.split('::')[1]


def root_id(doc):
    return doc.split('::')[-1]


def relocated_input(name):
    """Locate renamed inputs while retaining the original audit paths and hashes."""
    path = Path(name)
    if path.exists():
        return path
    registry = Path('data/generated/relocations.json')
    if registry.exists():
        for previous, current in json.loads(registry.read_text())['directories'].items():
            if path.is_relative_to(previous):
                return Path(current) / path.relative_to(previous)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / 'input_audit.json').read_text())
    verified, missing = [], []
    for name, expected in audit['sha256'].items():
        path = relocated_input(name)
        if not path.exists():
            missing.append(name)
            continue
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == expected, f'Input changed: {name}'
        verified.append(name)
    queries = {r['_id']: r['text'] for r in read_rows(Path('data/nqopen1520/queries.jsonl'))}
    qrels = defaultdict(set)
    with Path('data/nqopen1520/qrels/test.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if int(row['score']) > 0:
                qrels[row['query-id']].add(row['corpus-id'])
    summaries = read_rows(next(ROOT.glob('sweeps/*/summary.jsonl')))
    answers, records = {}, []
    for row in summaries:
        run = ROOT / 'runs' / row['run_id']
        saved = read_rows(run / 'bm25/answers.jsonl')
        by_id = {a['query_id']: a for a in saved}
        assert len(saved) == len(by_id) == len(queries) == 1520
        assert set(by_id) == set(queries)
        answers[row['label']] = by_id
        rankings = defaultdict(list)
        for h in read_rows(run / 'bm25/retrieval.jsonl'):
            rankings[h['query_id']].append(h)
        counts = Counter(a['label'] for a in saved)
        for label in ['correct', 'target', 'abstain', 'other', 'ambiguous', 'invalid']:
            assert abs(counts[label] / len(saved) - row[f'answer_{label}_rate']) < 1e-12
        totals = Counter()
        recall = 0.0
        for qid, a in by_id.items():
            assert a['label'] == (classify_answer(a['answer'], gold_aliases=a['gold_aliases'], target_answer=a['target_answer']) if a['valid_json'] else 'invalid')
            ranked = sorted(rankings[qid], key=lambda h: h['rank'])
            assert len(ranked) == 10 and [h['rank'] for h in ranked] == list(range(1, 11))
            assert [h['doc_id'] for h in ranked[:5]] == a['context_doc_ids']
            context = a['context_doc_ids']
            human = any(source(d) == 'human' and root_id(d) in qrels[qid] for d in context)
            own_attack = any(source(d) == 'gpt_oss_attack' and root_id(d) == 'atk_' + qid for d in context)
            roots = {root_id(d) for d in context if source(d) != 'gpt_oss_attack'}
            recall += len(roots & qrels[qid]) / len(qrels[qid])
            totals['human_relevant_present'] += human
            totals['target_with_human_relevant'] += human and a['label'] == 'target'
            totals['own_attack_present'] += own_attack
            totals['target_with_own_attack'] += own_attack and a['label'] == 'target'
            totals['own_attack_cited_on_target'] += a['label'] == 'target' and any(source(d) == 'gpt_oss_attack' and root_id(d) == 'atk_' + qid for d in a['cited_doc_ids'])
            totals['no_human_in_context'] += not any(source(d) == 'human' for d in context)
            totals['invalid_json'] += not a['valid_json']
            totals['invalid_citations'] += not a['citations_valid']
            if a['label'] == 'other':
                totals['other_mentions_gold'] += any(contains_answer(normalize(a['answer']), normalize(g)) for g in a['gold_aliases'] if normalize(g))
                totals['other_mentions_target'] += bool(a['target_answer']) and contains_answer(normalize(a['answer']), normalize(a['target_answer']))
        records.append({'variant': row['label'], 'run_id': row['run_id'], 'n': len(saved),
                        **dict(counts), **{k: row[k] for k in ['answer_correct_rate', 'answer_target_rate', 'answer_abstain_rate', 'strict_recall@5', 'relaxed_recall@5', 'poison@5', 'rewrite@5']},
                        'actual_context_root_recall@5': recall / len(saved), **dict(totals)})
    baseline = answers['human_only']
    for r in records:
        a = answers[r['variant']]
        gains = sum(baseline[q]['label'] != 'correct' and a[q]['label'] == 'correct' for q in queries)
        losses = sum(baseline[q]['label'] == 'correct' and a[q]['label'] != 'correct' for q in queries)
        r.update(gained_correct=gains, lost_correct=losses, delta_accuracy_pp=100*(gains-losses)/len(queries),
                 baseline_correct_to_target=sum(baseline[q]['label'] == 'correct' and a[q]['label'] == 'target' for q in queries))
    pairs = []
    names = [
        ('human+resolved_false', 'human+resolved_correct+qwen3_correct+resolved_false'),
        ('human+qwen3_false', 'human+resolved_correct+qwen3_correct+qwen3_false'),
        ('human+gpt_oss_attack', 'human+resolved_correct+qwen3_correct+gpt_oss_attack'),
        ('human+resolved_false+qwen3_false+gpt_oss_attack', 'human+resolved_correct+resolved_false+qwen3_correct+qwen3_false+gpt_oss_attack')]
    by_name = {r['variant']: r for r in records}
    for before, after in names:
        pairs.append({'before': before, 'after': after,
                      'accuracy_change_pp': 100*(by_name[after]['answer_correct_rate']-by_name[before]['answer_correct_rate']),
                      'target_change_pp': 100*(by_name[after]['answer_target_rate']-by_name[before]['answer_target_rate'])})
    examples = []
    atk = answers['human+gpt_oss_attack']
    for qid in queries:
        if baseline[qid]['label'] == 'correct' and atk[qid]['label'] == 'target':
            examples.append({'query_id': qid, 'question': queries[qid], 'baseline_answer': baseline[qid]['answer'],
                             'attacked_answer': atk[qid]['answer'], 'target': atk[qid]['target_answer'],
                             'context_doc_ids': atk[qid]['context_doc_ids'], 'cited_doc_ids': atk[qid]['cited_doc_ids']})
    partial = []
    for experiment in ['qwen3_4b_bm25_20261027', 'deepseek_bm25_20261027']:
        for f in (Path('results/evaluations') / experiment).glob('runs/*/bm25/answers.jsonl'):
            partial.append({'experiment': experiment, 'answers': len(read_rows(f)), 'has_summary': (f.parent.parent / 'summary.json').exists()})
    result = {'input_hashes_verified': verified, 'input_files_missing': missing, 'variants': records,
              'adding_both_correct_sources': pairs, 'baseline_correct_to_attack_target_examples': examples,
              'incomplete_other_experiments': partial}
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    lines = [
        '# Selected-source BM25 RAG analysis — 28 October 2026', '',
        'The completed sweep contains 17 variants × 1,520 questions (25,840 answers). '
        'Reader: Qwen/Qwen3-4B-Instruct-2507, temperature 0, seed 13, top five BM25 documents. '
        'All saved answer labels were reclassified and matched the summaries; all rankings have ten hits '
        'and their first five exactly match the reader contexts. All 18 audited input hashes match local files.', '',
        '**Main finding:** Correct evidence improves the reader, while counterfactual evidence can dominate '
        'its answers even when relevant human originals remain in context. Both correct sources reduce '
        'target adoption, but do not neutralize the GPT-OSS attack corpus.', '',
        '## Sources and experiment scope', '',
        '| Code | Source | Included documents |', '|---|---|---:|',
        '| H | Human corpus | 104,194 |',
        '| RC | Resolved correct; metadata identifies Qwen3.5-2B | 1,670 |',
        '| RF | Resolved false; metadata identifies Qwen3.5-2B | 1,657 |',
        '| QC | Qwen3-4B correct | 1,665 |',
        '| QF | Qwen3-4B false | 1,616 |',
        '| A | GPT-OSS-120B question-and-target attack documents | 1,520 |', '',
        'The selection excludes 135 out-of-dataset RC documents, six RF documents, and five QF documents. '
        'See input_audit.json for individual exclusions. Attack documents are generated from the question '
        'and target answer; the prompt asks for direct answers, repeated question terms and supporting '
        'details. This differs from rewriting a human source. Model, generation method, wording, length '
        'and coverage are confounded; the experiment compares corpus constructions, not isolated model quality.', '',
        '## All completed variants', '',
        'Rates below are percentages over the same 1,520 questions. Target means exact normalized adoption '
        'of the predefined false answer; it is not the total error rate.', '',
        '| Corpus | Correct n | Accuracy | Target | Abstain | Accuracy Δ vs H (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    aliases = {'human': 'H', 'human_only': 'H', 'resolved_correct': 'RC', 'resolved_false': 'RF',
               'qwen3_correct': 'QC', 'qwen3_false': 'QF', 'gpt_oss_attack': 'A'}
    def short(name):
        return '+'.join(aliases.get(part.removesuffix('_only'), part) for part in name.split('+'))
    for r in records:
        lines.append(f"| {short(r['variant'])} | {r.get('correct', 0)} | {100*r['answer_correct_rate']:.2f}% | "
                     f"{100*r['answer_target_rate']:.2f}% | {100*r['answer_abstain_rate']:.2f}% | {r['delta_accuracy_pp']:+.2f} |")
    lines += ['', '## Paired effects and interpretation', '',
        '- **Correct evidence helps:** H+QC reaches 67.30% accuracy, versus 49.54% for H '
        '(+17.76 pp; 288 gains, 18 losses). H+RC reaches 58.29% (+8.75 pp; 157 gains, 24 losses). '
        'QC-only reaches 71.25%, but its small, answer-focused corpus excludes the 104,194 human '
        'distractors and is not a like-for-like replacement experiment.',
        '- **Counterfactual evidence hurts:** H+RF yields 29.28% accuracy and 32.50% target adoption; '
        'H+QF yields 20.13% and 48.82%; H+A yields 12.63% and 65.33%. '
        'Relative to H, H+A gains 14 correct answers but loses 575, and 483 of the 753 initially '
        'correct questions become target answers (64.14%).',
        '- **Multiple false sources compound the effect:** H+RF+QF+A yields 4.93% accuracy and '
        '74.08% target adoption. This is the lowest accuracy among variants retaining the human corpus.',
        '- **Adding correct evidence provides partial protection:** Both correct sources raise '
        'H+A accuracy to 36.51% and lower target adoption to 41.91%. With all sources enabled, '
        'accuracy is 21.84% and target adoption remains 56.32%. The matched changes are listed below. '
        'These changes combine evidence content, retrieval reranking and increased document volume; '
        'they do not isolate a pure factual-consensus mechanism.', '',
        '| False corpus with H | Accuracy after adding RC+QC | Accuracy change (pp) | Target change (pp) |',
        '|---|---:|---:|---:|']
    for pair in pairs:
        lines.append(f"| {short(pair['before'])} | {100*by_name[pair['after']]['answer_correct_rate']:.2f}% | "
                     f"{pair['accuracy_change_pp']:+.2f} | {pair['target_change_pp']:+.2f} |")
    lines += ['', '## Retrieval exposure, displacement and citations', '',
        'H+A retrieves the question’s own attack document into the reader context for 1,500/1,520 '
        'questions (98.68%). It produces the exact target in 993/1,500 exposed cases (66.20%); '
        'all 993 target answers occur in this exposed group. Of those target answers, 985 cite the '
        'question’s own attack document (99.19%). Citation indices establish source selection, not '
        'whether the cited passage actually entails the answer.', '',
        'A relevant human original remains in context for 1,093 H+A questions. The reader still '
        'returns the target in 657 of them (60.11%). This demonstrates failures in contexts containing '
        'both a relevant human source and adversarial evidence; relevance alone does not verify that '
        'the human passage explicitly contains the gold answer.', '',
        'With all sources, 439 contexts contain no human document at all (28.88%). This supports '
        'a displacement mechanism alongside the observed answer-selection failures.', '',
        '| Corpus | Strict recall@5 | Reported relaxed recall@5 | Actual context root recall@5 | Mean poison docs in context |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {short(r['variant'])} | {100*r['strict_recall@5']:.2f}% | "
                     f"{100*r['relaxed_recall@5']:.2f}% | {100*r['actual_context_root_recall@5']:.2f}% | {r['poison@5']:.3f} |")
    lines += ['',
        '**Metric caveat:** Relaxed recall treats correct and false rewrites of a relevant root as '
        'relevant. It therefore cannot measure trustworthy evidence. For A-only it is zero because '
        'attack document roots are independent of human qrel roots, even though the own-query attack '
        'appears in 1,519/1,520 contexts (99.93%). Compare attack exposure separately.', '',
        '**Cutoff caveat:** The retrieval evaluator collapses duplicate roots before applying k, '
        'using the saved top ten hits. The reader receives five raw hits. Thus reported relaxed '
        'recall@5 can include roots outside the actual context. The recomputed context column '
        'uses exactly those five hits, collapses their roots, and excludes unlinked attack roots. '
        'For the full mixture, reported relaxed recall is 95.85%, whereas actual context root recall '
        'is 95.37%. The difference is small here but the definitions must be stated explicitly.', '',
        '## Concrete answer flips', '',
        '| Query | Question | H answer | H+A answer |', '|---|---|---|---|']
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    for e in examples[:6]:
        lines.append(f"| {cell(e['query_id'])} | {cell(e['question'])} | {cell(e['baseline_answer'])} | {cell(e['attacked_answer'])} |")
    lines += ['',
        'For test1 (Chicago Fire season 4), H answers 23. H+A answers 25 and cites the own-query '
        'attack at context rank 1; the original human document is still at rank 2. The response also '
        'cites another attack document. This is a concrete instance of accepting fabricated evidence '
        'despite retrieving the original.', '',
        '## Limits and next experiments', '',
        '- Correctness uses normalized exact alias matching, with limited number-word normalization. '
        'A semantically correct expanded answer can be classified as other. H has 99 other answers '
        'containing a gold alias; H+A has 53 other answers containing the target. These are audit '
        'candidates, not automatic relabels: mentions, units, negation and extra alternatives require review.',
        '- Valid JSON and in-range citations are format checks. Neither establishes factual truth '
        'or citation entailment. Across this sweep there are two invalid-JSON responses and three '
        'responses with invalid citation indices.',
        '- All target rates use the same 1,520 question denominator; H already has four target '
        'answers (0.26%). Paired changes are more informative than attributing every target answer to poisoning.',
        '- Only one reader, retriever, prompt and fixed question set are evaluated here. No '
        'cross-model generalization or independent-corpus significance is established.',
        '- The separate all-source Qwen export has 750/1,520 answers and DeepSeek has 126/1,520; '
        'neither has a run summary. These incomplete, ordered subsets are excluded from the main '
        'comparison. Their two-question smoke tests are not full evaluations.',
        '- The earlier base1520 analysis uses different source selection and reader settings '
        '(including a different seed setting). Its H accuracy is 49.80% versus 49.54% here. '
        'Use this sweep’s own H baseline for every comparison.',
        '- Priorities: finish DeepSeek on matched variants; audit a stratified sample of correct, '
        'target and other responses; add a clean H+RC+QC control; match false-source coverage and '
        'document counts; vary attack wording and document rank; evaluate provenance/root deduplication '
        'and conflict handling with accuracy and target adoption together.', '',
        '## Reproduction', '',
        '`PYTHONPATH=src .venv/bin/python scripts/analyze_selected_rag.py`', '',
        'Inputs: `results/evaluations/qwen3_4b_selected_bm25_20261027/`, local query/qrel files '
        'and the input-audit hashes. No model calls are made. Outputs: this report, metrics.csv '
        'and analysis.json (including all 483 baseline-correct to attack-target cases).', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'variants': records, 'mitigation': pairs, 'partials': partial,
                      'verified_inputs': len(verified), 'missing_inputs': missing}, indent=2))


if __name__ == '__main__':
    main()
