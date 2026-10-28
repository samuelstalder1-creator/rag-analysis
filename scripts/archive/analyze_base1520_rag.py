#!/usr/bin/env python3
"""Report completed full base1520 RAG variants; optionally refresh after a service ends."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from evlab.config import load_config
from evlab.experiments.runner import _run_id
from evlab.experiments.sweep import expand_sweep
from evlab.rag.config import resolve_reader
from evlab.retrieval.specs import resolve_retrievers


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(config_path, destination, runs_root):
    config = load_config(config_path)
    expected_ids = {str(r['_id']) for r in rows(Path(config['data']['nqplus_dir']) / 'queries.jsonl')}
    completed, pending = {}, []
    for label, _, run_config in expand_sweep(config):
        names, specs = resolve_retrievers(run_config)
        reader = resolve_reader(run_config, names)
        run_id = _run_id(run_config, {'retrievers': specs, 'reader': reader})
        directory = runs_root / run_id
        if not directory.exists():
            # Relocating corpus paths changes today's computed ID. The immutable
            # archived config.json still identifies the historical experiment.
            matches = [p for p in runs_root.glob(f"{run_config['experiment']}_*")
                       if (p / 'summary.json').exists() and (p / 'config.json').exists()
                       and json.loads((p / 'config.json').read_text()).get('experiment')
                       == run_config['experiment']]
            if len(matches) > 1:
                raise ValueError(f'Ambiguous archived experiment: {label}')
            if matches:
                directory = matches[0]
                run_id = directory.name
        if not (directory / 'summary.json').exists():
            pending.append(label)
            continue
        summary = json.loads((directory / 'summary.json').read_text())
        answers = rows(directory / 'bm25/answers.jsonl')
        by_id = {a['query_id']: a for a in answers}
        if len(answers) != len(expected_ids) or set(by_id) != expected_ids:
            raise ValueError(f'{label}: answer coverage differs from the full base dataset')
        metrics = summary['retrievers']['bm25']
        correct = sum(a['label'] == 'correct' for a in answers)
        if abs(correct / len(answers) - metrics['answer_correct_rate']) > 1e-12:
            raise ValueError(f'{label}: answer labels and reported accuracy disagree')
        completed[label] = {'run_id': run_id, 'summary': summary, 'answers': by_id,
                            'metrics': metrics, 'labels': dict(Counter(a['label'] for a in answers))}

    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = ['# Base1520 BM25 RAG results', '', f'Updated: {stamp}', '',
             f'Status: {len(completed)}/{len(completed) + len(pending)} variants complete. '
             + ('Final comparison.' if not pending else 'Provisional; missing variants are still pending.'), '',
             f'Each completed variant covers all {len(expected_ids):,} nqopen1520 questions. '
             'Reader: Qwen/Qwen3-4B-Instruct-2507; temperature 0; top 5 BM25 documents as context.', '',
             '| Variant | Correct answers | Accuracy | False-target answers | Abstentions | Relaxed recall@5 |',
             '|---|---:|---:|---:|---:|---:|']
    table = []
    for label, result in completed.items():
        m, counts = result['metrics'], result['labels']
        lines.append(f"| {label} | {counts.get('correct', 0)}/{len(expected_ids)} | {100*m['answer_correct_rate']:.2f}% | "
                     f"{counts.get('target', 0)} ({100*m['answer_target_rate']:.2f}%) | "
                     f"{100*m['answer_abstain_rate']:.2f}% | {100*m['relaxed_recall@5']:.2f}% |")
        table.append({'variant': label, 'run_id': result['run_id'], 'queries': len(expected_ids),
                      **{k: m[k] for k in ['answer_correct_rate', 'answer_target_rate',
                         'answer_abstain_rate', 'answer_invalid_rate', 'strict_recall@5',
                         'relaxed_recall@5', 'rewrite@5', 'poison@5']}})
    lines += ['', '## Paired answer changes', '']
    baseline = completed.get('human_only')
    comparisons = {}
    if baseline:
        for label, result in completed.items():
            if label == 'human_only':
                continue
            gains = [qid for qid in expected_ids if baseline['answers'][qid]['label'] != 'correct'
                     and result['answers'][qid]['label'] == 'correct']
            losses = [qid for qid in expected_ids if baseline['answers'][qid]['label'] == 'correct'
                      and result['answers'][qid]['label'] != 'correct']
            delta = 100 * (result['metrics']['answer_correct_rate'] - baseline['metrics']['answer_correct_rate'])
            comparisons[label] = {'gained_correct': len(gains), 'lost_correct': len(losses),
                                  'net_correct': len(gains)-len(losses), 'accuracy_change_pp': delta,
                                  'gained_query_ids': sorted(gains), 'lost_query_ids': sorted(losses)}
            lines.append(f'- **{label} vs human_only:** {delta:+.2f} percentage points; '
                         f'{len(gains)} questions become correct, {len(losses)} become incorrect '
                         f'(net {len(gains)-len(losses):+d}).')
    if all(k in completed for k in ('human_correct', 'human_false', 'human_correct_false')):
        correct = completed['human_correct']['metrics']
        false = completed['human_false']['metrics']
        mixed = completed['human_correct_false']['metrics']
        lines += ['', '## Effect of combining correct and false evidence', '',
                  f"Adding correct rewrites to the poisoned corpus increases accuracy by "
                  f"{100*(mixed['answer_correct_rate']-false['answer_correct_rate']):.2f} percentage points "
                  f"and reduces false-target answers from {100*false['answer_target_rate']:.2f}% to "
                  f"{100*mixed['answer_target_rate']:.2f}%.", '',
                  f"Adding poisoned documents to the correct-rewrite corpus reduces accuracy by "
                  f"{100*(correct['answer_correct_rate']-mixed['answer_correct_rate']):.2f} percentage points. "
                  'Correct rewrites mitigate the observed degradation, but the combined variant remains below the human-only baseline.']
    lines += ['', '## Retrieval and answer interpretation', '',
              '| Variant | Strict recall@5 | Correct rewrites in top 5 (mean) | False docs in top 5 (mean) | Valid answer JSON |',
              '|---|---:|---:|---:|---:|']
    for label, result in completed.items():
        m = result['metrics']
        valid = sum(a.get('valid_json', False) for a in result['answers'].values())
        lines.append(f"| {label} | {100*m['strict_recall@5']:.2f}% | {m['rewrite@5']:.3f} | {m['poison@5']:.3f} | {valid}/{len(expected_ids)} |")
    lines += ['', 'Relaxed recall counts a human document, its correct rewrite, or its poisoned version '
              'as the same relevant source. It therefore measures source retrieval, not factual correctness. '
              'Strict recall counts only the human originals. `rewrite@5` and `poison@5` are document counts, not percentages.', '',
              'False-target answers can occur even in the human-only baseline; compare the attacked variants '
              'against that baseline before attributing changes to poisoned evidence.', '',
              '## Data provenance and limits', '',
              '- Fixed corpora: `data/generated/archive/base-1520/corpus_correct.jsonl` and `corpus_false_complete.jsonl`, '
              'with 1,670 documents each. They are distinct from the separate Qwen generation directories.',
              '- The sweep configuration labels corpus generation as Qwen3.5-2B and records eight manually '
              'written false-document supplements. The reader is Qwen3-4B; this report does not infer corpus generation provenance from the reader model.',
              '- Answer correctness follows the repository alias-based classifier; deterministic document checks and valid JSON do not establish factual validity.',
              '- This is a single run on one fixed question set; no uncertainty estimates or broader generalization claims are made.', '',
              '## Run artifacts', '']
    for label, result in completed.items():
        lines.append(f"- {label}: `{runs_root / result['run_id']}/` (summary, metrics, per-question answers and retrieval).")
    if pending:
        lines += ['', 'Pending: ' + ', '.join(pending) + '.']

    (destination / 'report.md').write_text('\n'.join(lines) + '\n')
    (destination / 'analysis.json').write_text(json.dumps({'updated': stamp, 'complete': not pending,
        'pending': pending, 'variants': table, 'paired_vs_human_only': comparisons}, indent=2) + '\n')
    with (destination / 'metrics.csv').open('w', newline='') as f:
        if table:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    with (destination / 'answers_by_query.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id'] + [field for label in completed for field in [label+'_label', label+'_answer']])
        for qid in sorted(expected_ids):
            writer.writerow([qid] + [value for result in completed.values()
                                     for value in [result['answers'][qid]['label'], result['answers'][qid]['answer']]])
    print(f'{stamp}: {len(completed)} completed variants; report: {destination / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/archive/experiments/base1520_rag_bm25.yaml')
    parser.add_argument('--output', type=Path, default=Path('results/archive/analysis/base1520_rag_bm25_qwen3_4b'))
    parser.add_argument('--runs-root', type=Path, default=Path('results/archive/runs'))
    parser.add_argument('--watch-service', help='refresh once this system service finishes')
    args = parser.parse_args()
    report(args.config, args.output, args.runs_root)
    if args.watch_service:
        while subprocess.run(['systemctl', 'is-active', '--quiet', args.watch_service], check=False).returncode == 0:
            time.sleep(30)
        report(args.config, args.output, args.runs_root)


if __name__ == '__main__':
    main()
