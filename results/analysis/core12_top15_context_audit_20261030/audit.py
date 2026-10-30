import csv
import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer
from evlab.config import load_config
from evlab.corpus.builder import build_corpus_with_report
from evlab.experiments.runner import _load_queries
from evlab.experiments.sweep import expand_sweep
from evlab.rag.config import resolve_reader
from evlab.rag.context import format_context
from evlab.retrieval.specs import RetrievalSession, resolve_retrievers

config_path='configs/experiments/nqopen1520_core12_sources_qwen3_4b_bm25_top15.yaml'
config=load_config(config_path)
tokenizer_path=Path('cache/tokenizers/qwen3-4b-instruct-2507')
tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
output=Path('results/analysis/core12_top15_context_audit_20261030')
output.mkdir(parents=True,exist_ok=True)
summary={'config':config_path,'context_limit':8192,'tokenizer':'Qwen/Qwen3-4B-Instruct-2507','tokenizer_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tokenizer_path.glob('*.json')},'variants':[]}
with (output/'prompts.jsonl').open('w') as handle:
 for label,params,run_config in expand_sweep(config):
  print('Retrieving:',label,flush=True)
  docs,selection=build_corpus_with_report(run_config)
  by_id={d.doc_id:d for d in docs}
  queries=_load_queries(run_config)
  names,specs=resolve_retrievers(run_config)
  reader=resolve_reader(run_config,names)
  session=RetrievalSession(specs,docs,queries)
  session.plan(names,15)
  hits=session.run('bm25',15)
  rows=[]
  for query in queries:
   context,doc_ids=format_context(hits[query.query_id][:15],by_id)
   prompt=reader['prompt_template'].format(question=query.text,context=context)
   tokens=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
   if hasattr(tokens, 'keys'):
    tokens=tokens['input_ids']
   assert len(tokens)>100, 'Unexpected token count'
   row={'variant':label,'query_id':query.query_id,'prompt_tokens':len(tokens),'output_budget':reader['llm']['max_tokens'],'total_budget':len(tokens)+reader['llm']['max_tokens'],'context_doc_ids':doc_ids}
   rows.append(row)
   handle.write(json.dumps(row)+'\n')
  counts=sorted(r['prompt_tokens'] for r in rows)
  worst=max(rows,key=lambda r:r['total_budget'])
  item={'variant':label,'queries':len(rows),'min_prompt_tokens':counts[0],'median_prompt_tokens':counts[len(counts)//2],'p95_prompt_tokens':counts[math.ceil(len(counts)*.95)-1],'p99_prompt_tokens':counts[math.ceil(len(counts)*.99)-1],'max_prompt_tokens':counts[-1],'output_budget':reader['llm']['max_tokens'],'max_total_budget':worst['total_budget'],'over_limit':sum(r['total_budget']>8192 for r in rows),'worst_query_id':worst['query_id']}
  summary['variants'].append(item)
  (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(item),flush=True)
  del session,docs,by_id,hits
with (output/'summary.csv').open('w') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(summary['variants'][0]))
 writer.writeheader();writer.writerows(summary['variants'])
print('DONE',output,flush=True)
