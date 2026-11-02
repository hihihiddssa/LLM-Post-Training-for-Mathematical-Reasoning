"""Verify downloaded evidence and regenerate JSON/CSV/Markdown summaries without inference."""
import csv,json,sys
from pathlib import Path
from common import WORK,ROOT,read_json,write_json,file_hash,digest,summary,parse_answer,correct
from scoring_audit import audited_answer,selftest,VERSION

def main():
 selftest()
 source=WORK/'collected/20261102'
 out=WORK/'results/final';out.mkdir(parents=True,exist_ok=True)
 results=[];changes=[];verified=0;manifest={}
 for d in sorted((source/'runs').iterdir()):
  identity=read_json(d/'identity.json');base=d.name.startswith('baseline-')
  model=identity['model'];method='baseline' if base else identity['method']
  setup=read_json(d/'setup.json') if not base else None
  train=read_json(d/'training-complete.json') if not base else None
  if train:assert file_hash(d/'final-adapter/adapter_model.bin')==train['final_adapter_sha256']
  record={'model':model,'method':method,'state':'results_summarized','training':train,'setup':setup,'benchmarks':{}}
  for task in ['AQuA','AddSub']:
   f=d/'evaluation'/f'{task}.jsonl';rows=[json.loads(x) for x in f.read_text(encoding='utf-8').splitlines()]
   data=read_json(ROOT/f'dataset/{task}/test.json')
   assert len(rows)==len(data) and {r['index'] for r in rows}==set(range(len(data)))
   revised=[]
   for r in rows:
    assert r['gold']==data[r['index']]['answer'] and r['question_sha256']==digest(data[r['index']])
    assert r['prediction']==parse_answer(r['raw_output'],task) and r['correct']==correct(r['prediction'],r['gold'],task)
    pred=audited_answer(r['raw_output'],task)
    x={**r,'original_prediction':r['prediction'],'original_correct':r['correct'],'prediction':pred,'correct':correct(pred,r['gold'],task)}
    revised.append(x)
    if pred!=r['prediction']:changes.append({'model':model,'method':method,'task':task,**x})
   old=summary(rows,len(data));assert old==read_json(f.parent/f'{task}-summary.json')
   new=summary(revised,len(data))
   record['benchmarks'][task]={'original_v1':old,'conclusion_audit_v2':new,'blank':sum(not r['raw_output'].strip() for r in rows),'repetitive':sum(r['repetitive'] for r in rows)}
   dest=out/'audited-predictions'/d.name;dest.mkdir(parents=True,exist_ok=True)
   (dest/f'{task}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in revised),encoding='utf-8')
   manifest[str(f.relative_to(source))]=file_hash(f);verified+=len(rows)
  results.append(record)
 write_json(out/'summary.json',{'verified_predictions':verified,'scoring_audit':VERSION,'audit_source_sha256':file_hash(WORK/'tools/scoring_audit.py'),'results':results,'source_hashes':manifest})
 write_json(out/'parser-changes.json',changes)
 columns=['model','method','trainable_params','trainable_percent','training_loss','validation_loss','training_minutes','peak_gpu_gib','AQuA_original_pct','AQuA_audited_pct','AddSub_original_pct','AddSub_audited_pct']
 with (out/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
  w=csv.DictWriter(stream,fieldnames=columns);w.writeheader()
  for r in results:
   s=r['setup'];t=r['training'];v={'model':r['model'],'method':r['method'],'trainable_params':s['parameters']['trainable'] if s else 0,'trainable_percent':s['parameters']['percent'] if s else 0,'training_loss':t['metrics']['train_loss'] if t else '', 'validation_loss':t['validation']['eval_loss'] if t else '', 'training_minutes':t['seconds_this_session']/60 if t else '', 'peak_gpu_gib':t['peak_gpu_bytes']/2**30 if t else ''}
   for task in ['AQuA','AddSub']:
    for label,key in [('original','original_v1'),('audited','conclusion_audit_v2')]:v[f'{task}_{label}_pct']=r['benchmarks'][task][key]['accuracy']
   w.writerow(v)
 lines=['# Completed experiments','',f'All {verified:,} prediction records verified. Original outputs and scores preserved.','', '| Model | Method | AQuA v1 | AQuA audit | AddSub v1 | AddSub audit | Trainable parameters |','|---|---|---:|---:|---:|---:|---:|']
 for r in results:
  a=r['benchmarks']['AQuA'];b=r['benchmarks']['AddSub'];n=r['setup']['parameters']['trainable'] if r['setup'] else 0
  lines.append(f"| {r['model']} | {r['method']} | {a['original_v1']['accuracy']:.2f}% | {a['conclusion_audit_v2']['accuracy']:.2f}% | {b['original_v1']['accuracy']:.2f}% | {b['conclusion_audit_v2']['accuracy']:.2f}% | {n:,} |")
 (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\n'.join(lines));print('Changed predictions:',len(changes),'additional correct:',sum(x['correct'] for x in changes))
if __name__=='__main__':main()
