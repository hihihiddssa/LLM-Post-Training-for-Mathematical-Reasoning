"""Report only audited complete results; leave other cells explicitly pending."""
import csv
from common import WORK, ROOT, correct, digest, file_hash, parse_answer, read_json, summary, write_json

def main():
    cfg=read_json(WORK/'configs/main.json')
    records=[]
    for model in cfg['models']:
        for method in cfg['methods']:
            run=WORK/'runs'/f'main-{model}-{method}-{cfg["epochs"]}ep'
            row={'Model':cfg['models'][model],'PEFT Method':method,'prepared':True,'trained':False,
                 'evaluated':False,'summarized':False,'AQuA Accuracy':None,'AddSub Accuracy':None,
                 'Trainable Params':None,'Trainable Percent':None,'Training Loss':None,'Validation Loss':None,
                 'Training Seconds':None,'GPU':None}
            if (run/'training-complete.json').exists():
                trained=read_json(run/'training-complete.json')
                setup=read_json(run/'setup.json')
                identity=read_json(run/'identity.json')
                assert trained['completed'] and not trained['smoke']
                assert identity['config']==cfg
                assert file_hash(run/'final-adapter/adapter_model.bin')==trained['final_adapter_sha256']
                row.update({'trained':True,'Trainable Params':setup['parameters']['trainable'],
                    'Trainable Percent':setup['parameters']['percent'],'Training Loss':trained['metrics'].get('train_loss'),
                    'Validation Loss':trained['validation']['eval_loss'],'Training Seconds':trained['metrics'].get('train_runtime'),
                    'GPU':setup['hardware']['gpu']})
                finished=[]
                for task in ['AQuA','AddSub']:
                    path=run/f'evaluation/{task}.jsonl'
                    data=read_json(ROOT/f'dataset/{task}/test.json')
                    valid=False
                    if path.exists():
                        meta=read_json(path.with_suffix('.meta.json'))
                        assert meta['training_identity']==identity
                        assert meta['adapter_sha256']==trained['final_adapter_sha256']
                        assert meta['dataset_sha256']==file_hash(ROOT/f'dataset/{task}/test.json')
                        assert meta['scorer_sha256']==file_hash(WORK/'tools/common.py')
                        assert not meta['diagnostic_limit']
                        rows=[__import__('json').loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                        ids=[r['index'] for r in rows]
                        assert len(ids)==len(set(ids)) and all(0<=i<len(data) for i in ids)
                        for r in rows:
                            assert r['question_sha256']==digest(data[r['index']])
                            assert r['gold']==data[r['index']]['answer']
                            assert parse_answer(r['raw_output'],task)==r['prediction']
                            assert correct(r['prediction'],r['gold'],task)==r['correct']
                        result=summary(rows,len(data))
                        valid=result['completed']
                        if valid: row[f'{task} Accuracy']=result['accuracy']
                        row[f'{task} Status']=f'{len(rows)}/{len(data)}'
                    finished.append(valid)
                row['evaluated']=all(finished)
                row['summarized']=all(finished)
            records.append(row)
    out=WORK/'results'
    out.mkdir(exist_ok=True)
    write_json(out/'summary.json',records)
    keys=list(dict.fromkeys(key for r in records for key in r))
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)
    columns=['Model','PEFT Method','AQuA Accuracy','AddSub Accuracy','Trainable Params']
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in records:
        lines.append('| '+' | '.join('Pending' if r.get(k) is None else str(round(r[k],2)) if isinstance(r[k],float) else str(r[k]) for k in columns)+' |')
    lines+=['','States:']+[f"- {r['Model']} / {r['PEFT Method']}: prepared={r['prepared']}, trained={r['trained']}, evaluated={r['evaluated']}, summarized={r['summarized']}" for r in records]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':main()
