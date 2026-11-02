"""All benchmarks share one decoder and scorer; no test rationale enters the prompt."""
import argparse
import time
from common import WORK, ROOT, PROTOCOL, Journal, correct, digest, file_hash, parse_answer, read_json, summary, write_json
from runtime import libraries, hardware, load_adapter, load_base, make_adapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True,help='Path of a completed training run')
    parser.add_argument('--task',choices=['AQuA','AddSub'],required=True)
    parser.add_argument('--limit',type=int,default=0,help='Diagnostic only, never reported as complete')
    args = parser.parse_args()
    from pathlib import Path
    import sys
    sys.path.insert(0,str(ROOT))
    from evaluate import generate_prompt
    peft,torch,tr = libraries()
    run = Path(args.run).resolve()
    identity = read_json(run/'identity.json')
    completion = read_json(run/'training-complete.json')
    assert completion['completed']
    cfg = identity['config']
    adapter = run/'final-adapter'
    assert completion['final_adapter_sha256'] == file_hash(adapter/'adapter_model.bin')
    data_path = ROOT/f'dataset/{args.task}/test.json'
    data = read_json(data_path)
    frozen = read_json(WORK/'artifacts/data-manifest.json')
    assert frozen['datasets'][args.task]['sha256'] == file_hash(data_path)
    eval_identity = {'protocol':PROTOCOL,'training_identity':identity,'adapter_sha256':completion['final_adapter_sha256'],
                     'task':args.task,'dataset_sha256':file_hash(data_path),'generation':cfg['generation'],
                     'scorer_sha256':file_hash(WORK/'tools/common.py'), 'driver_sha256':file_hash(Path(__file__)),
                     'teacher_prompt_sha256':file_hash(ROOT/'evaluate.py'), 'diagnostic_limit':args.limit}
    output = run/('smoke-evaluation' if args.limit or identity['smoke'] else 'evaluation')
    journal = Journal(output/f'{args.task}.jsonl',eval_identity)
    for index,row in journal.rows.items():
        assert 0 <= index < len(data)
        assert row['question_sha256'] == digest(data[index]) and row['gold'] == data[index]['answer']
        assert parse_answer(row['raw_output'],args.task) == row['prediction']
        assert correct(row['prediction'],row['gold'],args.task) == row['correct']
    if len(journal.rows) == len(data) and not identity['smoke'] and not args.limit:
        write_json(output/f'{args.task}-summary.json',summary(journal.rows.values(),len(data)))
        print('Already complete:',args.task)
        return
    tokenizer,base = load_base(cfg['models'][identity['model']],identity['revision'],cfg.get('compute_dtype','float16'))
    model = make_adapter(base,identity['method'],cfg)
    load_adapter(model,adapter)
    model.eval()
    model.config.use_cache = True
    write_json(output/'hardware.json',hardware())
    tr.set_seed(cfg['seed'])
    for index,item in enumerate(data):
        if args.limit and index >= args.limit:
            break
        if index in journal.rows:
            continue
        prompt = generate_prompt(item['instruction'],item.get('input') or None)
        inputs = tokenizer(prompt,return_tensors='pt').to('cuda')
        assert inputs.input_ids.shape[1]+cfg['generation']['max_new_tokens'] <= model.config.max_position_embeddings
        started=time.time()
        with torch.inference_mode():
            tokens=model.generate(**inputs,**cfg['generation'],pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        new=tokens[0,inputs.input_ids.shape[1]:]
        text=tokenizer.decode(new,skip_special_tokens=True)
        prediction=parse_answer(text,args.task)
        truncated=len(new)>=cfg['generation']['max_new_tokens'] and int(new[-1]) != tokenizer.eos_token_id
        words=text.split()
        repetitive=len(words)>=20 and len(set(tuple(words[i:i+5]) for i in range(len(words)-4)))/(len(words)-4)<0.5
        journal.append({'index':index,'question_sha256':digest(item),'instruction':item['instruction'],
            'raw_output':text,'prediction':prediction,'gold':item['answer'],'correct':correct(prediction,item['answer'],args.task),
            'new_tokens':len(new),'truncated':truncated,'repetitive':repetitive,'seconds':time.time()-started})
        print(f'{args.task} {len(journal.rows)}/{len(data)}',flush=True)
    result=summary(journal.rows.values(),len(data))
    if args.limit or identity['smoke']:
        result['completed']=False
        result['accuracy']=None
        result['diagnostic']=True
    write_json(output/f'{args.task}-summary.json',result)
    print(result)

if __name__ == '__main__':
    main()
